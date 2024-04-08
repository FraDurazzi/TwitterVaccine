#!/usr/bin/env python3
"""Starting from the graphs, compute the community structures."""

from __future__ import annotations

from time import time

import networkx as nx
import numpy as np
import pandas as pd
import pygenstability as stability
import sknetwork
import tqdm
from scipy import sparse

from build_graphs import DATAPATH, load_graph


def partition_core(
    tail: sparse.spmatrix,
    head: sparse.spmatrix,
    usermap: pd.Series,
    kind: str = "sk_louvain",
) -> pd.Series:
    """Compute the partition of the core of the graph.

    Remove dangling nodes and compute the community structure.
    Add the dangling nodes to the neighboring community.
    """
    adjacency = (tail @ head.T).tocsr()

    # take the binary version
    # this is needed to test weather a user tweeted to or retweeted from
    # only one other user
    adjacency.data = np.ones(adjacency.nnz)
    retweeters = adjacency.sum(0).A
    tweeters = adjacency.sum(1).A.T

    # users that either have no tweets and retweets just one other user or
    # have tweets that get retweeted only from another user and no retweets.
    # In term of topology, these are periphery nodes with either one outgoing or ingoing link
    periphery = np.logical_or(
        np.logical_and(retweeters == 0, tweeters == 1),
        np.logical_and(retweeters == 1, tweeters == 0),
    )
    # list of indices of adjacency matrix:
    core = np.argwhere(~periphery)[:, 1]
    periphery = np.argwhere(periphery)[:, 1]
    n_nodes = tail.shape[0]
    n_p = len(periphery)
    n_c = len(core)

    # projector to the periphery subset of nodes
    # shape: N x N_p
    proj_periphery = sparse.coo_matrix(
        (np.ones_like(periphery), (periphery, np.arange(len(periphery)))),
        shape=(n_nodes, n_p),
    ).tocsr()

    # projector to the core subset of nodes
    # shape: N x N_c
    proj_core = sparse.coo_matrix(
        (np.ones_like(core), (core, np.arange(len(core)))),
        shape=(n_nodes, n_c),
    ).tocsr()
    core_adj = proj_core.T @ (tail @ head.T).tocsr() @ proj_core

    # Extract all links connected to the periphery
    periphery_links = proj_periphery.T @ adjacency + (adjacency @ proj_periphery).T
    # link from core to periphery N_p x N_c
    core_periphery = periphery_links @ proj_core

    # compute the partition structure on the core of the network
    core_partition = partition(core_adj, kind=kind)

    if isinstance(core_partition, pd.DataFrame):
        for c in core_partition.columns:
            full_partition = pd.DataFrame(
                {
                    c: _hydrate_(
                        core_partition[c], core_periphery, proj_core, proj_periphery
                    )
                    for c in core_partition.columns
                },
                index=pd.Index(usermap, name="uid"),
            )
    else:
        full_partition = pd.Series(
            _hydrate_(core_partition, core_periphery, proj_core, proj_periphery),
            index=pd.Index(usermap, name="uid"),
        )
    return full_partition


def _hydrate_(
    core_partition: pd.Series,
    core_periphery: sparse.spmatrix,
    proj_core: sparse.spmatrix,
    proj_periphery: sparse.spmatrix,
) -> sparse.spmatrix:
    """From a list of labels compute the projector from nodes to community space.

    This function expects the network to be divided in two components:
        - a core
        - a periphery with nodes connected only to the core.

    It takes the core and pariphery parts and join them
    assigning the periphery nodes to its neighbor's community

    Parameters
    ----------
    core_partition: pd.Series
        partition of the nodes as a list of labels
    core_periphery: sparse.spmatrix
        adjacency matrix (only links between core and periphery)
    proj_core: sparse.spmatrix
        projector from nodes to node core
    proj_periphery: sparse.spmatrix
        projector from nodes to node periphery

    Returns
    -------
    component_projector: np.ndarray
        list of labels

    """
    # partition: N_c x N_comm
    n_c = len(core_partition)
    n_comm = core_partition.nunique()
    # projector from core nodes to the communities
    core_part_proj = sparse.coo_matrix(
        (
            np.ones_like(core_partition),
            (core_partition.index.to_numpy(), core_partition.to_numpy()),
        ),
        shape=(n_c, n_comm),
    ).tocsr()
    # assign the periphery nodes to the corresponding core node communities.
    # shape: N_p x N_comm
    periphery_part = core_periphery @ core_part_proj

    # partition of the whole adj
    # shape: N x N_comm
    full_partition = proj_core @ core_part_proj + proj_periphery @ periphery_part

    # compress the partition to a list of class indexes
    compressor = sparse.coo_matrix(np.arange(n_comm), shape=(1, n_comm)).tocsr()
    return (full_partition @ compressor.T).toarray().astype(int).squeeze()


def partition(
    adj: sparse.spmatrix, kind: str = "louvain", usermap: pd.Series | None = None
) -> pd.Series:
    """Compute partitions with various methods."""
    print("Computing", kind)

    t0 = time()
    if kind == "louvain":
        p = nx.community.greedy_modularity_communities(
            # use the undirected form.
            nx.from_scipy_sparse_array(
                adj, create_using=nx.Graph, edge_attribute="weight"
            ),
            weight="weight",
        )
    elif kind == "sk_louvain":
        # Much faster than networkx
        louvain = sknetwork.clustering.Louvain()
        p = louvain.fit_predict(adj)
        p = pd.Series(p, name="sk_louvain")
    elif kind == "leiden":
        # optimize modularity with leiden in igraph
        graph_ig = sparse2igraph(adj, directed=False)
        p = graph_ig.community_leiden(objective_function="modularity", weights="weight")
    elif kind == "infomap":
        graph_ig = sparse2igraph(adj)
        p = graph_ig.community_infomap(edge_weights="weight")
    elif kind == "fastgreedy":
        graph_ig = sparse2igraph(adj)
        p = graph_ig.community_fastgreedy(weights="weight").as_clustering()
    elif kind == "stability":
        transition, steadystate = compute_transition_matrix(
            adj + 0.1 * adj.T, niter=1000
        )
        stab = stability.run(
            transition @ sparse.diags(steadystate, offsets=0, shape=transition.shape),
            n_workers=15,
            tqdm_disable=False,
        )
        p = pd.DataFrame(
            {
                f"stab_{p_id}": stab["community_id"][p_id]
                for p_id in stab["selected_partitions"]
            }
        )

    if kind in {"infomap", "leiden", "fastgreedy"}:
        # convert igraph result to pd.Series
        p = {u: ip for ip, _p in enumerate(p) for u in _p}
        p = pd.Series(p.values(), index=p.keys())
    print("Elapsed time", time() - t0)
    print(p.value_counts())

    if usermap is not None:
        p.index = p.index.map(usermap)

    return p


def compute_transition_matrix(
    matrix: sparse.csr_matrix, niter: int = 10000
) -> tuple[sparse.spmatrix, sparse.spmatrix]:
    r"""Return the transition matrix.

    Parameters
    ----------
    matrix : sparse.spmatrix
        the adjacency matrix (square shape)
    niter : int (default=10000)
        number of iteration to converge to the steadystate. (Default value = 10000)

    Returns
    -------
    trans : np.spmatrix
        The transition matrix.
    v0 : np.matrix
        the steadystate

    """
    # marginal
    tot = matrix.sum(0).A1
    # fix zero division
    tot_zero = tot == 0
    tot[tot_zero] = 1
    # transition matrix
    trans = matrix @ sparse.diags(1 / tot)

    # fix transition matrix with zero-sum rows
    trans += sparse.diags(tot_zero.astype(np.float64), offsets=0, shape=trans.shape)

    v0 = matrix.sum(0) + 1
    # v0 = sparse.csr_matrix(np.random.random(matrix.shape[0]))
    v0 = v0.reshape(matrix.shape[0], 1) / v0.sum()
    trange = tqdm.trange(0, niter)
    for i in trange:
        # evolve v0
        v1 = v0.copy()

        v0 = trans.T @ v0
        diff = np.sum(np.abs(v1 - v0))
        if i % 100 == 0:
            trange.set_description(desc=f"diff: {diff}|", refresh=True)
        if diff < 1e-5:
            break
    print(f"TRANS: performed {i + 1} itertions. (diff={diff:2.5f})")

    return trans, v0.A1


def plot_comm_size(parts: pd.DataFrame) -> None:
    """Plot the community sizes."""
    from matplotlib import pyplot as plt

    fig, axs = plt.subplots(
        ncols=len(parts.columns), sharey=True, figsize=(3 * len(parts.columns), 5)
    )

    print(parts.nunique())
    for ax, part in zip(axs, parts.columns):
        print(part)
        counts = parts[part].value_counts()
        counts = counts.sort_values(ascending=False).cumsum()
        counts /= len(parts[part])
        ax.scatter(range(len(counts)), counts)
        ax.semilogx()
        ax.set_title(part)
        # ax.set_xlim(1, 20)
        ax.axhline(0.9)
    axs[0].set_ylabel("Cumulative ratio.")
    plt.savefig("plot_community_sizes.pdf")
    plt.close()


def simplify_community_struct(
    community: pd.Series, comm_size: int = 0, coverage: float = 0.0
) -> pd.Series:
    """Reduce the number of communities.

    The reduction can be done with:
    - a community size cutoff
    - a coverage ratio

    all the other (small) communities are merged together.
    """
    counts = community.value_counts().sort_values(ascending=False)
    if comm_size > 0:
        # remove communities smaller than comm_size
        keep = counts[counts > comm_size].index
    elif coverage > 0.0:
        # keep larger communities that cover at least coverage ratio of the network
        keep = counts.cumsum() / len(community)
        keep_num = len(keep[keep <= coverage]) + 1
        keep = keep.iloc[:keep_num].index

    new_parts = {v: i for i, v in enumerate(keep)}
    return community.map(lambda x: new_parts.get(x, len(new_parts)))


def sparse2igraph(adjacency: sparse.spmatrix, **kwargs: dict) -> igraph.Graph:
    """Convert to igraph."""
    import igraph

    i, j, v = sparse.find(adjacency)
    graph = igraph.Graph(edges=zip(i, j), **kwargs)
    graph.es["weight"] = v
    return graph


def main(deadline: str) -> None:
    """Do the main."""
    tail, head, usermap = load_graph(deadline)
    print("N users =", tail.shape[0])
    print("N edges =", head.nnz, tail.nnz)
    p = pd.DataFrame()

    pp = partition_core(tail, head, usermap, kind="leiden")
    p["leiden"] = pp
    p.index = pp.index

    p["louvain"] = partition_core(tail, head, usermap, kind="sk_louvain")

    # Infomap produce very small communities
    # p["infomap"] = partition_core(tail, head, usermap, kind="infomap")

    pstab = partition_core(tail, head, usermap, kind="stability")
    for c in pstab.columns:
        p[c] = pstab[c]

    for part in p.columns:
        p[part + "_90"] = simplify_community_struct(p[part], coverage=0.9)

    p.to_csv(DATAPATH / f"communities_{deadline}.csv.gz")


if __name__ == "__main__":
    for deadline in ["test", "2021-06-01"]:
        main(deadline)
