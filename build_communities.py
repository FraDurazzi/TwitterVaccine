#!/usr/bin/env python3
"""Starting from the graphs, compute the community structures."""

from __future__ import annotations

from time import time

import igraph
import networkx as nx
import pandas as pd
import sknetwork

from build_graphs import DATAPATH


def partition(
    g: nx.Graph | igraph.Graph, kind: str = "louvain", usermap: pd.Series | None = None
) -> pd.Series:
    """Compute partitions with various methods."""
    print("Computing", kind)

    t0 = time()
    if kind == "louvain":
        p = nx.community.greedy_modularity_communities(g, weight="weight")
    elif kind == "sk_louvain":
        # Much faster than networkx
        louvain = sknetwork.clustering.Louvain()
        p = louvain.fit_predict(g.adjacency)
        p = pd.Series(
            p,
            index=pd.Index(data=[int(n) for n in g["names"]], dtype="int64"),
            name="sk_louvain",
        )
    elif kind == "leiden":
        # optimize modularity with leiden
        p = g.community_leiden(
            objective_function="modularity", weights="weight", resolution=1
        )
    elif kind == "infomap":
        p = g.community_infomap(edge_weights="weight")
    elif kind == "fastgreedy":
        p = g.community_fastgreedy(weights="weight").as_clustering()

    if kind in {"infomap", "leiden", "fastgreedy"}:
        # convert to pd.Series
        vs = g.get_vertex_dataframe().astype("int64")
        p = {u: ip for ip, _p in enumerate(p) for u in _p}
        p = pd.Series(vs.index, index=pd.Index(data=vs.id, dtype="int64")).map(p)
    print("Elapsed time", time() - t0)
    print(p.value_counts())

    if usermap is not None:
        p.index = p.index.map(usermap)

    return p


def plot_comm_size(parts: pd.DataFrame) -> None:
    """Plot the community sizes."""
    from matplotlib import pyplot as plt

    fig, axs = plt.subplots(ncols=len(parts.columns), sharey=True)

    for ax, part in zip(axs, parts.columns):
        counts = parts[part].value_counts().sort_values(ascending=False).cumsum()
        counts /= len(parts[part])
        ax.scatter(range(len(counts)), counts)
        ax.semilogx()
        ax.set_title(part)
        ax.set_xlim(1, 20)
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


def main(deadline: str) -> None:
    """Do the main."""
    fname = DATAPATH / f"retweet_graph_undirected_{deadline}.graphml"
    usermap = pd.read_csv(DATAPATH / f"hyprgraph_{deadline}_usermap.csv.gz")["0"]
    p = pd.DataFrame()

    g_ig = igraph.read(fname, format="graphml")
    pp = partition(g_ig, kind="leiden", usermap=usermap)
    p["leiden"] = pp
    p.index = pp.index

    g_sk = sknetwork.data.from_graphml(fname)
    p["louvain"] = partition(g_sk, kind="sk_louvain", usermap=usermap)

    p["infomap"] = partition(g_ig, kind="infomap")

    plot_comm_size(p)

    for part in p.columns:
        p[part + "_5000"] = simplify_community_struct(p["leiden"], comm_size=5000)
        p[part + "_90"] = simplify_community_struct(p["leiden"], coverage=0.9)

    p.to_csv(DATAPATH / f"communities_{deadline}.csv.gz")


if __name__ == "__main__":
    for deadline in ["2021-06-01"]:
        main(deadline)
