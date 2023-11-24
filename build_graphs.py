#!/usr/bin/env python3
"""Build graphs from retweets.

It will build:
    1. retweet network (symmetric)
    2. retweet network (directed)
    3. return hyprgraph (directed, head + tail)
"""
from __future__ import annotations

import pathlib
from collections import Counter

import networkx as nx
import numpy as np
import pandas as pd
from locals import RAW_DATAPATH
from scipy import sparse

DATAPATH = pathlib.Path("data")
DATAPATH.mkdir(parents=True, exist_ok=True)


def load_data(deadline: str) -> pd.DataFrame:
    """Load the full dataset."""
    df_full = pd.read_csv(
        RAW_DATAPATH / "df_full.csv.gz",
        index_col="id",
        dtype={
            "id": str,
            "text": str,
            "user.id": str,
            "user.screen_name": str,
            "place": str,
            "url": str,
            "retweeted_status.id": str,
            "retweeted_status.user.id": str,
            "retweeted_status.url": str,
            "annotation": str,
            "user_annotation": str,
            "lang": str,
        },
        na_values=["", "[]"],
        parse_dates=["created_at"],
        lineterminator="\n",
        # nrows=100000,  # uncomment to test the code on a short dataset
    )
    return df_full[df_full.created_at < deadline]


def load_graph(df_full: pd.DataFrame) -> pd.DataFrame:
    """Load the whole dataset and compute the (tweet, retweets) pairs."""
    # columns:
    # id,created_at,text,user.id,user.screen_name,place,url,
    #      retweeted_status.id,retweeted_status.user.id,retweeted_status.url,
    #      annotation,user_annotation,lang

    # filter only tweets before a deadline.

    # users that retweet
    retweets = df_full.dropna(subset="retweeted_status.id")[
        ["user.id", "retweeted_status.id", "retweeted_status.user.id"]
    ]

    # use meaningful headers
    retweets.columns = ["source", "hyperlink", "target"]
    hyperlinks = {k: i for i, k in enumerate(retweets["hyperlink"].unique())}
    retweets["hyperlink"] = retweets["hyperlink"].map(lambda x: hyperlinks[x])
    return retweets


def write_hypergraph(retweets: pd.DataFrame, fname: str) -> None:
    """Write down the hyprgraph."""
    print("Building hyprgraph for", fname)

    users = set(retweets["source"]) | set(retweets["target"])
    users = {u: i for i, u in enumerate(users)}

    weight = retweets["hyperlink"].value_counts()

    target = np.array(
        [
            [
                r["hyperlink"],  # id of hyperlink
                users[r["source"]],  # id of source user
                users[r["target"]],  # i of target user
                weight.loc[r["hyperlink"]],  # counts of same hyperlinks
            ]
            for _, r in retweets.iterrows()
        ],
        ndmin=2,
    )
    print(target.shape)

    tail = sparse.coo_matrix(
        (target[:, 3], (target[:, 1], target[:, 0])),
        shape=(len(users), len(retweets)),
        dtype=int,
    ).tocsr()
    print("Tail", tail.shape)
    head = sparse.coo_matrix(
        (np.ones(target.shape[0]), (target[:, 2], target[:, 0])),
        shape=(len(users), len(retweets)),
        dtype=int,
    ).tocsr()
    print("Head", head.shape)

    # only get the largest component
    tail, head, comp_indx = extract_largest_component(tail, head)

    users = pd.Series(list(users.keys()), index=list(users.values()))
    users = users[comp_indx]
    print(users.shape, tail.shape, head.shape)

    sparse.save_npz(fname.format("head", "npz"), head)
    sparse.save_npz(fname.format("tail", "npz"), tail)

    users.to_csv(fname.format("usermap", "csv.gz"))

    return tail @ head.T, users


def extract_largest_component(
    tail: sparse.csr_matrix, head: sparse.csc_matrix
) -> (sparse.csr_matrix, sparse.csr_matrix):
    """Extract the largest component.

    remove users from smaller componets and retweets that involve those smaller components.
    """
    rtw_net = tail @ head.T
    print("Full adj", rtw_net.shape)

    n_comps, components = sparse.csgraph.connected_components(rtw_net, directed=False)

    largest_components = Counter(components).most_common(1)
    largest_component, new_N = largest_components[0]
    largest_component = np.argwhere(components == largest_component).flatten()
    print(f"Largest component with {new_N} users ({100 * new_N/tail.shape[0]:5.2f} %).")

    # projector to users in the largest component
    largest_component_proj = sparse.coo_matrix(
        (
            np.ones_like(largest_component),
            (np.arange(len(largest_component)), largest_component),
        ),
        shape=(new_N, tail.shape[0]),
    ).tocsr()
    tail = largest_component_proj @ tail
    head = largest_component_proj @ head

    _, retweets_to_keep = np.nonzero(np.asarray(tail.sum(0)) * np.asarray(head.sum(0)))
    print(f"Retweets in smaller componets: {tail.shape[1] - retweets_to_keep.shape[0]}")
    retweets_to_keep_proj = sparse.coo_matrix(
        (
            np.ones_like(retweets_to_keep),
            (retweets_to_keep, np.arange(len(retweets_to_keep))),
        ),
        shape=(tail.shape[1], len(retweets_to_keep)),
    )
    tail = tail @ retweets_to_keep_proj
    head = head @ retweets_to_keep_proj
    print("Tail", tail.shape)
    print("Head", head.shape)

    return tail, head, largest_component


def main(deadline: pd.Timestamp) -> None:
    """Do the main."""
    print("============")
    print(parse_date(deadline))
    print("============")

    retweets = load_graph(load_data(deadline))
    adj, users = write_hypergraph(
        retweets, DATAPATH / f"hyprgraph_{parse_date(deadline)}_{{}}.{{}}"
    )

    # Directed graph
    graph = nx.from_scipy_sparse_array(
        adj, create_using=nx.DiGraph, edge_attribute="weight"
    )
    # graph = nx.relabel_nodes(graph, dict(enumerate(users.index)))
    nx.write_graphml_lxml(
        graph,
        DATAPATH / f"retweet_graph_directed_{parse_date(deadline)}.graphml",
    )
    nx.write_graphml_lxml(
        graph.to_undirected(),
        DATAPATH / f"retweet_graph_undirected_{parse_date(deadline)}.graphml",
    )


def parse_date(date: str | pd.Timestamp) -> pd.Timestamp | str:
    """Toggle format from str to pd.Timestamp."""
    if isinstance(date, str):
        return pd.Timestamp(date + "T00:00:00+02")
    return date.isoformat().split()[0].split("T")[0]


if __name__ == "__main__":
    for deadline in [
        "2021-06-01",
        "2022-01-01",
        "2022-07-01",
        "2023-01-01",
        "2024-01-01",
    ]:
        main(parse_date(deadline))
