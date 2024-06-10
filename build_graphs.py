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
from scipy import sparse
# from DIRS import TRANSFORMERS_CACHE_DIR, DATA_DIR, LARGE_DATA_DIR
from dirs import LARGE_DATA_DIR

DATAPATH = pathlib.Path("data")
DATAPATH.mkdir(parents=True, exist_ok=True)
DEADLINES = [
    "test",
    "2021-06-01",
    "2022-01-01",
    "2022-07-01",
    "2023-01-01",
    "2024-01-01",
]


class Graph:
    """Twitter data to build a graph."""

    def __init__(self, data: pd.DataFrame) -> None:
        """Initialize."""
        self._data = data.dropna(subset="retweeted_status.id").rename(
            columns={
                "user.id": "target",
                "retweeted_status.user.id": "source",
            }
        )
        # I want hyperlinks counted from 0, 1... as this will become the column index.
        hyperlinks = {
            k: i for i, k in enumerate(self._data["retweeted_status.id"].unique())
        }
        self._data["hyperlink"] = self._data["retweeted_status.id"].map(
            lambda x: hyperlinks[x]
        )

        self.users = pd.Series(
            list(set(self._data["source"]) | set(self._data["target"])), name="user_id"
        ).rename_axis("user_index")
        users_inv = {u: i for i, u in self.users.items()}

        hg_links = self._data["hyperlink"]
        hg_source = self._data["source"].map(lambda x: users_inv[x])
        hg_target = self._data["target"].map(lambda x: users_inv[x])

        self.tail = sparse.coo_matrix(
            (np.ones(len(self._data)), (hg_source, hg_links)),
            shape=(len(users_inv), len(self._data)),
            dtype=int,
        ).tocsr()
        self.head = sparse.coo_matrix(
            (np.ones(len(self._data)), (hg_target, hg_links)),
            shape=(len(users_inv), len(self._data)),
            dtype=int,
        ).tocsr()

    def info(self) -> None:
        """Print information."""
        print("Tail", self.tail.shape, self.tail.nnz)
        print("Head", self.head.shape, self.head.nnz)

    def largest_component(self) -> None:
        """Keep only the largest componet."""
        tail, head, comp_indx = extract_largest_component(self.tail, self.head)
        self.tail = tail
        self.head = head
        self.users = self.users.iloc[comp_indx].reset_index(drop=True)
        # self._data do not need to be shrinked since the hyperlink dimension is not touched.

    def adj(self) -> sparse.spmatrix:
        """Return the sparse adjacency matrix."""
        return self.tail @ self.head.T

    def write(self, basepath: pathlib.Path) -> None:
        """Write necessary data to disk."""
        sparse.save_npz(basepath.parent / (basepath.name + "_head.npz"), self.head)
        sparse.save_npz(basepath.parent / (basepath.name + "_tail.npz"), self.tail)
        self._data[["hyperlink", "retweeted_status.id"]].to_csv(
            basepath.parent / (basepath.name + "_ids.csv.gz")
        )
        # Save users map to indeces for internal use (not shareable).
        self.users.to_csv(
            basepath.parent / (basepath.name + "_users.csv.gz"),
            index_label="user_index",
        )


def load_data(deadline: pd.Timestamp | str | None) -> Graph:
    """Load the full dataset."""
    from locals import RAW_DATAPATH

    df_full = pd.read_csv(
        LARGE_DATA_DIR / "df_full.csv.gz",
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
        nrows=100000 if deadline == "test" else None,
    )
    if deadline is None:
        return Graph(df_full)
    return Graph(df_full[df_full.created_at < deadline])


def load_graph(
    deadline: pd.Timestamp | str,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, pd.Series]:
    """Load head tail and usermap."""
    head = sparse.load_npz(DATAPATH / f"hypergraph_{deadline}_head.npz")
    tail = sparse.load_npz(DATAPATH / f"hypergraph_{deadline}_tail.npz")

    users = pd.read_csv(
        DATAPATH / f"hypergraph_{deadline}_users.csv.gz",
        index_col="user_index",
        dtype="int64",
    )["user_id"]

    return tail, head, users


def extract_largest_component(
    tail: sparse.csr_matrix, head: sparse.csc_matrix
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, np.ndarray]:
    """Extract the largest component.

    remove users from smaller componets and retweets that involve those smaller components.
    """
    rtw_net = tail @ head.T
    print("Full adj", rtw_net.shape)

    n_comps, components = sparse.csgraph.connected_components(rtw_net, directed=False)

    largest_components = Counter(components).most_common(1)
    largest_component, new_nn = largest_components[0]
    largest_component = np.argwhere(components == largest_component).flatten()
    print(
        f"Largest component with {new_nn} users ({100 * new_nn/tail.shape[0]:5.2f} %)."
    )

    # projector to users in the largest component
    largest_component_proj = sparse.coo_matrix(
        (
            np.ones_like(largest_component),
            (np.arange(len(largest_component)), largest_component),
        ),
        shape=(new_nn, tail.shape[0]),
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

    return tail, head, largest_component


def parse_date(date: str | pd.Timestamp) -> pd.Timestamp | str:
    """Toggle format from str to pd.Timestamp."""
    if isinstance(date, str):
        return pd.Timestamp(date + "T00:00:00+02")
    return date.isoformat().split()[0].split("T")[0]


def main(deadline: pd.Timestamp | str) -> None:
    """Do the main."""
    if isinstance(deadline, str):
        _deadline = deadline
    elif isinstance(deadline, pd.Timestamp):
        _deadline = parse_date(deadline)
    print("============")
    print(_deadline)
    print("============")

    datagraph = load_data(_deadline)
    datagraph.largest_component()
    datagraph.write(DATAPATH / f"hypergraph_{_deadline}")

    # Directed graph
    graph = nx.from_scipy_sparse_array(
        datagraph.adj(), create_using=nx.DiGraph, edge_attribute="weight"
    )
    # node label is saved in hyprgraph_deadline_usermap.csv.gz
    nx.write_graphml_lxml(
        graph,
        DATAPATH / f"retweet_graph_directed_{_deadline}.graphml",
    )
    nx.write_graphml_lxml(
        graph.to_undirected(),
        DATAPATH / f"retweet_graph_undirected_{_deadline}.graphml",
    )


if __name__ == "__main__":
    for deadline in DEADLINES:
        main(parse_date(deadline))
