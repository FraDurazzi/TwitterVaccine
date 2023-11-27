#!/usr/bin/env python3
"""For each user compute the number of links to each community."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from build_graphs import DATAPATH, load_graph


def projector(partition: pd.Series) -> sparse.csr_matrix:
    """Build a projector to the community space."""
    return sparse.coo_matrix(
        (np.ones(len(partition)), (partition, np.arange(len(partition)))),
        shape=(partition.nunique(), len(partition)),
        dtype="int64",
    ).tocsr()


def adj(deadline: str) -> tuple[sparse.spmatrix, pd.Series]:
    """Load the adjacency matrix."""
    tail, head, usermap = load_graph(deadline)
    print(tail.shape, head.shape, len(usermap))
    print(tail.max(), head.max())

    return tail @ head.T, usermap


def main(deadline: str) -> None:
    """Do the main."""
    comm = pd.read_csv(
        DATAPATH / f"communities_{deadline}.csv.gz",
    )

    adjacency, usermap = adj(deadline)

    for part in comm.columns:
        if "_" in part:
            print(part)
            proj = projector(comm[part])

            out_freq = adjacency @ proj.T
            in_freq = proj @ adjacency

            freq = pd.DataFrame.sparse.from_spmatrix(out_freq)
            freq += pd.DataFrame.sparse.from_spmatrix(in_freq.T)
            freq.index = pd.Index(usermap.sort_index(), dtype=usermap.dtypes, name="id")

            freq.to_csv(DATAPATH / f"communities_freq_{part}_{deadline}.csv.gz")


if __name__ == "__main__":
    for deadline in ["2021-06-01"]:
        main(deadline)
