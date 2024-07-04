#!/usr/bin/env python3

import pathlib

import pandas as pd

from build_embeddings import embed_fa2
from build_graphs import DATAPATH, load_graph

MAX_LIM = 10000


def load(deadline: str) -> pd.DataFrame:
    """Load updated positions if available otherwise the old ones."""
    for path in [
        pathlib.Path(x)
        for x in [
            DATAPATH / f"embedding_fa2_stronggrav_{deadline}_refined.csv.gz",
            DATAPATH / f"embedding_fa2_{deadline}.csv.gz",
        ]
    ]:
        if path.is_file():
            break

    # Just for the IDs
    old_pos = pd.read_csv(DATAPATH / f"embedding_fa2_{deadline}.csv.gz", index_col=0)

    new_pos = pd.read_csv(path, index_col=0)
    new_pos["user_id"] = old_pos["user_id"]
    return new_pos


def refine_positions(deadline: str, deadline_fix: str | None = None) -> None:
    """Refine and save positions with optional fixed positions."""
    positions = load(deadline)

    # linking pattern (to calculate the adjacency matrix)
    tail, head, usermap = load_graph(deadline)
    adj = tail @ head.T

    # load fixed positions
    if deadline_fix is not None:
        fixed_pos = load(deadline_fix)
        fixed_pos = (
            fixed_pos.set_index("user_id")
            .reindex(positions["user_id"])  # align to positions
            .reset_index()
            .set_index(positions.index)
            .dropna()
        )
        positions.loc[fixed_pos.index] = fixed_pos

    positions = embed_fa2(adj, positions, n_cycles=500)
    positions.to_csv(DATAPATH / f"embedding_fa2_stronggrav_{deadline}_refined.csv.gz")


def main() -> None:
    """Do the main."""

    refine_positions("pre")
    refine_positions("post", deadline_fix="pre")


if __name__ == "__main__":
    main()
