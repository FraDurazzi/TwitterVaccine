#!/usr/bin/env python3

import asyncio
import pathlib
from concurrent.futures import ThreadPoolExecutor as Executor

import pandas as pd
from matplotlib import pyplot as plt

from build_embeddings import embed_fa2
from build_graphs import DEADLINES, load_graph

MAX_LIM = 10000


def load(deadline: str) -> pd.DataFrame:
    """Load updated positions if available otherwise the old ones."""
    for path in [
        pathlib.Path(x)
        for x in [
            f"data/embedding_fa2_stronggrav_{deadline}_refined.csv.gz",
            f"data/embedding_fa2_stronggrav_{deadline}.csv.gz",
            f"data/embedding_fa2_{deadline}_refined.csv.gz",
            f"data/embedding_fa2_{deadline}.csv.gz",
        ]
    ]:
        if path.is_file():
            break

    old_pos = pd.read_csv(f"data/embedding_fa2_{deadline}.csv.gz", index_col=0)

    # path = pathlib.Path(f"data/embedding_fa2_{deadline}_refined.csv.gz")
    # if not path.exists():
    #     print("Use old positions")
    #     return old_pos

    print("Start from updated positions")
    new_pos = pd.read_csv(path, index_col=0)
    new_pos["user_id"] = old_pos["user_id"]
    return new_pos


def refine_positions(
    deadline: str, deadline_fix: str | None = None, cycles: int = 10
) -> None:
    """Refine and save positions with optional fixed positions."""
    positions = load(deadline)

    # communities (just for plotting)
    comms = pd.read_csv(f"data/communities_{deadline}.csv.gz", index_col="user_index")

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

    for i in range(cycles):
        print(f" Cycle {i}  for {deadline}")
        positions = embed_fa2(adj, positions, n_cycles=500)
        positions.to_csv(f"./data/embedding_fa2_stronggrav_{deadline}_refined.csv.gz")

        asyncio.run(
            plot_positions(
                pd.concat([positions, comms], axis=1),
                f"./data/embedding_fa2_stronggrav_{deadline}_refined.png",
            )
        )

        if deadline_fix is not None:
            positions.loc[fixed_pos.index] = fixed_pos


async def plot_positions(positions: pd.DataFrame, fname: str) -> None:
    """Plot."""
    fig, ax = plt.subplots(
        figsize=(10, 10),
        gridspec_kw={"left": 0, "right": 1, "bottom": 0, "top": 1},
    )
    ax.set_axis_off()
    ax.scatter(
        positions["fa2_x"],
        positions["fa2_y"],
        c=positions["leiden"],
        s=1,
        alpha=0.1,
        cmap="rainbow",
    )
    ax.set(xlim=(-MAX_LIM, MAX_LIM), ylim=(-MAX_LIM, MAX_LIM))
    plt.savefig(fname, dpi=300)


def main() -> None:
    """Do the main."""

    refine_positions(DEADLINES[1], cycles=1)

    with Executor(max_workers=8) as exc:
        for deadline in DEADLINES[2:]:
            exc.submit(refine_positions, deadline, deadline_fix=DEADLINES[1], cycles=1)


if __name__ == "__main__":
    main()
