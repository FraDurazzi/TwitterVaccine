#!/usr/bin/env python3
"""Starting from the graphs, compute the community structures."""

from __future__ import annotations

import numpy as np
import pandas as pd
from fa2_modified import ForceAtlas2
from scipy import sparse

from build_graphs import DEADLINES, load_graph


def main() -> None:
    """Do the main."""
    positions: dict[str, pd.DataFrame] = {}
    for deadline in DEADLINES:
        print(deadline)
        tail, head, usermap = load_graph(deadline)
        print("N users =", tail.shape[0])
        print("N edges =", head.nnz, tail.nnz)

        adj = tail @ head.T

        if deadline == DEADLINES[0]:
            # use a random initial position
            pos = embed_fa2(adj, init_pos=None, n_cycles=1000)
        elif deadline == DEADLINES[1]:
            # use positions from `test`
            init_pos = set_initial_pos(positions[DEADLINES[0]], usermap)
            pos = embed_fa2(
                adj,
                init_pos=init_pos,
                n_cycles=300,
            )
        else:
            # use positions from `2021-06-01`
            init_pos = set_initial_pos(positions[DEADLINES[1]], usermap)
            pos = embed_fa2(
                adj,
                init_pos=init_pos,
                n_cycles=100,
            )

        positions[deadline] = pd.concat([usermap, pos], axis=1)

        if deadline != DEADLINES[0]:
            pass

        positions[deadline].to_csv(f"data/embedding_fa2_{deadline}.csv.gz")


def set_initial_pos(old_pos: pd.DataFrame, new_usermap: pd.Series) -> pd.DataFrame:
    """Find old positions for the new map."""
    rng = np.random.default_rng()

    # initially set random positions
    new_pos = pd.concat(
        [
            new_usermap,
            pd.DataFrame(
                {
                    "fa2_x": rng.choice(old_pos["fa2_x"], size=len(new_usermap)),
                    "fa2_y": rng.choice(old_pos["fa2_y"], size=len(new_usermap)),
                }
            ),
        ],
        axis=1,
    )

    # the user_index should be incremental hence just use the user_id
    new_pos = new_pos.set_index("user_id")

    # same
    op = old_pos
    op = op.set_index("user_id")

    # assign the old position to user present in the previous step.
    new_pos.loc[op.index] = op

    # set the user_index as a incremental value
    new_pos = new_pos.reset_index(drop=True)
    new_pos.index.name = "user_index"
    return new_pos


def embed_fa2(
    adj: sparse.spmatrix, init_pos: pd.DataFrame | None, n_cycles: int = 50
) -> pd.DataFrame:
    """Embed into a 2D."""
    fa = ForceAtlas2(strongGravityMode=False, scalingRatio=3.0, gravity=3.0)
    if init_pos is None:
        pos = None
    else:
        pos = np.array([(x["fa2_x"], x["fa2_y"]) for _, x in init_pos.iterrows()])
    pos = fa.forceatlas2(adj + adj.T, pos=pos, iterations=n_cycles)
    return pd.DataFrame(pos, columns=["fa2_x", "fa2_y"])


if __name__ == "__main__":
    main()
