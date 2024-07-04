#!/usr/bin/env python3
"""Load the community on the first deadline and find the embeddings on all other deadlines."""

import numpy as np
import pandas as pd
from scipy import sparse

from build_graphs import DATAPATH, DEADLINES, load_graph


def load_base_community(
    kind: str, drop_basket: bool = False
) -> tuple[sparse.spmatrix, pd.Series]:
    """Load the base community struct."""
    community = pd.read_csv(
        DATAPATH / "communities_pre.csv.gz", index_col="user_index"
    )[kind]

    if drop_basket:
        # drop the last **noise** community
        community = community[community < community.max()]

    proj = sparse.coo_matrix(
        (np.ones(len(community)), (np.arange(len(community)), community)),
        shape=(len(community), community.max() + 1),
    ).tocsr()

    users = pd.read_csv(
        DATAPATH / "hypergraph_pre_users.csv.gz", index_col="user_index"
    ).loc[community.index]["user_id"]
    users = users.reset_index(drop=True)

    return proj, users


def proj_to_base(users: pd.Series, base_users: pd.Series) -> sparse.spmatrix:
    """Compute the projector that send User to base users."""
    iusers = pd.Series(users.index, index=users["user_id"], name="i").astype(int)
    ibase = pd.Series(base_users.index, index=base_users, name="j").astype(int)

    indices = pd.merge(iusers, ibase, how="inner", left_index=True, right_index=True)

    return sparse.coo_matrix(
        (np.ones(len(indices)), (indices["i"], indices["j"])),
        shape=(len(users), len(base_users)),
    )


def main(kind: str) -> None:
    """Do the main."""
    print("---", kind, "---", sep="\n")
    comm_proj, base_users = load_base_community(kind, drop_basket=False)

    for deadline in DEADLINES:
        print(deadline)
        tail, head, users = load_graph(deadline)
        adj = tail @ head.T

        # symmetrize
        adj = adj + adj.T
        degree = adj.sum(1)

        # projector to the initial users
        base_proj = proj_to_base(users, base_users)

        # compute the links to the existing communities
        cols = (adj @ base_proj @ comm_proj).toarray()
        # add the total number of links
        cols = np.concatenate([cols, degree], axis=1)

        # write to file
        pd.DataFrame(cols).astype(int).to_csv(
            DATAPATH / f"embedding_community_{kind}_{deadline}.csv.gz"
        )


if __name__ == "__main__":
    for kind in ["leiden", "louvain", "labelpropagation"]:
        main(kind)
