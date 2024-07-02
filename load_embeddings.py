#!/usr/bin/env python3
"""Use this module to load the embeddings.

The embeddings will be a pandas dataframe with user_id as index and all the embedding values as entries.
"""

from pathlib import Path

import numpy as np
import pandas as pd

EMBEDDINGS = [
    "laplacian",
    "norm_laplacian",
    "leiden",
    "louvain",
    "labelpropagation",
    "n2v",
    "fa2",
]
STOR = Path("/mnt/stor/users/mauro.faccin/twitter_vaccine/data_tw_tight/")
DIR = STOR if STOR.is_dir() else Path("data_tw_tight")


def load(kind: str, deadline: str) -> pd.DataFrame:
    """Load the embedding.

    Parameters:
    ----------
    kind : str
        type of embedding: one of ['laplacian', 'norm_laplacian', 'leiden', 'louvain', 'labelpropagation', 'n2v', 'fa2']
    deadline : str
        Temporal threshold used to build the network.
    """
    if kind == "leiden":
        data = pd.read_csv(
            DIR / f"embedding_community_leiden_{deadline}.csv.gz",
            index_col=0,
        )
    elif kind == "louvain":
        data = pd.read_csv(
            DIR / f"embedding_community_louvain_{deadline}.csv.gz",
            index_col=0,
        )
    elif kind == "labelpropagation":
        data = pd.read_csv(
            DIR / f"embedding_community_labelpropagation_{deadline}.csv.gz",
            index_col=0,
        )
    elif kind == "laplacian":
        data = pd.read_csv(
            DIR / f"embedding_laplacian_{deadline}.txt.gz", sep=" ", header=None
        )
        data = data.drop(columns=data.columns[0])
        data = data.abs()
        data *= len(data) / np.linalg.norm(data, ord=1, axis=0)
    elif kind == "norm_laplacian":
        data = pd.read_csv(
            DIR / f"embedding_norm_laplacian_{deadline}.txt.gz", sep=" ", header=None
        )
        data = data.drop(columns=data.columns[0])
        data = data.abs()
        data *= len(data) / np.linalg.norm(data, ord=1, axis=0)
    elif kind == "n2v":
        data = pd.read_csv(
            DIR / f"embedding_n2v_{deadline}.txt",
            sep=" ",
            header=None,
            skiprows=1,
            index_col=0,
        )
        data = data.sort_index()
    elif kind == "fa2":
        data = pd.read_csv(DIR / f"embedding_fa2_{deadline}.csv.gz", index_col=0).drop(
            columns="user_id", errors="ignore"
        )

        data = data / data.abs().mean(axis=0)
    else:
        raise NotImplementedError()

    users = pd.read_csv(
        DIR / f"hypergraph_{deadline}_users.csv.gz", index_col="user_index"
    )
    data.index = data.index.map(users["user_id"])
    return data


def main() -> None:
    """Do the main."""
    print("Using path:", DIR)
    print(load("fa2", "test"))
    print(load("laplacian", "test"))


if __name__ == "__main__":
    main()
