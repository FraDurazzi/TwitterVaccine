#!/usr/bin/env python3
"""Eigenvector embeddings.

Prepare the adjacency matrix for ./build_embeddings_eigs.m
Run this before that.
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# from scikits import umfpack
from scipy import sparse
from scipy.sparse import linalg

from build_graphs import DATAPATH, DEADLINES, load_graph

NUM_EIGS = 11


def find_smaller_eigenvectors(
    matrix: sparse.spmatrix, neig: int
) -> tuple[np.ndarray, np.ndarray]:
    """Find the `neig` eigpairs corresponding to the smallerst eigenvalues.

    WARNING: The `numpy` implementation is too slow due to the choice of C parameters.
    Use `Octave`.
    """
    # Find largest eigenvalue (positive defined matrix has all positive evals)
    eval = linalg.eigsh(matrix, k=1, which="LM", tol=1e-5)[0][0]

    # flip-shift the eigenvalues
    linalg.use_solver(useUmfpack=True)

    # find eigval eigvec
    evals, evecs = linalg.eigsh(
        eval * sparse.eye(matrix.shape[0]) - matrix,
        k=neig,
        which="LM",
        sigma=eval,
        tol=1e-5,
        v0=np.ones(matrix.shape[0]),
    )
    return eval - evals, evecs


def main() -> None:
    """Do the main."""
    fig, axs = plt.subplots(ncols=len(DEADLINES), sharex=True, sharey=True)
    for deadline, ax in zip(DEADLINES, axs):
        print(deadline)
        tail, head, usermap = load_graph(deadline)
        adj = (tail @ head.T).astype(np.float64)
        adj = adj + adj.T  # symmetrize

        _adj = adj.tocoo()
        data = pd.DataFrame({"i": _adj.row, "j": _adj.col, "val": _adj.data}, dtype=int)
        data.to_csv(
            DATAPATH / f"adj_{deadline}.m.gz", index=False, header=False, sep=" "
        )


if __name__ == "__main__":
    main()
