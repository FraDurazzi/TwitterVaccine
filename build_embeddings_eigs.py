#!/usr/bin/env python3
"""Eigenvector embeddings.

Prepare the adjacency matrix for ./build_embeddings_eigs.m
Run this before that.
"""

import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scikits import umfpack
from scipy import sparse
from scipy.sparse import linalg

from build_graphs import DEADLINES, load_graph

NUM_EIGS = 11


def find_smaller_eigenvectors(
    matrix: sparse.spmatrix, neig: int
) -> tuple[np.ndarray, np.ndarray]:
    """Find the `neig` eigpairs corresponding to the smallerst eigenvalues."""
    # fing largest eigenvalue (pos def matrix has all positive evals)
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
        comms = pd.read_csv(
            f"data/communities_{deadline}.csv.gz", index_col="user_index"
        )

        adj = (tail @ head.T).astype(np.float64)
        adj = adj + adj.T  # symmetrize

        _adj = adj.tocoo()
        data = pd.DataFrame({"i": _adj.row, "j": _adj.col, "val": _adj.data}, dtype=int)
        data.to_csv(f"data/adj_{deadline}.m.gz", index=False, header=False, sep=" ")
        continue

        print(adj.nnz, adj.shape)
        print("Density", adj.nnz / adj.shape[0] ** 2)

        degree = sparse.diags(adj.sum(axis=1).A1, shape=adj.shape, offsets=0)

        t0 = time.time()
        eigenvalues, eigenvectors = find_smaller_eigenvectors(
            degree - adj, neig=NUM_EIGS
        )
        print(time.time() - t0)
        print(eigenvalues)

        t0 = time.time()
        eigenvalues, eigenvectors = linalg.eigsh(
            degree - adj,
            k=NUM_EIGS,
            which="LM",
            sigma=0,
            tol=1e-5,
            v0=np.ones(adj.shape[0]),
        )
        print(time.time() - t0)
        print(eigenvalues)
        print(eigenvectors)

        # eigenvalues = -eigenvalues
        eigenvectors /= np.abs(eigenvectors).sum(0)

        eigs = pd.concat(
            [pd.DataFrame({"eigvalues": eigenvalues}).T, pd.DataFrame(eigenvectors)]
        )
        eigs.to_csv(f"./data/embedding_laplacian_{deadline}.csv.gz")

        ax.scatter(
            eigenvectors[:, 1],
            eigenvectors[:, 2],
            c=comms["louvain"],
            s=1,
            alpha=0.1,
            cmap="rainbow",
        )
        ax.set(title=deadline)
        # ax.loglog()
    plt.savefig("data/embedding_laplacian.pdf")


if __name__ == "__main__":
    main()
