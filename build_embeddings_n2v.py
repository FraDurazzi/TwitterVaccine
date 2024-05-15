#!/usr/bin/env python3
"""Eigenvector embeddings."""

import numpy as np
import node2vec
import networkx as nx

from build_graphs import DEADLINES, load_graph

NUM_EIGS = 11


def main() -> None:
    """Do the main."""
    for deadline in DEADLINES:
        tail, head, usermap = load_graph(deadline)

        adj = (tail @ head.T).astype(np.float64)

        print(adj.nnz, adj.shape)
        print("Density", adj.nnz / adj.shape[0] ** 2)

        n2v = node2vec.Node2Vec(
            nx.from_scipy_sparse_array(adj, create_using=nx.DiGraph),
            dimensions=NUM_EIGS - 1,
            walk_length=20,
            num_walks=1000,
            workers=8
        )
        model = n2v.fit()

        model.wv.save_word2vec_format(f"./data/embedding_n2v_{deadline}.txt")


if __name__ == "__main__":
    main()
