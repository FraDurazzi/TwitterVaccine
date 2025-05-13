#!/usr/bin/env python3
"""Eigenvector embeddings."""

import networkx as nx
import node2vec
import numpy as np

from build_graphs import DATAPATH, DEADLINES, load_graph

NUM_EIGS = 11


def main() -> None:
    """Do the main."""
    for deadline in DEADLINES:
        print("Embedding:", deadline)
        tail, head, usermap = load_graph(deadline)

        adj = (tail @ head.T).astype(np.float64)

        print(adj.nnz, adj.shape)
        print("Density", adj.nnz / adj.shape[0] ** 2)

        n2v = node2vec.Node2Vec(
            nx.from_scipy_sparse_array(adj, create_using=nx.DiGraph),
            dimensions=NUM_EIGS - 1,
            walk_length=50,
            num_walks=100,
            workers=20,
            p=1,  # backward probability
            q=0.5,  # Search further for homophily (highlights community structure)
        )
        model = n2v.fit()

        model.wv.save_word2vec_format(DATAPATH / f"embedding_n2v_{deadline}.txt")


if __name__ == "__main__":
    main()
