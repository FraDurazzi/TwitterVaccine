#!/usr/bin/env python3

import pandas as pd
from matplotlib import pyplot as plt
from sklearn import decomposition

from build_graphs import DEADLINES
from load_embeddings import load

DEADLINES = DEADLINES[1:]

USE = "louvain"
USE = "n2v"
USE = "laplacian"
USE = "fa2"


def embed2d(input: pd.DataFrame) -> pd.DataFrame:
    """Make a PCA to a 2D distribution."""
    if len(input.columns) == 2:
        return input

    # return input[input.columns[:2]]
    pca = decomposition.PCA(n_components=2).fit_transform(input)
    return pd.DataFrame(pca)


def crop(emb1: pd.Series, emb2: pd.Series, q: float) -> dict[str, list]:
    """Return xlim and ylim to crop at given quant1iles."""
    xlim = emb1.quantile([q, 1 - q])
    ylim = emb2.quantile([q, 1 - q])

    if xlim.iloc[0] > 0:
        xlims = xlim.to_list()
    else:
        xlims = [-xlim.abs().max(), xlim.abs().max()]

    if ylim.iloc[0] > 0:
        ylims = ylim.to_list()
    else:
        ylims = [-ylim.abs().max(), ylim.abs().max()]

    return {"xlim": xlims, "ylim": ylims}


def main() -> None:
    """Do the main."""
    fig, axes = plt.subplots(ncols=len(DEADLINES), nrows=2, sharex="row", sharey="row")

    for deadline, axs in zip(DEADLINES, axes.T):
        print(deadline)
        try:
            emb = load(deadline=deadline, kind=USE)
        except FileNotFoundError:
            continue
        except EOFError:
            continue
        print(emb)
        # comm = pd.read_csv(f"./data/communities_{deadline}.csv.gz")["leiden"]
        emb_comms = load(deadline=deadline, kind="leiden")
        emb_comms = emb_comms.drop(emb_comms.columns[-1], axis=1)
        comm = pd.Series(
            [x.argmax() for x in emb_comms.to_numpy()], index=emb_comms.index
        )

        emb = emb.iloc[:, -2:]
        emb1 = emb.iloc[:, -1]
        emb2 = emb.iloc[:, -2]

        if USE == "fa2":
            emb1 = emb1 / emb1.abs().mean()
            emb2 = emb2 / emb2.abs().mean()

        opts = {"cmap": "rainbow", "s": 0.2, "alpha": 0.05, "lw": 0}
        ax_opts: dict[str, list] = {"xticklabels": [], "yticklabels": []}
        ax_opts.update(crop(emb1, emb2, 0.01))
        print(ax_opts)

        ax = axs[0]
        ax.scatter(emb1, emb2, c=comm, **opts)
        ax.set(title="\n".join([deadline, f"N={len(emb)}"]), **ax_opts)

        ax = axs[1]
        ax.scatter(emb1, emb2, c=comm, **opts)
        ax.grid()

        if deadline == DEADLINES[0]:
            ax_opts.update(crop(emb1, emb2, 0.1))

            ax.set(ylabel="zoom", **ax_opts)

        # ax.hist(emb1, bins=100, lw=2, log=True)
        # ax.semilogy()

    fig.tight_layout()
    fig.savefig("./data/embedding_laplacian.png", dpi=200)


if __name__ == "__main__":
    main()
