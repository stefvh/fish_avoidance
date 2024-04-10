from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np


def plot(
    filename,
    jc_probabilities,
    x_bins,
    y_bins,
    x_label,
    y_label,
    z_names,
):
    n_states, n_x_states, n_y_states = jc_probabilities.shape

    fig, axes = plt.subplots(1, n_states, figsize=(4 + 6 * n_states, 12))

    for i in range(n_states):
        ax = axes[i]
        # plot jc as main figure

        im = ax.imshow(
            jc_probabilities[i].T,
            cmap="GnBu_r",
            origin="lower",
            interpolation="none",
            # extent=[2.5, 102.5, 2.5, 102.5],
            vmin=0,
            vmax=0.12,
        )
        ax.set_xticks(np.arange(n_x_states))
        xticklabels = [f"{x:.1f}" for x in x_bins]
        ax.set_xticklabels(xticklabels)

        ax.set_yticks(np.arange(n_y_states))
        yticklabels = [f"{y:.1f}" for y in y_bins]
        ax.set_yticklabels(yticklabels)

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        # cbar = fig.colorbar(im, ax=ax)

        # plot marginal probabilities histograms
        divider = make_axes_locatable(ax)
        axbottom = divider.append_axes("bottom", size=1, pad=0.3, sharex=ax)
        axleft = divider.append_axes("left", size=1, pad=0.4, sharey=ax)

        axbottom.set_xlabel(x_label, fontsize=16)
        axleft.set_ylabel(y_label, fontsize=16)

        # marginal probabilities
        p_x = np.sum(jc_probabilities[i], axis=1)
        p_y = np.sum(jc_probabilities[i], axis=0)

        # histogram covering space between each other
        axbottom.hist(x_bins, bins=x_bins, weights=p_x, orientation="vertical")
        axbottom.set_xlim((x_bins[0], x_bins[-1]))
        axbottom.set_ylim((0, 0.2))
        axleft.hist(y_bins, bins=y_bins, weights=p_y, orientation="horizontal")
        axleft.set_ylim((y_bins[0], y_bins[-1]))
        axleft.set_xlim((0, 1.0))

        ax.set_title(f"{z_names[i]}")

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()