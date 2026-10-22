from matplotlib import pyplot as plt
import numpy as np

def plot_co(
    filename,
    co_probabilities,
    x_bins,
    x_label,
    y_label,
    z_names,
):
    z_x_weights = []

    N_Z = sum([co_probabilities[i]["N_Z"] for i in range(len(co_probabilities))])
    for i in range(len(co_probabilities)):
        q_x = co_probabilities[i]["Q_X"] / N_Z
        z_x_weights.append(q_x)
    
    plot(
        filename=filename,
        x_bins=x_bins,
        z_x_weights=z_x_weights,
        x_label=x_label,
        y_label=y_label,
        z_names=z_names,
    )

def plot(
    filename,
    x_bins,
    z_x_weights,
    x_label,
    y_label,
    z_names,
):
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    bin_width = x_bins[1] - x_bins[0]

    for i, x_weights in enumerate(z_x_weights):
        w_bar = bin_width
        ax.bar(
            x_bins +  0.5 * w_bar,
            x_weights,
            width=w_bar,
            alpha=0.5,
            label=z_names[i],
            edgecolor="black",
        )
        # ax.set_xticks(np.arange(len(x_weights)))
        # xticklabels = [f"{x:.1f}" for x in x_bins]
        # ax.set_xticklabels(xticklabels)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    ax.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()