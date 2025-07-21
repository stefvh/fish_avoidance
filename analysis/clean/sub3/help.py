import os

from matplotlib import pyplot as plt


def save_plot(fig, filename):
    output_folder = "analysis/clean/figs/sub3"
    os.makedirs(output_folder, exist_ok=True)
    fig.savefig(os.path.join(output_folder, filename))
    plt.close()
    print(f"Figure saved at {output_folder}/{filename}")