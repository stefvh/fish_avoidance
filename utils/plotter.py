from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt


plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "text.latex.preamble": r"\usepackage{amsfonts}" r"\usepackage{amsmath}",
    }
)


def plot_multipdf(n_total_rows, 
                  n_fig_rows, 
                  n_fig_cols, 
                  f_ax_plot, 
                  output_filepath,
                  wspace=2/6,):
    with PdfPages(f"{output_filepath}") as pdf:
        for i in range(n_total_rows):
            if i % n_fig_rows != 0:
                continue
            else:
                if i > 0:
                    pdf.savefig()
                    plt.close(fig)
            fig, axes = plt.subplots(
                n_fig_rows, n_fig_cols, figsize=(2 * n_fig_cols + 2, 2 * n_fig_rows)
            )
            fig.subplots_adjust(wspace=wspace)

            for j in range(n_fig_rows):
                if i + j >= n_total_rows:
                    break
                ax_row = axes[j]
                f_ax_plot(ax_row, i, j)
        pdf.savefig()
        plt.close(fig)
