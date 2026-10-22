import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from analysis.clean.freezing_conditions import compute_related_freezing
import utils.plotter as plotter


def compute_error(
    df: pd.DataFrame, delta: int = 4, threshold_es: float = 2.0, weight_es=1
):
    # Shift 'cos_avoid' and 'speed_fish' by delta frames
    df["cos_avoid_shifted"] = df["cos_avoid"].shift(-delta)
    df["speed_fish_shifted"] = df["speed_fish"].shift(-delta)

    df["error_turn"] = 1 - df["cos_avoid_shifted"]
    df["error_turn"] /= 2  # Normalize to [0, 1]

    df["error_speed"] = np.where(df["freezing_related"] > 0, weight_es, 0)
    # df["error_speed"] = np.where(df["speed_fish_shifted"] < threshold_es, weight_es, 0)

    # if error_speed > 0, then error_turn = 0
    df.loc[df["error_speed"] > 0, "error_turn"] = 0

    df["error"] = df["error_turn"] + df["error_speed"]

    # Drop rows where we don't have valid shifted data
    df.dropna(subset=["cos_avoid_shifted", "speed_fish_shifted"], inplace=True)

    return df


def plot_error(
    df: pd.DataFrame,
    output_filename: str,
    output_folder: str,
):
    # Plotting the results
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 10), gridspec_kw={"height_ratios": [2, 1]}
    )

    # First subplot: Heatmap of error as a function of distance and v_app
    plot_heatmap(
        fig,
        ax1,
        df,
        v_app_col="v_app",
        distance_col="distance",
        error_col="error",
        v_app_range=(-20, 20),
        distance_range=(0, 40),
        bins_v_app=50,
        bins_distance=50,
    )

    # Second subplot: Error as a function of distance with bins, mean, and standard deviation
    plot_marginal(
        ax2,
        df,
        distance_col="distance",
        error_col="error",
        distance_range=(0.5, 10),
        bins_distance=40,
    )

    plt.tight_layout()
    fig.savefig(f"{output_folder}/{output_filename}.pdf")


def plot_heatmap(
    fig, 
    ax,
    df: pd.DataFrame,
    v_app_col: str,
    distance_col: str,
    error_col: str,
    v_app_range: tuple,
    distance_range: tuple,
    bins_v_app: int = 10,
    bins_distance: int = 10,
):
    # Filter rows based on the range limit
    df_filtered = df[
        (df[v_app_col] >= v_app_range[0])
        & (df[v_app_col] <= v_app_range[1])
        & (df[distance_col] >= distance_range[0])
        & (df[distance_col] <= distance_range[1])
    ]

    # Get the v_app, distance, and error values within the range
    v_app_values = df_filtered[v_app_col].values
    distance_values = df_filtered[distance_col].values
    error_values = df_filtered[error_col].values

    # Compute the 2D histogram for error
    heatmap, xedges, yedges = np.histogram2d(
        distance_values,  # distance goes on the x-axis
        v_app_values,     # v_app goes on the y-axis
        bins=[bins_distance, bins_v_app],  
        weights=error_values,
        density=False,
    )

    # Normalize the heatmap by counting occurrences in each bin
    counts, _, _ = np.histogram2d(
        distance_values, v_app_values, bins=[bins_distance, bins_v_app]
    )
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        heatmap = np.true_divide(heatmap, counts)
        heatmap[counts == 0] = np.nan  # Set bins with no data to 0

    # First subplot: Heatmap using imshow
    im = ax.imshow(
        heatmap.T,
        origin="lower",
        # aspect="auto",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        cmap="viridis",
    )
    fig.colorbar(im, ax=ax)
    ax.set_xlabel(r"Distance")
    ax.set_ylabel(r"Robot approach speed")


def plot_marginal(
    ax,
    df: pd.DataFrame,
    distance_col: str,
    error_col: str,
    distance_range: tuple,
    bins_distance: int = 10,
):
    df_filtered = df[
        (df[distance_col] >= distance_range[0])
        & (df[distance_col] <= distance_range[1])
    ]
        
    # Second subplot: Error as a function of distance with bins, mean, and standard deviation
    distance_bins = np.linspace(distance_range[0], distance_range[1], bins_distance + 1)

    # Ensure using .loc to avoid SettingWithCopyWarning
    df_de = df_filtered.copy()
    df_de.loc[:, 'distance_bin'] = pd.cut(df_de[distance_col], bins=distance_bins)

    # Group by the distance bins and calculate mean and standard deviation
    grouped = df_de.groupby('distance_bin', observed=False)[error_col].agg(['mean', 'std'])
    bin_centers = (distance_bins[:-1] + distance_bins[1:]) / 2  # Get bin centers

    # Plot mean error with standard deviation as error bars
    ax.errorbar(
        bin_centers, grouped['mean'], yerr=grouped['std'], fmt='o', capsize=5, label='Mean Error with Std Dev'
    )
    ax.set_xlabel(r"Distance")
    ax.set_ylabel(r"Mean Error")

    ### (i) Cumulative Mean Error Based on Binned Data
    cumulative_mean_error_bins = [
        df_filtered[df_filtered[distance_col] <= dist][error_col].mean() for dist in bin_centers
    ]
    ax.plot(bin_centers, cumulative_mean_error_bins, label='Cumulative Mean Error (Binned)', linestyle='--', color='green')

    # find where the cumulative mean error is greater than 0.5 for the lowest distance
    idx = np.argmax(np.array(cumulative_mean_error_bins) > 0.1)
    ax.text(bin_centers[idx] + 0.05, 0.85, r'$r=$' + f'{bin_centers[idx]:.2f}', color='r')
    ax.axvline(x=bin_centers[idx], color='r', linestyle='--', label='Threshold Distance')

    ax.axhline(y=0.05, color='r', linestyle='--', label='Threshold Error')

if __name__ == "__main__":
    df = pd.read_csv("analysis/clean/measurements.csv")

    # TEST
    df = compute_related_freezing(df)

    output_sub_folder = "analysis/clean/figs/error"
    if not os.path.exists(output_sub_folder):
        os.makedirs(output_sub_folder)

    delta = 2
    list_threshold_es = [0.0] # [2.0]  # [1.0, 2.0]
    list_weight_es = [1.0]  # [0.5, 1.0]
    list_exp_id = df["experiment_id"].unique()

    for threshold_es in list_threshold_es:
        for weight_es in list_weight_es:
            output_folder = f"{output_sub_folder}/threshold={threshold_es}_weight={weight_es}"
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            df_all = df.copy()
            df_all = compute_error(df_all, delta=delta, threshold_es=threshold_es, weight_es=weight_es)
            output_filename = f"model_error_all"
            plot_error(df_all, output_filename, output_folder)
            for exp_id in list_exp_id:
                df_exp = df[df["experiment_id"] == exp_id]
                df_exp = df_exp.copy()  
                df_exp = compute_error(df_exp, delta=delta, threshold_es=threshold_es, weight_es=weight_es)
                output_filename = f"model_error_exp={exp_id}"
                plot_error(df_exp, output_filename, output_folder)

