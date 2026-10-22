import os
import numpy as np

from core.experiment import Experiment
from core.study import study

from analysis.full.get_peak_data import get_peaks_data
from analysis.full.run_reg_fe import get_sampled_dataframe

from scipy.ndimage import gaussian_filter

import matplotlib.pyplot as plt

def plot_heatmap(dataframe, bins=20, range_apv=None, range_distance=None):
    distance = dataframe['distance'].to_numpy()
    apv = dataframe['apv'].to_numpy()
    rv_rev = dataframe['rev_shifted'].to_numpy()

    # Define bin edges
    apv_bins = np.linspace(range_apv[0], range_apv[1], bins + 1) if range_apv else np.linspace(apv.min(), apv.max(), bins + 1)
    distance_bins = np.linspace(range_distance[0], range_distance[1], bins + 1) if range_distance else np.linspace(distance.min(), distance.max(), bins + 1)
    
    # Digitize the data into bins
    apv_idx = np.digitize(apv, apv_bins) - 1
    distance_idx = np.digitize(distance, distance_bins) - 1
    
    # Initialize an array to hold the mean rv_rev values and counts
    mean_rv_rev = np.full((bins, bins), np.nan)
    counts = np.zeros((bins, bins))
    
    # Calculate the sum of rv_rev and counts for each bin
    for i in range(len(rv_rev)):
        if 0 <= apv_idx[i] < bins and 0 <= distance_idx[i] < bins:
            if np.isnan(mean_rv_rev[distance_idx[i], apv_idx[i]]):
                mean_rv_rev[distance_idx[i], apv_idx[i]] = rv_rev[i]
            else:
                mean_rv_rev[distance_idx[i], apv_idx[i]] += rv_rev[i]
            counts[distance_idx[i], apv_idx[i]] += 1
    
    # Calculate the mean rv_rev for each bin
    with np.errstate(invalid='ignore'):
        mean_rv_rev = mean_rv_rev / counts

    # set nan to zero
    mean_rv_rev[np.isnan(mean_rv_rev)] = 0

    fig, ax = plt.subplots(figsize=(8, 6))

    mean_rv_rev = gaussian_filter(mean_rv_rev, sigma=2.0)
    
    # Plot the heatmap
    cax = ax.imshow(mean_rv_rev.T, origin='lower', aspect='auto',
                    extent=[distance_bins[0], distance_bins[-1], apv_bins[0], apv_bins[-1]],
                    cmap='viridis', interpolation='nearest')

    # Add colorbar
    cbar = fig.colorbar(cax, ax=ax, label='$v_{ret}$')
    
    # Plot settings
    ax.set_title(r"fish retreat speed $v_{ret}$")
    ax.set_xlabel(r"fish-robot distance $d_{fr}$")
    ax.set_ylabel(r"robot approach speed $v_{app}$")
    fig.savefig(f"{study.plot_output_folder}/vretSHIFT_dfr_vapp.pdf")


if __name__ == "__main__":
    data = study.data

    control = data.copy()
    control['rev_shifted'] = control['rev'].shift(-8)
    control = control.dropna()

    sample_time_points, _ = get_peaks_data(
        func_bound=None,
        func_multiplier=1,
        acc_max_height=0,
        vel_max_height=0,
        acc_prominence=15,
        do_group=False,
    )
    rv_df, ra_df = get_sampled_dataframe(sample_time_points)
    plot_heatmap(control)