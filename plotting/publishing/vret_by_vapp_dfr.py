import os
import numpy as np

from core.experiment import Experiment
from core.study import study

from analysis.full.get_peak_data import get_peaks_data
from analysis.full.run_reg_fe import get_sampled_dataframe

from scipy.ndimage import gaussian_filter, generic_filter

import matplotlib.pyplot as plt

import utils.plotter as plotter

from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

def nanmean_filter(values):
    """Custom filter function to calculate the mean, ignoring NaNs."""
    non_nan_values = values[~np.isnan(values)]
    if non_nan_values.size > 0:
        return np.mean(non_nan_values)
    else:
        return np.nan


def plot_heatmap(dataframe, bins=20, range_apv=None, range_distance=None):
    distance = dataframe['distance'].to_numpy()
    apv = dataframe['apv'].to_numpy()
    rv_rev = dataframe['rev_shifted'].to_numpy()

    apv_min = -20 # apv.min()
    apv_max = apv.max()

    distance_min = distance.min()
    distance_max = 40 # distance.max()

    # Define bin edges
    apv_bins = np.linspace(range_apv[0], range_apv[1], bins + 1) if range_apv else np.linspace(apv_min, apv_max, bins + 1)
    distance_bins = np.linspace(range_distance[0], range_distance[1], bins + 1) if range_distance else np.linspace(distance_min, distance_max, bins + 1)
    
    # Digitize the data into bins
    apv_idx = np.digitize(apv, apv_bins) - 1
    distance_idx = np.digitize(distance, distance_bins) - 1
    
    # Initialize an array to hold the mean rv_rev values and counts
    heatmap = np.full((bins, bins), np.nan)
    counts = np.zeros((bins, bins))
    
    # Calculate the sum of rv_rev and counts for each bin
    for i in range(len(rv_rev)):
        if 0 <= apv_idx[i] < bins and 0 <= distance_idx[i] < bins:
            if np.isnan(heatmap[distance_idx[i], apv_idx[i]]):
                heatmap[distance_idx[i], apv_idx[i]] = rv_rev[i]
            else:
                heatmap[distance_idx[i], apv_idx[i]] += rv_rev[i]
            counts[distance_idx[i], apv_idx[i]] += 1
    
    # # Calculate the mean rv_rev for each bin
    # with np.errstate(invalid='ignore'):
    #     mean_rv_rev = mean_rv_rev / counts
    # # set nan to zero
    # mean_rv_rev[np.isnan(mean_rv_rev)] = 0
    # mean_rv_rev = gaussian_filter(mean_rv_rev, sigma=2.0)

    with np.errstate(divide='ignore', invalid='ignore'):
        heatmap = np.true_divide(heatmap, counts)
        heatmap[counts == 0] = np.nan  # Set bins with no data to 0
    
    # Use a filter to fill in NaN bins with neighborhood averages
    heatmap_nanfiltered = generic_filter(heatmap, nanmean_filter, size=3, mode='constant', cval=np.nan)
    heatmap_smoothed = gaussian_filter(np.nan_to_num(heatmap_nanfiltered, nan=0), sigma=2.0, mode='reflect')

    # Create the figure
    fig = plt.figure(figsize=(6, 4))

    # Create a GridSpec layout for alignment
    gs = GridSpec(2, 2, width_ratios=[3.5, 1.5], height_ratios=[1, 1], wspace=0.2)
    
    # Main heatmap on the left (spanning the full height)
    ax_main = fig.add_subplot(gs[:, 0])  # Main heatmap spans the full height

    # Print number of observations of heatmap
    print(f"Number of observations: {np.sum(counts)}")

    # Plot the heatmap
    im_main = ax_main.imshow(heatmap_smoothed.T, origin='lower', aspect='auto',
                    extent=[distance_bins[0], distance_bins[-1], apv_bins[0], apv_bins[-1]],
                    cmap='viridis', interpolation='nearest')
    ax_main.set_xlabel(r"fish-robot distance $d_{FR}$ (cm)", fontsize=12)
    ax_main.set_ylabel(r"robot approach speed $v_{app}^R$ (cm/s)", fontsize=12)

    ax_main.text(0.02, 1.05, r"\textbf{A.}", fontsize=12, weight='bold', 
                transform=ax_main.transAxes, verticalalignment='center', horizontalalignment='center')

    # Smaller heatmap on the top right
    ax_smaller = fig.add_subplot(gs[0, 1])

    main_max = np.nanmax(heatmap_smoothed)  # Find the maximum value of the main heatmap
    main_min = np.nanmin(heatmap_smoothed)  # Find the minimum value of the main heatmap

    heatmap_clipped = np.clip(heatmap, main_min, main_max)  # Clip values above the main max

    # Plot the smaller heatmap
    im_smaller = ax_smaller.imshow(heatmap_clipped.T, origin='lower', cmap='viridis', aspect='auto',
                            extent=[distance_bins[0], distance_bins[-1], apv_bins[0], apv_bins[-1]],
                            interpolation='nearest')
    # Annotate as B
    ax_smaller.text(0.02, 1.125, r"\textbf{B.}", fontsize=12, weight='bold', 
                 transform=ax_smaller.transAxes, verticalalignment='center', horizontalalignment='center')
    

    # Create vertical colorbar to the right, spanning both heatmaps
    ax_colorbar = fig.add_subplot(gs[1, 1])
    fig.colorbar(im_main, cax=ax_colorbar, orientation='vertical')
    # make colorbar smaller in width
    ax_colorbar.set_aspect(0.5)
    # move to the left of the space
    ax_colorbar.set_position([0.66, 0.12, 0.1, 0.325])
    # # label title colorbar horizontal on the right
    # ax_colorbar.yaxis.set_label_position("right")
    # # move it a bit up
    # ax_colorbar.set_ylabel('average \n acceleration (fish)', rotation=0, labelpad=40)

    ax_colorbar.text(2.25, 0.35, 'average \n fish avoidance \n speed \n' + r' $\bar{v}_{avoid}^F$', 
                     horizontalalignment='left', fontsize=10, fontweight='bold', transform=ax_colorbar.transAxes)
    
    # Plot settings

    fig.savefig(f"{study.plot_output_folder}/PUBLISH/vret_vapp_dfr.pdf")


if __name__ == "__main__":
    data = study.data

    control = data.copy()
    control['rev_shifted'] = control['rev'].shift(-8)
    control = control.dropna()

    plot_heatmap(control)