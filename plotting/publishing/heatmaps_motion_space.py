from matplotlib.path import Path
import numpy as np

from core.experiment import Experiment
from core.study import study
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter, generic_filter

import utils.plotter as plotter

from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable


def relative_position(control):
    # Fish at origin
    x_rel = control["x_robot"] - control["x_fish"]
    y_rel = control["y_robot"] - control["y_fish"]

    heading_fish = control["heading_fish"]
    heading_fish_adjusted = heading_fish - np.pi / 2

    cos_theta = np.cos(heading_fish_adjusted)
    sin_theta = np.sin(heading_fish_adjusted)
    
    x_rot = cos_theta * x_rel + sin_theta * y_rel
    y_rot = -sin_theta * x_rel + cos_theta * y_rel

    x_rot = x_rel
    y_rot = y_rel

    return x_rot, y_rot

def nanmean_filter(values):
    """Custom filter function to calculate the mean, ignoring NaNs."""
    non_nan_values = values[~np.isnan(values)]
    if non_nan_values.size > 0:
        return np.mean(non_nan_values)
    else:
        return np.nan


def create_heatmap(x_rel, y_rel, a_f, bins=100, range_limit=(-20, 20), sigma=5.0):
    # Clip acceleration values
    # a_f = a_f.clip(lower=-20, upper=100)

    within_range = (x_rel >= range_limit[0]) & (x_rel <= range_limit[1]) & (y_rel >= range_limit[0]) & (y_rel <= range_limit[1])
    
    x_rel = x_rel[within_range]
    y_rel = y_rel[within_range]
    a_f = a_f[within_range]

    heatmap, xedges, yedges = np.histogram2d(x_rel, y_rel, bins=bins, range=[[range_limit[0], range_limit[1]], [range_limit[0], range_limit[1]]], weights=a_f, density=False)
    counts, _, _ = np.histogram2d(x_rel, y_rel, bins=bins, range=[[range_limit[0], range_limit[1]], [range_limit[0], range_limit[1]]])
    
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        heatmap = np.true_divide(heatmap, counts)
        heatmap[counts == 0] = np.nan  # Set bins with no data to 0
    
    if sigma > 0:
        # Use a filter to fill in NaN bins with neighborhood averages
        heatmap = generic_filter(heatmap, nanmean_filter, size=3, mode='constant', cval=np.nan)

        heatmap_smoothed = gaussian_filter(np.nan_to_num(heatmap, nan=0), sigma=sigma, mode='constant', cval=0)
        # Restore NaNs where counts were zero
        # heatmap_smoothed[np.isnan(heatmap)] = np.nan
        heatmap = heatmap_smoothed
    
    # # Apply Gaussian smoothing to the heatmap, ignoring NaNs
    # if sigma > 0:
    #     heatmap_smoothed = gaussian_filter(np.nan_to_num(heatmap, nan=0), sigma=sigma, mode='constant', cval=0)
    #     # Restore NaNs where counts were zero
    #     heatmap_smoothed[np.isnan(heatmap)] = np.nan
    #     heatmap = heatmap_smoothed

    #     # Use a filter to fill in NaN bins with neighborhood averages
    #     heatmap = generic_filter(heatmap, nanmean_filter, size=3, mode='constant', cval=np.nan)

    return heatmap, xedges, yedges

def get_heatmap(control, type, rotated, sigma, bins):
    x_rel = control["x_robot"] - control["x_fish"]
    y_rel = control["y_robot"] - control["y_fish"]

    if rotated:
        heading_fish = control["heading_fish"]
        heading_fish_adjusted = heading_fish + np.pi / 2

        cos_theta = np.cos(heading_fish_adjusted)
        sin_theta = np.sin(heading_fish_adjusted)
        
        x_rot = cos_theta * x_rel + sin_theta * y_rel
        y_rot = -sin_theta * x_rel + cos_theta * y_rel
    else:
        x_rot = x_rel
        y_rot = y_rel

    a_f = control[type].to_numpy()

    heatmap, xedges, yedges = create_heatmap(x_rot, y_rot, a_f, sigma=sigma, bins=bins)

    return heatmap, xedges, yedges

def plot_motion_heatmap(control, type="acceleration_fish", rotated=False):
    # Create the figure
    fig = plt.figure(figsize=(6, 4))

    # Create a GridSpec layout for alignment
    gs = GridSpec(2, 2, width_ratios=[3, 1.5], height_ratios=[1, 0.6], wspace=0.2)

    # Main heatmap on the left (spanning the full height)
    ax_main = fig.add_subplot(gs[:, 0])  # Main heatmap spans the full height

    heatmap_main, xedges, yedges = get_heatmap(control, type, rotated, sigma=1, bins=50)
    main_max = np.nanmax(heatmap_main)  # Find the maximum value of the main heatmap
    main_min = np.nanmin(heatmap_main)  # Find the minimum value of the main heatmap

    # save heatmap_main
    np.save(f"{study.plot_output_folder}/PUBLISH/heatmap_{type}.npy", heatmap_main)

    # Plot the main heatmap
    im_main = ax_main.imshow(heatmap_main.T, origin='lower', cmap='viridis', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])

    # Add contour lines to the main heatmap
    X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
    levels = np.linspace(np.nanmin(heatmap_main), np.nanmax(heatmap_main), num=3)
    ax_main.contour(X, Y, heatmap_main.T, levels=levels, colors='white', linewidths=2.0)

    # Add an arrow indicating fish orientation
    if rotated:
        ax_main.arrow(0, 0, 0, 2.5, head_width=0.7, head_length=1, fc='black', ec='black', alpha=0.8)
    else:
        ax_main.arrow(0, 0, 2.5, 0, head_width=0.7, head_length=1, fc='black', ec='black', alpha=0.8)

    # Add small circle at the origin
    ax_main.plot(0, 0, 'o', color='black', markersize=3)

    ax_main.set_xlabel(r'relative $x$ position (robot)')
    ax_main.set_ylabel(r'relative $y$ position (robot)')

    # X-ticks by 5
    ax_main.set_xticks(np.arange(-20, 21, 5))

    # Annotate as A
    ax_main.text(0.02, 1.05, r"\textbf{A.}", fontsize=12, weight='bold', 
                 transform=ax_main.transAxes, verticalalignment='center', horizontalalignment='center')

    # Smaller heatmap on the top right
    ax_smaller = fig.add_subplot(gs[0, 1])

    heatmap_smaller, xedges, yedges = get_heatmap(control, type, rotated, sigma=0, bins=50)
    heatmap_smaller_clipped = np.clip(heatmap_smaller, main_min, main_max)  # Clip values above the main max

    # Plot the smaller heatmap
    im_smaller = ax_smaller.imshow(heatmap_smaller_clipped.T, origin='lower', cmap='viridis', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])

    # put the y-ticks of the im_smaller on the right
    # ax_smaller.yaxis.tick_right()
    # ax_smaller.yaxis.set_label_position("right")

    # if rotated:
    #     ax_smaller.arrow(0, 0, 0, 2.5, head_width=0.7, head_length=1, fc='black', ec='black', alpha=0.8)
    # else:
    #     ax_smaller.arrow(0, 0, 2.5, 0, head_width=0.7, head_length=1, fc='black', ec='black', alpha=0.8)
    # ax_smaller.plot(0, 0, 'o', color='black', markersize=3)

    # Annotate as B
    ax_smaller.text(0.02, 1.125, r"\textbf{B.}", fontsize=12, weight='bold', 
                 transform=ax_smaller.transAxes, verticalalignment='center', horizontalalignment='center')

    # Create vertical colorbar to the right, spanning both heatmaps
    ax_colorbar = fig.add_subplot(gs[1, 1])
    fig.colorbar(im_main, cax=ax_colorbar, orientation='vertical')
    # make colorbar smaller in width
    ax_colorbar.set_aspect(0.5)
    # move to the left of the space
    ax_colorbar.set_position([0.625, 0.15, 0.1, 0.25])
    # # label title colorbar horizontal on the right
    # ax_colorbar.yaxis.set_label_position("right")
    # # move it a bit up
    # ax_colorbar.set_ylabel('average \n acceleration (fish)', rotation=0, labelpad=40)

    ax_colorbar.text(4.5, 0.35, 'average \n fish avoidance \n acceleration \n' + r' $\bar{a}_{avoid}^F$', 
                     horizontalalignment='left', fontsize=10, fontweight='bold', transform=ax_colorbar.transAxes)


    # Adjust the layout to ensure everything fits
    plt.tight_layout()

    # Save the figure
    fig.savefig(f"{study.plot_output_folder}/PUBLISH/motion_{type}.pdf")
    

def create_frequency_heatmap(x_rot, y_rot, bins=100, range_limit=(-20, 20), sigma=0):
    # Create a 2D histogram without weights, just counting occurrences
    heatmap, xedges, yedges = np.histogram2d(x_rot, y_rot, bins=bins, range=[[range_limit[0], range_limit[1]], [range_limit[0], range_limit[1]]])
    counts, _, _ = np.histogram2d(x_rot, y_rot, bins=bins, range=[[range_limit[0], range_limit[1]], [range_limit[0], range_limit[1]]])
    
    # Set bins with zero counts to NaN
    heatmap[heatmap == 0] = np.nan
    
    # Apply Gaussian smoothing to the heatmap, ignoring NaNs
    if sigma > 0:
        # Use a filter to fill in NaN bins with neighborhood averages
        heatmap = generic_filter(heatmap, nanmean_filter, size=3, mode='constant', cval=np.nan)

        heatmap_smoothed = gaussian_filter(np.nan_to_num(heatmap, nan=0), sigma=sigma, mode='constant', cval=0)
        # Restore NaNs where counts were zero
        # heatmap_smoothed[np.isnan(heatmap)] = np.nan
        heatmap = heatmap_smoothed

    # if sigma > 0:
    #     heatmap_smoothed = gaussian_filter(np.nan_to_num(heatmap, nan=0), sigma=sigma, mode='constant', cval=0)
    #     # Restore NaNs where counts were zero
    #     heatmap_smoothed[np.isnan(heatmap)] = np.nan
    #     heatmap = heatmap_smoothed

    #     # Use a filter to fill in NaN bins with neighborhood averages
    #     heatmap = generic_filter(heatmap, nanmean_filter, size=3, mode='constant', cval=np.nan)
    
    # # Normalize the heatmap to values between 0 and 1
    # heatmap_min = np.nanmin(heatmap)
    # heatmap_max = np.nanmax(heatmap)
    
    # if heatmap_max > heatmap_min:  # Only normalize if there's a valid range
    #     heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)

    return heatmap, xedges, yedges

def plot_spatial_heatmap(control, rotated=False):
    x_rel = control["x_robot"] - control["x_fish"]
    y_rel = control["y_robot"] - control["y_fish"]

    x_rot = x_rel
    y_rot = y_rel

    # Create the figure
    fig = plt.figure(figsize=(6, 4))

    # Create a GridSpec layout for alignment
    gs = GridSpec(2, 2, width_ratios=[3, 1.5], height_ratios=[1, 0.6], wspace=0.2)

    bins = 50

    # Main heatmap on the left (spanning the full height)
    ax_main = fig.add_subplot(gs[:, 0])  # Main heatmap spans the full height

    heatmap_main, xedges, yedges = create_frequency_heatmap(x_rot, y_rot, bins=bins, sigma=1)
    main_max = np.nanmax(heatmap_main)  # Find the maximum value of the main heatmap
    main_min = np.nanmin(heatmap_main)  # Find the minimum value of the main heatmap

    # Plot the main heatmap
    im_main = ax_main.imshow(heatmap_main.T, origin='lower', cmap='viridis', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])

    # Add contour lines to the main heatmap
    X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
    levels = np.linspace(np.nanmin(heatmap_main), np.nanmax(heatmap_main), num=3)
    ax_main.contour(X, Y, heatmap_main.T, levels=levels, colors='white', linewidths=2.0)

    # Add an arrow indicating fish orientation
    if rotated:
        ax_main.arrow(0, 0, 0, 2.5, head_width=0.7, head_length=1, fc='black', ec='black', alpha=0.8)
    else:
        ax_main.arrow(0, 0, 2.5, 0, head_width=0.7, head_length=1, fc='black', ec='black', alpha=0.8)

    # Add small circle at the origin
    ax_main.plot(0, 0, 'o', color='black', markersize=3)

    ax_main.set_xlabel(r'relative $x$ position (robot)')
    ax_main.set_ylabel(r'relative $y$ position (robot)')

    # Annotate as A
    ax_main.text(0.02, 1.05, r"\textbf{A.}", fontsize=12, weight='bold', 
                 transform=ax_main.transAxes, verticalalignment='center', horizontalalignment='center')

    # Smaller heatmap on the top right
    ax_smaller = fig.add_subplot(gs[0, 1])

    heatmap_smaller, xedges, yedges = create_frequency_heatmap(x_rot, y_rot, bins=bins, sigma=0)
    heatmap_smaller_clipped = np.clip(heatmap_smaller, main_min, main_max)  # Clip values above the main max

    # Plot the smaller heatmap
    im_smaller = ax_smaller.imshow(heatmap_smaller_clipped.T, origin='lower', cmap='viridis', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])

    # put the y-ticks of the im_smaller on the right
    # ax_smaller.yaxis.tick_right()
    # ax_smaller.yaxis.set_label_position("right")

    if rotated:
        ax_smaller.arrow(0, 0, 0, 2.5, head_width=0.7, head_length=1, fc='black', ec='black', alpha=0.8)
    else:
        ax_smaller.arrow(0, 0, 2.5, 0, head_width=0.7, head_length=1, fc='black', ec='black', alpha=0.8)

    ax_smaller.plot(0, 0, 'o', color='black', markersize=3)

    # Annotate as B
    ax_smaller.text(0.02, 1.125, r"\textbf{B.}", fontsize=12, weight='bold', 
                 transform=ax_smaller.transAxes, verticalalignment='center', horizontalalignment='center')

    # Create vertical colorbar to the right, spanning both heatmaps
    ax_colorbar = fig.add_subplot(gs[1, 1])
    # fig.colorbar(im_main, cax=ax_colorbar, orientation='vertical')
    # # make colorbar smaller in width and move to right of the space
    # ax_colorbar.set_aspect(0.5)
    # ax_colorbar.set_position([0.625, 0.15, 0.1, 0.25])

    # Use make_axes_locatable to adjust the colorbar size and position
    divider = make_axes_locatable(ax_colorbar)

    # Append a narrower colorbar to the right of ax_colorbar
    cax = divider.append_axes("left", size="5%", pad=0.01)
    fig.colorbar(im_main, cax=cax, orientation='vertical')

    # Turn off the axis of the colorbar subplot
    ax_colorbar.axis('off')
    # # label title colorbar horizontal on the right
    # ax_colorbar.yaxis.set_label_position("right")
    # # move it a bit up
    # ax_colorbar.set_ylabel('average \n acceleration (fish)', rotation=0, labelpad=40)

    ax_colorbar.text(0.25, 0.4, 'spatial \n frequency (robot)', 
                     horizontalalignment='left', fontsize=10, fontweight='bold', transform=ax_colorbar.transAxes)


    # Adjust the layout to ensure everything fits
    plt.tight_layout()

    plt.savefig(f"{study.plot_output_folder}/PUBLISH/spatial_heatmap.pdf")


if __name__ == "__main__":
    data = study.data
    control = data.copy()

    control["ax_fish"] = control["dx_fish"].diff() / 0.04
    control["ay_fish"] = control["dy_fish"].diff() / 0.04
    control["rea"] = control["ax_fish"] * control["x_fr_n"] + control["ay_fish"] * control["y_fr_n"]

    plot_motion_heatmap(control, type="rea")
    plot_spatial_heatmap(control)