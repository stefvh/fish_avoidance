import numpy as np

from core.experiment import Experiment
from core.study import study

from analysis.full.get_peak_data import get_peaks_data
from analysis.full.run_reg_fe import get_sampled_dataframe

import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter


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


def create_heatmap(x_rel, y_rel, a_f, bins=100, range_limit=(-20, 20), sigma=5.0):
    # Clip acceleration values
    # a_f = a_f.clip(lower=-20, upper=100)

    within_range = (x_rel >= range_limit[0]) & (x_rel <= range_limit[1]) & (y_rel >= range_limit[0]) & (y_rel <= range_limit[1])
    
    x_rel = x_rel[within_range]
    y_rel = y_rel[within_range]
    a_f = a_f[within_range]

    # # Create 2D histogram bins
    # heatmap = np.full((bins, bins), -np.inf)  # Initialize with -inf to find max values
    # xedges = np.linspace(range_limit[0], range_limit[1], bins + 1)
    # yedges = np.linspace(range_limit[0], range_limit[1], bins + 1)
    
    # # Digitize the positions to find which bin each point falls into
    # x_idx = np.digitize(x_rel, xedges) - 1
    # y_idx = np.digitize(y_rel, yedges) - 1
    
    # # Iterate through the data points
    # a_f = a_f.to_numpy()
    # for i in range(len(a_f)):
    #     if 0 <= x_idx[i] < bins and 0 <= y_idx[i] < bins:
    #         # Update the maximum value for each bin
    #         heatmap[x_idx[i], y_idx[i]] = max(heatmap[x_idx[i], y_idx[i]], a_f[i])
    
    # # Replace -inf with 0 (where no data was present)
    # heatmap[heatmap == -np.inf] = 0

    heatmap, xedges, yedges = np.histogram2d(x_rel, y_rel, bins=bins, range=[[range_limit[0], range_limit[1]], [range_limit[0], range_limit[1]]], weights=a_f, density=False)
    counts, _, _ = np.histogram2d(x_rel, y_rel, bins=bins, range=[[range_limit[0], range_limit[1]], [range_limit[0], range_limit[1]]])
    
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        heatmap = np.true_divide(heatmap, counts)
        heatmap[counts == 0] = np.nan  # Set bins with no data to 0

    # Apply Gaussian smoothing, ignoring NaNs
    if sigma > 0:
        heatmap_smoothed = gaussian_filter(np.nan_to_num(heatmap, nan=0), sigma=sigma, mode='constant', cval=0)
        
        # Restore NaNs to the smoothed heatmap where counts were zero
        heatmap_smoothed[np.isnan(heatmap)] = np.nan
        heatmap = heatmap_smoothed

    # # # Apply Gaussian smoothing
    # heatmap = gaussian_filter(heatmap, sigma=sigma,)  # Adjust sigma for more or less smoothing

    return heatmap, xedges, yedges


def normalize_heatmap(heatmap):
    heatmap_min = np.min(heatmap)
    heatmap_max = np.max(heatmap)
    
    # Normalize between 0 and 1
    if heatmap_max > heatmap_min:  # Avoid division by zero
        heatmap_normalized = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)
    else:
        heatmap_normalized = heatmap

    return heatmap_normalized


def plot_heatmap_rf(control):
    x_rel = control["x_robot"] - control["x_fish"]
    y_rel = control["y_robot"] - control["y_fish"]

    heading_fish = control["heading_fish"]
    cos_theta = np.cos(heading_fish)
    sin_theta = np.sin(heading_fish)
    
    x_rot = cos_theta * x_rel + sin_theta * y_rel
    y_rot = -sin_theta * x_rel + cos_theta * y_rel

    x_rot = x_rel
    y_rot = y_rel

    a_f = control["acceleration_fish"]

    heatmap, xedges, yedges = create_heatmap(x_rot, y_rot, a_f)

    # heatmap = normalize_heatmap(heatmap)

    plt.figure(figsize=(8, 6))
    plt.imshow(heatmap.T, origin='lower', cmap='viridis', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    plt.colorbar(label='Average Fish Acceleration')
    plt.xlabel('Relative X Position (Robot)')
    plt.ylabel('Relative Y Position (Robot)')
    plt.title('Heatmap of Robot Position Relative to Fish (Fish at Origin)')
    plt.savefig(f"{study.plot_output_folder}/fish_acceleration_heatmap_rf.png")


def plot_heatmap_fr(control):
    x_rel = control["x_fish"] - control["x_robot"]
    y_rel = control["y_fish"] - control["y_robot"]

    heading_fish = control["heading_fish"]
    heading_fish_adjusted = heading_fish - np.pi / 2

    cos_theta = np.cos(heading_fish_adjusted)
    sin_theta = np.sin(heading_fish_adjusted)
    
    x_rot = cos_theta * x_rel + sin_theta * y_rel
    y_rot = -sin_theta * x_rel + cos_theta * y_rel

    x_rot = x_rel
    y_rot = y_rel


    a_f = control["acceleration_fish"]

    heatmap, xedges, yedges = create_heatmap(x_rot, y_rot, a_f, sigma=2, bins=100)

    # heatmap = normalize_heatmap(heatmap)

    plt.figure(figsize=(8, 6))
    plt.imshow(heatmap.T, origin='lower', cmap='viridis', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    plt.colorbar(label='Average Fish Acceleration')
    plt.xlabel('Relative X Position (Fish)')
    plt.ylabel('Relative Y Position (Fish)')
    plt.title('Heatmap of Fish Position Relative to Robot (Robot at Origin)')
    plt.savefig(f"{study.plot_output_folder}/fish_acceleration_heatmap_fr.png")


def create_frequency_heatmap(x_rot, y_rot, bins=100, range_limit=(-20, 20)):
    # Create a 2D histogram without weights, just counting occurrences
    heatmap, xedges, yedges = np.histogram2d(x_rot, y_rot, bins=bins, range=[[range_limit[0], range_limit[1]], [range_limit[0], range_limit[1]]])
    
    return heatmap, xedges, yedges

def plot_spatial_heatmap(control):
    x_rot, y_rot = relative_position(control)

    heatmap, xedges, yedges = create_frequency_heatmap(x_rot, y_rot, bins=100)

    heatmap = gaussian_filter(heatmap, sigma=4,)

    # Plotting the frequency heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(heatmap.T, origin='lower', cmap='viridis', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    plt.colorbar(label='Spatial Frequency')
    plt.xlabel('Relative X Position (Robot)')
    plt.ylabel('Relative Y Position (Robot)')
    plt.title('Heatmap of Robot Position Relative to Fish (Fish at Origin)')
    # plt.xlim(-20, 20)
    # plt.ylim(-20, 20)
    plt.savefig(f"{study.plot_output_folder}/fish_spatial_heatmap.png")


def plot_arrows_binned(control, bins=20, range_limit=(-20, 20)):
    # Only consider rows of control where control["rv_rev"] > 1
    # control = control[control["rv_rev"] > 2]
    control = control[control["apv"] > 10]

    x_rel = control["x_fish"] - control["x_robot"]
    y_rel = control["y_fish"] - control["y_robot"]

    heading_fish = control["heading_robot"]
    # heading_fish = heading_fish + np.pi / 2

    cos_theta = np.cos(heading_fish)
    sin_theta = np.sin(heading_fish)
    
    x_rot = cos_theta * x_rel + sin_theta * y_rel
    y_rot = -sin_theta * x_rel + cos_theta * y_rel

    dx_fish = control["rv_dx_fish"].to_numpy()
    dy_fish = control["rv_dy_fish"].to_numpy()

    # Create a 2D histogram grid
    xedges = np.linspace(range_limit[0], range_limit[1], bins + 1)
    yedges = np.linspace(range_limit[0], range_limit[1], bins + 1)
    
    # Initialize arrays to store average directions
    avg_dx = np.zeros((bins, bins))
    avg_dy = np.zeros((bins, bins))
    counts = np.zeros((bins, bins))
    
    # Digitize the positions into bins
    x_idx = np.digitize(x_rot, xedges) - 1
    y_idx = np.digitize(y_rot, yedges) - 1
    
    # Sum up the direction vectors within each bin
    for i in range(len(x_rot)):
        if 0 <= x_idx[i] < bins and 0 <= y_idx[i] < bins:
            avg_dx[x_idx[i], y_idx[i]] += dx_fish[i]
            avg_dy[x_idx[i], y_idx[i]] += dy_fish[i]
            counts[x_idx[i], y_idx[i]] += 1
    
    # Average the direction vectors
    with np.errstate(invalid='ignore'):
        avg_dx = np.divide(avg_dx, counts, where=counts!=0)
        avg_dy = np.divide(avg_dy, counts, where=counts!=0)
    
    # Normalize the arrows to unit length
    magnitude = np.sqrt(avg_dx**2 + avg_dy**2)
    with np.errstate(invalid='ignore', divide='ignore'):
        avg_dx = np.divide(avg_dx, magnitude, where=magnitude!=0)
        avg_dy = np.divide(avg_dy, magnitude, where=magnitude!=0)
    
    # Create the meshgrid for plotting
    x_centers = (xedges[:-1] + xedges[1:]) / 2
    y_centers = (yedges[:-1] + yedges[1:]) / 2
    X, Y = np.meshgrid(x_centers, y_centers)

    # Calculate vector from bin center to origin
    vector_to_origin_x = -X
    vector_to_origin_y = -Y
    magnitude_vector_to_origin = np.sqrt(vector_to_origin_x**2 + vector_to_origin_y**2)
    
    # Calculate the dot product and normalize it
    dot_product = avg_dx * vector_to_origin_x + avg_dy * vector_to_origin_y
    dot_product_normalized = dot_product / magnitude_vector_to_origin
    
    # Now we need to ensure that the normalized dot product reflects the desired scaling:
    # -1 for arrows pointing directly to the origin, 1 for arrows pointing away.
    dot_product_normalized = dot_product_normalized / magnitude  # This adjusts the scaling
    
    # Normalize color scale to be between -1 (towards origin) and 1 (away from origin)
    colors = np.clip(dot_product_normalized, -1, 1)

    # Flip the color values
    colors = -colors

    # Mask out bins where counts == 0
    mask = counts > 0
    X, Y, avg_dx, avg_dy, colors = X[mask], Y[mask], avg_dx[mask], avg_dy[mask], colors[mask]
    

    plt.figure(figsize=(8, 6))
    
    # Plot the arrows using quiver
    plt.quiver(X, Y, avg_dx, avg_dy, colors, angles='xy', scale_units='xy', scale=1, cmap='coolwarm', clim=(-1, 1))
    # Add a colorbar
    plt.colorbar(label='Direction Relative to Origin (-1: towards origin, 1: away from origin)')
    
    # plt.quiver(X, Y, avg_dx, avg_dy, angles='xy', scale_units='xy', scale=1, color='blue')
    
    # Mark the origin where the fish is located
    plt.plot(0, 0, 'ro')  # Fish at the origin
    
    # Plot settings
    # Set the grid to align with the bin edges
    plt.xticks(xedges, rotation=90)
    plt.yticks(yedges)
    plt.grid(True)
    
    # Plot settings
    plt.xlabel('Relative X Position (Robot)')
    plt.ylabel('Relative Y Position (Robot)')
    plt.title('Binned and Normalized Average Direction of Fish Relative to Robot Position')
    plt.xlim(xedges[0], xedges[-1])
    plt.ylim(yedges[0], yedges[-1])
    
    plt.savefig(f"{study.plot_output_folder}/fish_arrows_binned.png")
    


def plot_arrows(control, downsample_factor=10, scale=10):
    control = control[control["apv"] > 2]
    control = control[control["rv_rev"] > 0]

    x_rel = control["x_fish"] - control["x_robot"]
    y_rel = control["y_fish"] - control["y_robot"]

    heading_fish = control["heading_robot"]
    heading_fish = heading_fish - np.pi / 2

    cos_theta = np.cos(heading_fish)
    sin_theta = np.sin(heading_fish)
    
    x_rot = cos_theta * x_rel + sin_theta * y_rel
    y_rot = -sin_theta * x_rel + cos_theta * y_rel

    dx_fish = control["rv_dx_fish"].to_numpy()
    dy_fish = control["rv_dy_fish"].to_numpy()

    x_rot = x_rot[::downsample_factor]
    y_rot = y_rot[::downsample_factor]
    dx_fish = dx_fish[::downsample_factor]
    dy_fish = dy_fish[::downsample_factor]
    

    plt.figure(figsize=(8, 6))
    
    # Plot the arrows using quiver
    plt.quiver(x_rot, y_rot, dx_fish, dy_fish, angles='xy', scale_units='xy', scale=scale, color='blue')
    
    # Mark the origin where the fish is located
    plt.plot(0, 0, 'ro')  # Fish at the origin
    
    # Plot settings
    plt.xlabel('Relative X Position (Robot)')
    plt.ylabel('Relative Y Position (Robot)')
    plt.title('Relative Position of Robot with Fish Direction Indicated')
    plt.xlim(-20, 20)
    plt.ylim(-20, 20)
    plt.grid(True)
    plt.savefig(f"{study.plot_output_folder}/fish_arrows.png")


if __name__ == "__main__":
    data = study.data

    control = data.copy()
    # plot_heatmap_rf(control)
    # plot_heatmap_fr(control)
    # plot_spatial_heatmap(control)

    sample_time_points, _ = get_peaks_data(
        func_bound=None,
        func_multiplier=1,
        acc_max_height=0,
        vel_max_height=0,
        acc_prominence=12.5,
        do_group=False,
    )
    rv_df, ra_df = get_sampled_dataframe(sample_time_points)
    # plot_arrows_binned(rv_df)
    plot_arrows(rv_df)

