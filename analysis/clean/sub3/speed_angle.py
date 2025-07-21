from matplotlib import pyplot as plt
import numpy as np

from core.experiment import Experiment
from core.study import study

from analysis.clean.reactions import extract_trials
from analysis.clean.sub3.help import save_plot

import seaborn as sns

import utils.plotter

def plot_post_angle_speed(exp_trials):
    # Initialize lists to hold combined angle_fr and fish_speed values
    angles = []
    speeds = []
    
    # Calculate the number of experiments and set up the figure
    n_experiments = len(exp_trials)
    fig = plt.figure()

    # Iterate over each experiment
    for exp_idx, trials in enumerate(exp_trials):
        for trial in trials:
            # Get data up to end of approach index
            end_idx = trial["end_approach_index"]
            trial_data = trial["data"]
            
            # Extract post-approach data (10 frames after end approach index)
            post_approach_data = trial_data.iloc[end_idx+6:end_idx + 10]
            
            # If post-approach data exists, get angle_rf and fish_speed
            if not post_approach_data.empty:
                max_speed = post_approach_data["speed_fish"].max()
                max_speed_idx = post_approach_data["speed_fish"].idxmax()
                max_angle = post_approach_data.loc[max_speed_idx, "angle_rf"]
                
                # Append to combined lists
                angles.append(max_angle)
                speeds.append(max_speed)

    ax = fig.add_subplot(111, polar=True)
    # ax_polar_combined.scatter(angles, speeds, color='orange', alpha=0.7)
    ax.set_xlabel(r'relative angle (rad)')
    # ax.set_ylabel('Fish Speed')

    # Customize angular ticks to show radians instead of degrees
    radian_ticks = np.linspace(0, 2 * np.pi, 9)  # Define tick positions in radians
    tick_labels = [
        r"$0$" if tick == 0 else  # Hide 2π duplicate label at 0
        r"$\frac{1}{4}\pi$" if tick == 0.25 * np.pi else
        r"$\frac{1}{2}\pi$" if tick == 0.5 * np.pi else
        r"$\frac{3}{4}\pi$" if tick == 0.75 * np.pi else
        r"$\pi$" if tick == np.pi else
        r"$\frac{5}{4}\pi$" if tick == 1.25 * np.pi else
        r"$\frac{3}{2}\pi$" if tick == 1.5 * np.pi else
        r"$\frac{7}{4}\pi$" if tick == 1.75 * np.pi else
        r"" if tick == 2 * np.pi else
        ""
        for tick in radian_ticks
    ]
    tick_label_size = 12
    ax.set_xticks(radian_ticks)
    ax.set_xticklabels(tick_labels, fontsize=tick_label_size)  # Apply custom tick labels


    # Define bins for speed (radial) and angle (angular)
    speed_bins = 6
    angle_bins = 24
    speed_edges = np.linspace(0, max(speeds), speed_bins + 1)  # 6 edges for 5 bins
    angle_edges = np.linspace(0, 2 * np.pi, angle_bins + 1)  # 13 edges for 12 bins

    # Create a 2D histogram with specified bins
    angles = np.array(angles)
    # angles = np.where(angles < 0, -np.pi - angles, angles)
    # angles = np.where(angles > 0, np.pi - angles, angles)
    hist, _, _ = np.histogram2d(angles, speeds, bins=[angle_edges, speed_edges])

    # Density maximum to 1
    hist = hist / hist.max()

    # Use the edges directly for r and theta so that they match the required shape for pcolormesh
    theta, r = np.meshgrid(angle_edges, speed_edges)  # r has shape (6,), theta has shape (13,)

    # # Plot the histogram data using pcolormesh for the density overlay
    pcm = ax.pcolormesh(theta, r, hist.T, shading='auto', cmap='Blues', alpha=1.0)
    # # pcm.set_array(np.ma.masked_where(hist.T == 0, hist.T))  # Mask empty bins

    # Move radial labels to 3/8π (67.5 degrees)
    ax.set_rlabel_position(67.5)

    # Format radial ticks (fish speed) using LaTeX
    radial_ticks = np.linspace(0, max(speeds), 7)  # 6 ticks for 5 bins
    radial_labels = ["", r"5", r"10", r"15", r"20", r"25", r""]
    ax.set_yticks(radial_ticks)
    radial_label_size = 10
    ax.set_yticklabels(radial_labels, fontsize=radial_label_size)

    # Add a rounded "Fish Speed" label at 67.5 degrees (3/8π)
    angle = 67.5  # degrees
    radius = max(speeds) + 2.0  # Slightly outside the maximum radius
    ax.text(
        np.radians(angle), radius, r"fish speed (cm/s)",  # Use LaTeX for the label
        fontsize=radial_label_size, va="center", ha="center",
        rotation=angle - 90,  # Rotate the text tangentially
        rotation_mode="anchor",
        transform=ax.transData
    )
    # Color bar for density
    fig.colorbar(pcm, ax=ax, label="Density")

    plt.tight_layout()
    save_plot(fig, "post_angle_speed.pdf")


if __name__ == "__main__":
    exp_trials = extract_trials()
    # distance_bins = list(range(40, -5, -5))  # Generates [40, 35, 30, ..., 5, 0]

    plot_post_angle_speed(exp_trials)

