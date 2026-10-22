import os
from matplotlib import cm, pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

from core.experiment import Experiment
from core.study import study
from plotting.publishing.subset_trials import get_trials_by_data


def plot_encounters(trials, output_filename: str, output_folder: str):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    marker_styles = [
        "o",
        "s",
        "D",
        "^",
        "v",
        "<",
        ">",
        "p",
        "h",
    ]  # Different markers for each trial

    for trial_idx, trial in enumerate(trials):
        marker_style = marker_styles[trial_idx % len(marker_styles)]
        plot_trial(ax, trial["data"], trial["m_trial_id"], marker_style)

    plt.savefig(f"{output_folder}/{output_filename}.pdf")


def plot_trial(ax, data: pd.DataFrame, trial_id, marker_style):
    distances = data["distance"].to_numpy()
    ef = np.argmin(distances)
    td = 10

    # Get the fish and robot positions and orientations at frame_turn
    np_heading_fish = data["heading_fish"].to_numpy()
    np_x_rel = data["x_robot"].to_numpy() - data["x_fish"].to_numpy()
    np_y_rel = data["y_robot"].to_numpy() - data["y_fish"].to_numpy()

    # np_heading_robot = data['heading_robot'].to_numpy()
    # np_speed_robot = data['speed_robot'].to_numpy()
    # robot_heading = np_heading_robot[frame_turn]
    # robot_speed = np_speed_robot[frame_turn]

    # Generate colormap from light to dark
    num_frames = 2 * td  # Total number of frames to plot
    cmap = plt.get_cmap(
        "viridis"
    )  # You can choose other colormaps like 'plasma' or 'inferno'
    colors = cmap(
        np.linspace(0, 1, num_frames)
    )  # Generate colors that range from light to dark

    # Get robot's velocity (dx_robot and dy_robot)
    np_dx_robot = data['dx_robot'].to_numpy()  # Robot velocity in x direction
    np_dy_robot = data['dy_robot'].to_numpy()  # Robot velocity in y direction

    np_xrr, np_yrr = rotate_to_fish_frame(np_heading_fish, np_x_rel, np_y_rel)

    # Assuming turn_fish is in radians and gives the direction of turn
    np_turn_fish = data['turn_heading_fish'].to_numpy()

    np_dxr, np_dyr = rotate_to_fish_frame(np_heading_fish, np_dx_robot, np_dy_robot)
    # Normalize to have unit length for each row
    np_dxr /= np.linalg.norm([np_dxr, np_dyr], axis=0)
    np_dyr /= np.linalg.norm([np_dxr, np_dyr], axis=0)

    for frame_idx, frame in enumerate(range(ef - td, ef + td)):
        fish_heading = np_heading_fish[frame]

        # Calculate the relative robot position to the fish
        x_rel = np_x_rel[frame]
        y_rel = np_y_rel[frame]

        # Rotate the relative position to align with the fish's orientation
        # x_rel_rotated, y_rel_rotated = rotate_to_fish_frame(fish_heading, x_rel, y_rel)
        x_rel_rotated = np_xrr[frame]
        y_rel_rotated = np_yrr[frame]

        # Plot the rotated positions with a color that progresses over time
        ax.scatter(
            x_rel_rotated,
            y_rel_rotated,
            color=colors[frame_idx],
            s=20,
            marker=marker_style,
        )

        # Rotate the robot's velocity (dx_robot, dy_robot) to the fish's orientation
        dx_rotated = np_dxr[frame]
        dy_rotated = np_dyr[frame]

        # dx_rotated, dy_rotated = rotate_to_fish_frame(fish_heading, np_dx_robot[frame], np_dy_robot[frame])

        # Normalize dx, dy to have unit length
        # dx_rotated /= np.linalg.norm([dx_rotated, dy_rotated])
        # dy_rotated /= np.linalg.norm([dx_rotated, dy_rotated])

        # Add arrows to represent robot movement direction and speed
        ax.quiver(x_rel_rotated, y_rel_rotated, dx_rotated, dy_rotated, angles='xy', scale_units='xy', scale=2, 
                  color=colors[frame_idx], alpha=0.7, headwidth=3, headlength=5)

        # Plot focal fish at (0, 0)
        ax.scatter(0, 0, color="red", s=20, marker="o")

        ax.text(x_rel_rotated, y_rel_rotated, str(trial_id), fontsize=6, color='black')

        if np_turn_fish[frame] > np.pi / 12:
            turn_direction = np_turn_fish[frame] 
            ax.quiver(x_rel_rotated, y_rel_rotated, np.cos(turn_direction), np.sin(turn_direction),
                angles='xy', scale_units='xy', scale=2, color='red', headwidth=3, headlength=5, alpha=0.8)

    # Add labels and title if needed
    ax.set_xlabel("Rotated X (Relative to fish)")
    ax.set_ylabel("Y (Relative to fish)")
    ax.set_title("Rotated Relative Robot Positions")


def rotate_to_fish_frame(fish_heading, x, y):
    """Rotate (x, y) coordinates such that fish is at origin and heading to the right (0 radians)."""
    cos_theta = np.cos(-fish_heading)
    sin_theta = np.sin(-fish_heading)
    x_rotated = x * cos_theta - y * sin_theta
    y_rotated = x * sin_theta + y * cos_theta
    return x_rotated, y_rotated


if __name__ == "__main__":
    df = pd.read_csv("analysis/clean/measurements.csv")

    output_folder = "analysis/clean/figs/encounter_in_space"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # list_exp_id = df["experiment_id"].unique()

    exp_trials = get_trials_by_data(df, study.list_experiments)

    for exp_id, trials in enumerate(exp_trials):
        output_filename = f"encounters_exp={exp_id}"

        plot_encounters(trials, output_filename, output_folder)
