import os
from matplotlib import cm, pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from analysis.one.wrapper_fixed import run_plm_model
from analysis.one.wrapper_mixed import run_lmer_model
from core.experiment import Experiment
from core.study import study

from analysis.clean.reactions import extract_trials
from analysis.clean.sub3.help import save_plot

from analysis.clean.avoidance_mechanisms import get_data

import seaborn as sns

import utils.plotter

from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def plot_combined(exp_trials):
        # Prepare data for both subplots
    angles = []
    speeds = []
    displacements = []
    end_speeds = []
    cv_speeds = []

    # Extract data for each trial
    for trials in exp_trials:
        for trial in trials:
            trial_data = trial["data"]
            end_idx = trial["end_approach_index"]

            # Extract post-approach data (10 frames after end approach index)
            post_data = trial_data.iloc[end_idx + 1 : end_idx + 10]

            # If post-approach data exists, get angle_rf and fish_speed
            if not post_data.empty:
                max_speed = post_data["speed_fish"].max()
                max_speed_idx = post_data["speed_fish"].idxmax()
                max_angle = post_data.loc[max_speed_idx, "angle_rf"]

                # Append to combined lists
                angles.append(max_angle)
                speeds.append(max_speed)

            # Extract post-phase data for both plots
            post_data = trial_data.iloc[end_idx + 1 : end_idx + 1 + 20]

            # For displacement plot
            fish_start = post_data.iloc[0][["x_fish", "y_fish"]].values
            fish_end = post_data.iloc[-1][["x_fish", "y_fish"]].values
            robot_start = post_data.iloc[0][["x_robot", "y_robot"]].values
            robot_end = post_data.iloc[-1][["x_robot", "y_robot"]].values

            dx_start = fish_start[0] - robot_start[0]
            dy_start = fish_start[1] - robot_start[1]

            dx_end = fish_end[0] - robot_end[0]
            dy_end = fish_end[1] - robot_end[1]

            dx_dsp = dx_end - dx_start
            dy_dsp = dy_end - dy_start
            total_displacement = np.sqrt(dx_dsp**2 + dy_dsp**2)
            displacements.append(total_displacement)

            # For speed plot
            end_speeds.append(post_data["speed_fish"].iloc[-1])

            # For coefficient of variation
            cv_speeds.append(post_data["speed_fish"].std() / post_data["speed_fish"].mean())

    # Create the figure
    fig = plt.figure(figsize=(24, 8))
    gs = GridSpec(1, 2, width_ratios=[1, 1], wspace=0.1)

    tick_label_size = 30
    radial_label_size = 24
    axis_label_size = 20

    ######################################
    # Left subplot: Polar plot
    ax = fig.add_subplot(gs[0], polar=True)

    # Customize angular ticks to show radians instead of degrees
    radian_ticks = np.linspace(0, 2 * np.pi, 9)  # Define tick positions in radians
    tick_labels = [
        (
            r"$0$"
            if tick == 0  # Hide 2π duplicate label at 0
            else (
                r"$\frac{1}{4}\pi$"
                if tick == 0.25 * np.pi
                else (
                    r"$\frac{1}{2}\pi$"
                    if tick == 0.5 * np.pi
                    else (
                        r"$\frac{3}{4}\pi$"
                        if tick == 0.75 * np.pi
                        else (
                            r"$\pi$"
                            if tick == np.pi
                            else (
                                r"$\frac{5}{4}\pi$"
                                if tick == 1.25 * np.pi
                                else (
                                    r"$\frac{3}{2}\pi$"
                                    if tick == 1.5 * np.pi
                                    else (
                                        r"$\frac{7}{4}\pi$"
                                        if tick == 1.75 * np.pi
                                        else r"" if tick == 2 * np.pi else ""
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
        for tick in radian_ticks
    ]
    ax.set_xticks(radian_ticks)
    ax.set_xticklabels(
        tick_labels, fontsize=tick_label_size
    )  # Apply custom tick labels
    ax.tick_params(axis='x', which='major', pad=15)

    # Define bins for speed (radial) and angle (angular)
    speed_bins = 6
    angle_bins = 24
    speed_edges = np.linspace(0, max(speeds), speed_bins + 1)  # 6 edges for 5 bins
    angle_edges = np.linspace(0, 2 * np.pi, angle_bins + 1)  # 13 edges for 12 bins

    # Colormap
    # Get the existing colormap
    original_cmap = cm.get_cmap("PuBu", 256)
    # Create a new colormap with white at the bottom
    new_colors = original_cmap(np.linspace(0, 1, 256))
    new_colors[0] = np.array([1, 1, 1, 1])  # Replace the first color with white (RGBA)
    custom_cmap = ListedColormap(new_colors)

    # Histogram
    angles = np.array(angles)
    hist, _, _ = np.histogram2d(angles, speeds, bins=[angle_edges, speed_edges])
    hist = hist / hist.max()
    theta, r = np.meshgrid(angle_edges, speed_edges)
    pcm = ax.pcolormesh(theta, r, hist.T, shading="auto", cmap=custom_cmap, alpha=1.0)
    ax.set_rlabel_position(67.5)

    # Format radial ticks (fish speed) using LaTeX
    radial_ticks = np.linspace(0, max(speeds), 7)  # 6 ticks for 5 bins
    radial_labels = ["", r"5", r"10", r"15", r"20", r"25", r""]
    ax.set_yticks(radial_ticks)
    ax.set_yticklabels(radial_labels, fontsize=radial_label_size)

    # Add a rounded "Fish Speed" label at 67.5 degrees (3/8π)
    angle = 67.5  # degrees
    radius = max(speeds) + 3.0  # Slightly outside the maximum radius
    # ax.text(
    #     np.radians(angle),
    #     radius,
    #     r"$v^F_{\max}$ (cm/s)",  # Use LaTeX for the label
    #     fontsize=30,
    #     va="center",
    #     ha="center",
    #     rotation=angle - 90,  # Rotate the text tangentially
    #     rotation_mode="anchor",
    #     transform=ax.transData,
    # )
    ax.set_xlabel(r"relative angle $\theta_{RF}$ (rad)", fontsize=30)
    ax.set_ylabel(r"max. fish speed $v^F_{\max}$ (cm/s)", fontsize=30, labelpad=35)

    # Color bar for density
    cbar = fig.colorbar(pcm, ax=ax, label="Density", shrink=1.0)
    cbar.set_label(r"density", fontsize=30)
    cbar_tick_size = 24
    cbar.ax.tick_params(labelsize=cbar_tick_size)
    # cbar.ax.set_aspect(30)  # Adjust height-to-width ratio of the colorbar

    # Annotate with A.
    ax.text(
        0,
        1.1,
        r"\textbf{A.}",
        transform=ax.transAxes,
        fontsize=34,
        fontweight="bold",
        va="top",
        ha="right",
    )

    ######################################
    # --- Right: Displacement density ---
    ax = fig.add_subplot(gs[1])
    ax.hist(
        end_speeds,
        bins=20,
        color="tab:purple",
        alpha=0.5,
        label="Displacement",
        edgecolor="black",
        linewidth=1,
        density=True,
        zorder=20,
    )
    # sns.histplot(
    #     displacements,
    #     kde=True,
    #     bins=20,
    #     ax=ax_disp,
    #     color="green",
    #     alpha=0.7,
    #     label="Density"
    # )
    # ax_disp.set_title("Combined Total Displacement")
    axis_label_size = 30
    ax.set_xlabel(r"trial-end fish speed (cm/s)", fontsize=axis_label_size)
    ax.set_ylabel(r"density", fontsize=axis_label_size)
    # Adjust tick size
    tick_label_size = 24
    ax.tick_params(axis="both", which="major", labelsize=tick_label_size)
    ax.set_xlim(0, 25)
    # Approximate displacement with Gaussian (single)
    data = np.array(end_speeds).reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(data)

    means = gmm.means_.flatten()
    variances = gmm.covariances_.flatten()
    weights = gmm.weights_.flatten()

    x = np.linspace(min(data), max(data), 1000).flatten()
    y = np.zeros_like(x)
    for mean, var, weight in zip(means, variances, weights):
        component = (
            weight
            * (1 / np.sqrt(2 * np.pi * var))
            * np.exp(-((x - mean) ** 2) / (2 * var))
        )
        y += component
    ax.plot(x, y, color="purple", linewidth=2, label="GMM Total D")

    # Annotate with B.
    ax.text(
        0,
        1.1,
        r"\textbf{B.}",
        transform=ax.transAxes,
        fontsize=34,
        fontweight="bold",
        va="top",
        ha="right",
    )

    # End speeds
    # ax2 is inset of ax
    ax2 = ax.inset_axes([0.575, 0.6, 0.4, 0.35])
    ax2.hist(
        displacements,
        bins=20,
        color="tab:cyan",
        alpha=0.5,
        label="End Speed",
        edgecolor="black",
        linewidth=1,
        density=True,
        zorder=1,
    )
    # Overlay Gaussian Mixture Model (GMM) on total displacement
    data = np.array(displacements).reshape(-1, 1)
    gmm = GaussianMixture(n_components=1, random_state=42)
    gmm.fit(data)

    means = gmm.means_.flatten()
    variances = gmm.covariances_.flatten()
    weights = gmm.weights_.flatten()

    x = np.linspace(min(data), max(data), 1000).flatten()
    y = np.zeros_like(x)
    for mean, var, weight in zip(means, variances, weights):
        component = (
            weight
            * (1 / np.sqrt(2 * np.pi * var))
            * np.exp(-((x - mean) ** 2) / (2 * var))
        )
        y += component
        # ax.plot(x, component, label=f"Gaussian (mean={mean:.2f})")

    # Total GMM distribution
    ax2.plot(x, y, color="darkcyan", linewidth=2, label="GMM Total")
    ax2.set_xlabel(r"fish displacement (cm)", fontsize=30)
    ax2.tick_params(axis="both", which="major", labelsize=24)

    # Final adjustments and save
    fig.subplots_adjust(
        left=-0.075, right=0.95, top=0.9, bottom=0.125
    )  # Fine-tune borders
    output_folder = "analysis/clean/figs/sub3"
    os.makedirs(output_folder, exist_ok=True)
    filename = "post_avoidance_dynamics.pdf"
    fig.savefig(os.path.join(output_folder, filename), bbox_inches="tight")
    plt.close()
    print(f"Figure saved at {output_folder}/{filename}")

    print(np.mean(displacements))
    print(np.std(displacements))


def statistics(exp_trials):
    pre_max_speeds = []
    post_max_speeds = []

    lower_bound = 35
    upper_bound = 40

    subjects = []
    conditions = []
    speeds = []

    for trials in exp_trials:
        for trial in trials:
            trial_data = trial["data"]

            pre_max_speed = np.nan
            post_max_speed = np.nan

            # Process pre
            pre_data = trial_data[
                (trial_data["distance"] >= lower_bound)
                & (trial_data["distance"] < upper_bound)
            ]
            if not pre_data.empty:
                pre_max_speed = pre_data["speed_fish"].max()

            # Process post
            end_idx = trial["end_approach_index"]
            post_data = trial_data.iloc[end_idx + 1: end_idx + 10]
            if not post_data.empty:
                post_max_speed = post_data["speed_fish"].max()

            # Append to lists
            if not np.isnan(pre_max_speed) and not np.isnan(post_max_speed):
                subjects.append(trial["experiment_id"])
                conditions.append("pre")
                speeds.append(pre_max_speed)
                subjects.append(trial["experiment_id"])
                conditions.append("post")
                speeds.append(post_max_speed)

    df = pd.DataFrame({"subject": subjects, "condition": conditions, "speed": speeds})
    result = run_plm_model(
        df, 
        "speed", 
        ["condition"], 
        ["subject"], 
        "within", 
        "individual",
    )
    print(result)


def plot_ind_end_speed(exp_trials):
    # Prepare data for both subplots
    end_speeds = []
    df = []

    # Extract data for each trial
    for exp_id, trials in enumerate(exp_trials):
        exp_end_speeds = []
        for trial in trials:
            trial_data = trial["data"]
            end_idx = trial["end_approach_index"]
            # Extract post-approach data (10 frames after end approach index)
            post_data = trial_data.iloc[end_idx + 1 : end_idx + 10]
            # Extract post-phase data for both plots
            post_data = trial_data.iloc[end_idx + 1 : end_idx + 1 + 20]
            exp_end_speeds.append(post_data["speed_fish"].iloc[-1])

            df.append(
                {
                    "subject": exp_id + 1,
                    "speed": post_data["speed_fish"].iloc[-1],
                },
            )   
        end_speeds.append(exp_end_speeds)

    fig, axes = plt.subplots(5, 1, figsize=(10, 20))
    for i, ax in enumerate(axes):
        exp_end_speeds = end_speeds[i]
        ax.hist(
            exp_end_speeds,
            bins=10,
            color="tab:purple",
            alpha=0.5,
            label="Displacement",
            edgecolor="black",
            linewidth=1,
            density=True,
            zorder=20,
        )
        data = np.array(exp_end_speeds).reshape(-1, 1)
        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(data)

        means = gmm.means_.flatten()
        variances = gmm.covariances_.flatten()
        weights = gmm.weights_.flatten()

        x = np.linspace(min(data), max(data), 1000).flatten()
        y = np.zeros_like(x)
        for mean, var, weight in zip(means, variances, weights):
            component = (
                weight
                * (1 / np.sqrt(2 * np.pi * var))
                * np.exp(-((x - mean) ** 2) / (2 * var))
            )
            y += component
        axis_label_size = 18
        ax.plot(x, y, color="purple", linewidth=2, label="GMM Total D")
        ax.set_ylabel(r"density", fontsize=axis_label_size)
        ax.set_xlim(0, 25)
        ax.set_ylim(0, 0.4)
        
    ax = axes[-1]
    ax.set_xlabel(r"trial-end fish speed (cm/s)", fontsize=axis_label_size)
    tick_label_size = 24
    ax.tick_params(axis="both", which="major", labelsize=tick_label_size)

    # Final adjustments and save
    fig.subplots_adjust(
        left=-0.075, right=0.95, top=0.9, bottom=0.125
    )  # Fine-tune borders
    output_folder = "analysis/clean/figs/sub3"
    os.makedirs(output_folder, exist_ok=True)
    filename = "individual_end_speed.pdf"
    fig.savefig(os.path.join(output_folder, filename), bbox_inches="tight")
    plt.close()
    print(f"Figure saved at {output_folder}/{filename}")

    # print the mean and variance for each subject
    for i, exp_end_speeds in enumerate(end_speeds):
        print(f"Subject {i+1}:")
        print(np.mean(exp_end_speeds))
        print(np.std(exp_end_speeds))

    df = pd.DataFrame(df)
    summary = run_lmer_model(df, "speed", ["1"], ["subject"])
    print(summary)


if __name__ == "__main__":
    exp_trials = extract_trials()
    # plot_combined(exp_trials)
    # statistics(exp_trials)
    # plot_ind_end_speed(exp_trials)

    # output the total number of frames (combined) from all the trials in exp_trials
    total_frames = 0
    for trials in exp_trials:
        for trial in trials:
            trial_data = trial["data"]
            total_frames += trial_data.shape[0]
    print(total_frames)