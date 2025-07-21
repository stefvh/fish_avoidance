from matplotlib import pyplot as plt
import numpy as np

from core.experiment import Experiment
from core.study import study

from analysis.clean.reactions import extract_trials
from analysis.clean.sub3.help import save_plot

from analysis.clean.avoidance_mechanisms import get_data

import seaborn as sns


def plot_average_angle_per_bin(exp_trials, distance_bins):
    # Create labels for each bin including "Post" for the post-approach bin
    bin_labels = [
        f"{distance_bins[i]}-{distance_bins[i+1]}"
        for i in range(len(distance_bins) - 1)
    ]
    bin_labels.append("Post")

    # Create a figure with subplots for each bin
    num_bins = len(distance_bins)
    fig, axs = plt.subplots(num_bins, 1, figsize=(10, 4 * num_bins), sharex=True)

    for bin_idx in range(len(distance_bins) - 1):
        # Lists to store angles for the current bin across all trials
        bin_angles = []

        lower_bound = distance_bins[bin_idx + 1]
        upper_bound = distance_bins[bin_idx]

        # Collect angle data for each trial within the current distance bin
        for trials in exp_trials:
            for trial in trials:
                trial_data = trial["data"]
                bin_data = trial_data[
                    (trial_data["distance"] >= lower_bound)
                    & (trial_data["distance"] < upper_bound)
                ]
                if not bin_data.empty:
                    angles = bin_data["angle_rf"]
                    avg_angle = np.arctan2(
                        np.mean(np.sin(angles)), np.mean(np.cos(angles))
                    )
                    avg_angle = np.mod(avg_angle, 2 * np.pi)
                else:
                    avg_angle = np.nan
                if not np.isnan(avg_angle):
                    bin_angles.append(avg_angle)

        # Density and scatter plot for the current distance bin
        sns.kdeplot(bin_angles, ax=axs[bin_idx], fill=True, color="blue", alpha=0.5)
        sns.scatterplot(
            x=bin_angles,
            y=[0] * len(bin_angles),
            color="blue",
            alpha=0.3,
            ax=axs[bin_idx],
        )
        axs[bin_idx].set_ylabel("Density")
        axs[bin_idx].set_title(
            f"Density of Average Angle RF for Distance Bin ({lower_bound}-{upper_bound})"
        )

    # Additional density plot for the post-approach bin
    post_approach_angles = []
    for trials in exp_trials:
        for trial in trials:
            trial_data = trial["data"]
            end_idx = trial["end_approach_index"]
            post_approach_data = trial_data.iloc[end_idx : end_idx + 10]
            if not post_approach_data.empty:
                angles = post_approach_data["angle_rf"]
                avg_post_angle = np.arctan2(
                    np.mean(np.sin(angles)), np.mean(np.cos(angles))
                )
                avg_post_angle = np.mod(avg_post_angle, 2 * np.pi)
            else:
                avg_post_angle = np.nan
            if not np.isnan(avg_post_angle):
                post_approach_angles.append(avg_post_angle)

    # Density and scatter plot for post-approach data
    sns.kdeplot(post_approach_angles, ax=axs[-1], fill=True, color="blue", alpha=0.5)
    sns.scatterplot(
        x=post_approach_angles,
        y=[0] * len(post_approach_angles),
        color="blue",
        alpha=0.3,
        ax=axs[-1],
    )
    axs[-1].set_ylabel("Density")
    axs[-1].set_title("Density of Average Angle RF for Post-approach")

    for ax in axs:
        ax.set_xlim(0, 2 * np.pi)
        ax.axvline(
            x=0.5 * np.pi, color="black", linestyle="--"
        )  # Pi/2 line for reference
        ax.axvline(x=np.pi, color="black", linestyle="-")  # Pi line for reference
        ax.axvline(
            x=1.5 * np.pi, color="black", linestyle="--"
        )  # 3Pi/2 line for reference

    # Common x-axis label
    axs[-1].set_xlabel("Average Angle RF")
    plt.tight_layout()
    save_plot(fig, "average_angle_per_bin.pdf")


def plot_angle_at_max_speed(exp_trials, distance_bins):
    # Define bin labels for the plot titles
    bin_labels = [
        f"{distance_bins[i]}-{distance_bins[i+1]}"
        for i in range(len(distance_bins) - 1)
    ]
    bin_labels.append("Post")

    # Create a figure with one row per distance bin
    num_bins = len(distance_bins)
    fig, axs = plt.subplots(num_bins, 1, figsize=(10, 4 * num_bins), sharex=True)

    for bin_idx in range(len(distance_bins) - 1):
        # List to store angles for the current bin at max speed
        bin_angles = []

        lower_bound = distance_bins[bin_idx + 1]
        upper_bound = distance_bins[bin_idx]

        # Collect angle data at max speed for each trial within the current distance bin
        for trials in exp_trials:
            for trial in trials:
                trial_data = trial["data"]
                bin_data = trial_data[
                    (trial_data["distance"] >= lower_bound)
                    & (trial_data["distance"] < upper_bound)
                ]
                if not bin_data.empty:
                    max_speed_idx = bin_data["speed_fish"].idxmax()
                    max_angle = bin_data.loc[max_speed_idx, "angle_rf"]
                    bin_angles.append(max_angle)

        # Density and scatter plot for the current bin
        sns.kdeplot(bin_angles, ax=axs[bin_idx], fill=True, color="red", alpha=0.5)
        sns.scatterplot(
            x=bin_angles,
            y=[0] * len(bin_angles),
            color="red",
            alpha=0.3,
            ax=axs[bin_idx],
        )
        axs[bin_idx].set_ylabel("Density")
        axs[bin_idx].set_title(
            f"Density of Angle RF at Max Speed for Distance Bin ({lower_bound}-{upper_bound})"
        )

    # Additional density plot for post-approach data
    post_approach_angles = []
    for trials in exp_trials:
        for trial in trials:
            trial_data = trial["data"]
            end_idx = trial["end_approach_index"]
            post_approach_data = trial_data.iloc[end_idx : end_idx + 10]
            if not post_approach_data.empty:
                max_speed_idx = post_approach_data["speed_fish"].idxmax()
                max_angle = post_approach_data.loc[max_speed_idx, "angle_rf"]
                post_approach_angles.append(max_angle)

    # Density and scatter plot for post-approach data
    sns.kdeplot(post_approach_angles, ax=axs[-1], fill=True, color="red", alpha=0.5)
    sns.scatterplot(
        x=post_approach_angles,
        y=[0] * len(post_approach_angles),
        color="red",
        alpha=0.3,
        ax=axs[-1],
    )
    axs[-1].set_ylabel("Density")
    axs[-1].set_title("Density of Angle RF at Max Speed for Post-approach")

    # Common x-axis label
    axs[-1].set_xlabel("Angle RF at Max Speed")
    plt.tight_layout()
    save_plot(fig, "angle_at_max_speed.pdf")


def plot_density_v_avoid_speed(exp_trials):
    fig, ax = plt.subplots(figsize=(10, 6))
    selected_angles = []

    for trials in exp_trials:
        for trial in trials:
            trial_data = trial["data"]
            filtered_data = trial_data[
                (trial_data["v_avoid"] > 0) & (trial_data["speed_fish"] > 2)
            ]
            selected_angles.extend(filtered_data["angle_rf"].tolist())

    # Density plot of angle_rf
    sns.kdeplot(selected_angles, ax=ax, color="purple", fill=True, alpha=0.7)
    sns.scatterplot(
        x=selected_angles,
        y=[0] * len(selected_angles),
        color="purple",
        alpha=0.3,
        ax=ax,
    )
    ax.set_xlabel("Angle RF")
    ax.set_ylabel("Density")
    ax.set_title("Density of Angle RF for (v_avoid > 0 & speed_fish > 2)")

    ax.axvline(x=0.5 * np.pi, color="black", linestyle="--")  # Pi/2 line for reference
    ax.axvline(x=np.pi, color="black", linestyle="-")  # Pi line for reference
    ax.axvline(x=1.5 * np.pi, color="black", linestyle="--")  # 3Pi/2 line for reference

    ax.set_xlim(0, 2 * np.pi)

    plt.tight_layout()
    save_plot(fig, "density_v_avoid_speed.pdf")


def plot_circular_density(exp_trials):
    # Collect average angles for the 35-40 bin and post-approach bin
    bin_35_40_angles = []
    post_approach_angles = []

    # Define the 35-40 bin bounds
    lower_bound = 35.0
    upper_bound = 40.0

    for trials in exp_trials:
        for trial in trials:
            trial_data = trial["data"]

            # Process 35-40 bin
            bin_data = trial_data[
                (trial_data["distance"] >= lower_bound)
                & (trial_data["distance"] < upper_bound)
            ]
            if not bin_data.empty:
                angles = bin_data["angle_rf"]
                avg_angle = np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))
                avg_angle = np.mod(avg_angle, 2 * np.pi)
                if not np.isnan(avg_angle):
                    bin_35_40_angles.append(avg_angle)

            # Process post-approach bin
            end_idx = trial["end_approach_index"]
            post_approach_data = trial_data.iloc[end_idx : end_idx + 10]
            if not post_approach_data.empty:
                angles = post_approach_data["angle_rf"]
                avg_angle = np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))
                avg_angle = np.mod(avg_angle, 2 * np.pi)
                if not np.isnan(avg_angle):
                    post_approach_angles.append(avg_angle)

    # Create the circular density plot
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(8, 6))

    # Plot KDE for the 35-40 bin
    sns.kdeplot(
        bin_35_40_angles,
        ax=ax,
        fill=True,
        alpha=0.5,
        label="35-40 Bin",
        color="blue",
        linewidth=1.5,
    )

    # Plot KDE for the post-approach bin
    sns.kdeplot(
        post_approach_angles,
        ax=ax,
        fill=True,
        alpha=0.5,
        label="Post-approach",
        color="orange",
        linewidth=1.5,
    )

    # Add labels and legend
    ax.set_title(
        "Circular Density of Average Angles (35-40 Bin vs Post-approach)", va="bottom"
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))

    # # Add radial lines for reference at pi/2, pi, and 3pi/2
    # ax.axvline(np.pi / 2, color="black", linestyle="--", alpha=0.7, label="Pi/2")
    # ax.axvline(np.pi, color="black", linestyle="-", alpha=0.7, label="Pi")
    # ax.axvline(1.5 * np.pi, color="black", linestyle="--", alpha=0.7, label="3Pi/2")

    plt.tight_layout()
    save_plot(fig, "circular_density.pdf")


def plot_circular_histogram(exp_trials):
    # Collect angles for the 35-40 bin and post-approach bin
    bin_35_40_angles = []
    post_approach_angles = []

    # Define the 35-40 bin bounds
    lower_bound = 35
    upper_bound = 40

    for trials in exp_trials:
        for trial in trials:
            trial_data = trial["data"]

            # Process 35-40 bin
            bin_data = trial_data[
                (trial_data["distance"] >= lower_bound)
                & (trial_data["distance"] < upper_bound)
            ]
            if not bin_data.empty:
                angles = bin_data["angle_rf"]
                bin_35_40_angles.extend(angles)

            # Process post-approach bin
            end_idx = trial["end_approach_index"]
            post_approach_data = trial_data.iloc[end_idx : end_idx + 10]
            if not post_approach_data.empty:
                angles = post_approach_data["angle_rf"]
                post_approach_angles.extend(angles)

    # Convert angle lists to numpy arrays
    bin_35_40_angles = np.array(bin_35_40_angles)
    post_approach_angles = np.array(post_approach_angles)

    # Calculate normalized densities
    num_bins = 30
    bin_width = (2 * np.pi) / num_bins  # Width of each bin

    counts_35_40, edges_35_40 = np.histogram(
        bin_35_40_angles, bins=num_bins, range=(0, 2 * np.pi)
    )
    densities_35_40 = counts_35_40 / (len(bin_35_40_angles) * bin_width)

    counts_post, edges_post = np.histogram(
        post_approach_angles, bins=num_bins, range=(0, 2 * np.pi)
    )
    densities_post = counts_post / (len(post_approach_angles) * bin_width)

    # Create the polar plot
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(8, 6))

    # Plot density histogram for the 35-40 bin
    ax.bar(
        edges_35_40[:-1],
        densities_35_40,
        width=np.diff(edges_35_40),
        edgecolor="black",
        alpha=0.5,
        label="35-40 Bin",
        color="blue",
        zorder=3,
    )

    # Plot density histogram for the post-approach bin
    ax.bar(
        edges_post[:-1],
        densities_post,
        width=np.diff(edges_post),
        edgecolor="black",
        alpha=0.5,
        label="Post-approach",
        color="orange",
        zorder=2,
    )

    # Add legend and title
    ax.set_title(
        "Circular Histogram of Angles (35-40 Bin vs Post-approach)", va="bottom"
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))

    plt.tight_layout()
    save_plot(fig, "circular_histogram.pdf")


def plot_angle_all_data():
    import utils.plotter

    df = get_data()
    df = df[(df["speed_fish"] > 2)]

    # plot circular density of angle_rf
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(6, 6))

    num_bins = 30
    bin_width = (2 * np.pi) / num_bins  # Width of each bin

    angles = df["angle_rf"].to_numpy()
    counts, edges = np.histogram(
        angles, bins=num_bins, range=(0, 2 * np.pi)
    )
    densities = counts / (angles.shape[0] * bin_width)

    ax.bar(
        edges[:-1],
        densities,
        width=np.diff(edges),
        edgecolor="black",
        alpha=1.0,
        color="tab:blue",
    )

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
    ax.tick_params(axis="both", which="major", labelsize=18)
    ax.set_xticks(radian_ticks)
    ax.set_xticklabels(
        tick_labels, fontsize=21
    )  # Apply custom tick labels
    ax.tick_params(axis='x', which='major', pad=10)

    ax.set_xlabel(r"relative angle $\theta_{RF}$", fontsize=24, labelpad=5)
    ax.set_ylabel(r"density", fontsize=24, labelpad=30)
    

    plt.tight_layout()
    save_plot(fig, "circular_histogram_alldata.pdf")



if __name__ == "__main__":
    exp_trials = extract_trials()
    distance_bins = list(range(40, -5, -5))  # Generates [40, 35, 30, ..., 5, 0]

    # plot_average_angle_per_bin(exp_trials, distance_bins)
    # plot_angle_at_max_speed(exp_trials, distance_bins)
    # plot_density_v_avoid_speed(exp_trials)
    # plot_circular_density(exp_trials)
    # plot_circular_histogram(exp_trials)
    plot_angle_all_data()
