import os
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde

from core.experiment import Experiment
from core.study import study

from analysis.clean.reactions import extract_trials, get_baseline_data_subjects
from analysis.clean.sub4.help import save_plot

import utils.plotter

from scipy.stats import binomtest


def measure_distance_based(exp_trials, distance_bins):
    # Initialize data structures to store lists of lists of lists for each experiment
    n_experiments = len(exp_trials)
    speed_data = [[] for _ in range(n_experiments)]
    angle_data = [[] for _ in range(n_experiments)]
    turn_data = [[] for _ in range(n_experiments)]
    accel_data = [[] for _ in range(n_experiments)]

    # Iterate over each experiment and each trial within the experiment
    for exp_idx, trials in enumerate(exp_trials):
        # Initialize lists of lists for each distance bin within this experiment
        exp_speed_bins = [[] for _ in range(len(distance_bins))]
        exp_angle_bins = [[] for _ in range(len(distance_bins))]
        exp_turn_bins = [[] for _ in range(len(distance_bins))]
        exp_accel_bins = [[] for _ in range(len(distance_bins))]

        for trial in trials:
            # Get data up to end of approach index
            end_idx = trial["end_approach_index"]
            trial_data = trial["data"].iloc[:end_idx]

            # Bin the data based on the distance bins
            for bin_idx in range(len(distance_bins) - 1):
                lower_bound = distance_bins[bin_idx + 1]
                upper_bound = distance_bins[bin_idx]

                # Filter rows within the current distance bin
                bin_data = trial_data[
                    (trial_data["distance"] >= lower_bound)
                    & (trial_data["distance"] < upper_bound)
                ]

                # Append speed, turn, and acceleration values to respective bin lists
                exp_speed_bins[bin_idx].append(bin_data["speed_fish"].tolist())
                exp_angle_bins[bin_idx].append(bin_data["angle_rf"].tolist())
                exp_turn_bins[bin_idx].append(bin_data["turn_fish"].tolist())
                exp_accel_bins[bin_idx].append(bin_data["acceleration_fish"].tolist())

            # Additional bin for 10 frames after the end of approach
            post_approach_data = trial["data"].iloc[end_idx : end_idx + 10]
            if not post_approach_data.empty:
                exp_speed_bins[-1].append(post_approach_data["speed_fish"].tolist())
                exp_angle_bins[-1].append(post_approach_data["angle_rf"].tolist())
                exp_turn_bins[-1].append(post_approach_data["turn_fish"].tolist())
                exp_accel_bins[-1].append(
                    post_approach_data["acceleration_fish"].tolist()
                )

        # After processing all trials in the experiment, add lists to main data structures
        speed_data[exp_idx] = exp_speed_bins
        angle_data[exp_idx] = exp_angle_bins
        turn_data[exp_idx] = exp_turn_bins
        accel_data[exp_idx] = exp_accel_bins

    return speed_data, angle_data, turn_data, accel_data


def plot_frequencies(
    df_baseline, speed_data, angle_data, acceleration_data, turn_data, distance_bins
):
    # Define bin labels for x-axis (e.g., 40-35, 35-30, ..., 5-0, Post)
    bin_labels = [
        f"{distance_bins[i]}-{distance_bins[i+1]}"
        for i in range(len(distance_bins) - 1)
    ]
    bin_labels.append("Post")  # Label for the last "Post" bin

    combined_freq_turn = []
    combined_freq_acc = []

    base_freq_turn = []

    # Iterate over each experiment's data
    for exp_idx, (exp_speed_data, exp_accel_data, exp_turn_data) in enumerate(
        zip(speed_data, acceleration_data, turn_data)
    ):
        # Compute thresholds from df_baseline
        df_exp_bas = df_baseline[df_baseline["experiment_id"] == exp_idx]
        df_exp_bas["abs_turn_fish"] = df_exp_bas["turn_fish"].abs()
        max_acc_mean = df_exp_bas["acceleration_fish"].mean()
        max_acc_std = df_exp_bas["acceleration_fish"].std()
        max_acc_95th = df_exp_bas["acceleration_fish"].quantile(0.95)

        max_turn_mean = df_exp_bas["abs_turn_fish"].mean()
        max_turn_std = df_exp_bas["abs_turn_fish"].std()
        max_turn_95th = df_exp_bas["abs_turn_fish"].quantile(0.975)
        max_turn_95th = max_turn_mean + 2 * max_turn_std
        base_turn_freq = df_exp_bas["abs_turn_fish"].gt(max_turn_95th).mean()
        base_freq_turn.append(base_turn_freq)

        speed_95th = df_exp_bas["speed_fish"].quantile(0.95)

        # Initialize lists to hold values for each statistic across bins for this experiment
        means_per_bin = []
        stds_per_bin = []
        cvs_per_bin = []
        max_accel_per_bin = []
        max_abs_turn_per_bin = []

        # Iterate over each bin in the current experiment
        for bin_idx in range(len(distance_bins)):
            # Initialize lists for speed statistics
            bin_means = []
            bin_stds = []
            bin_cvs = []

            # Calculate speed statistics (mean, std, CV) for each trial in this bin
            if bin_idx < len(exp_speed_data):  # Check if bin exists in experiment data
                bin_trials = exp_speed_data[bin_idx]

                for trial_values in bin_trials:
                    if len(trial_values) > 0:
                        trial_mean = np.mean(trial_values)
                        trial_std = np.std(trial_values)
                        trial_cv = trial_std / trial_mean if trial_mean != 0 else np.nan

                        bin_means.append(trial_mean)
                        bin_stds.append(trial_std)
                        bin_cvs.append(trial_cv)

            # Append speed statistics to overall lists for this experiment
            means_per_bin.append(bin_means)
            stds_per_bin.append(bin_stds)
            cvs_per_bin.append(bin_cvs)

            # Calculate max acceleration for each trial in this bin
            if bin_idx < len(exp_accel_data):
                max_accel_per_bin.append(
                    [
                        np.max(trial)
                        for trial in exp_accel_data[bin_idx]
                        if len(trial) > 0
                    ]
                )

            # Calculate max absolute turn for each trial in this bin
            if bin_idx < len(exp_turn_data):
                max_abs_turn_per_bin.append(
                    [
                        np.max(np.abs(trial))
                        for trial in exp_turn_data[bin_idx]
                        if len(trial) > 0
                    ]
                )

        # Calculate frequency of prominent turns and accelerations for each bin
        prominent_speed_freq = []
        prominent_turn_freq = []
        prominent_accel_freq = []

        max_trials = max([trial_count for trial_count in map(len, exp_turn_data)])

        for bin_idx in range(len(distance_bins)):
            total_trials = len(exp_turn_data[bin_idx])
            prominent_speeds = sum(
                any(speed > speed_95th for speed in trial)
                for trial in exp_speed_data[bin_idx]
            )
            prominent_turns = sum(
                any(abs(turn) > max_turn_95th for turn in trial)
                for trial in exp_turn_data[bin_idx]
            )
            prominent_accels = sum(
                any(accel > max_acc_95th for accel in trial)
                for trial in exp_accel_data[bin_idx]
            )

            prominent_speed_freq.append(
                prominent_speeds / max_trials if total_trials > 0 else 0
            )
            prominent_turn_freq.append(
                prominent_turns / max_trials if total_trials > 0 else 0
            )
            prominent_accel_freq.append(
                prominent_accels / max_trials if total_trials > 0 else 0
            )

        combined_freq_turn.append(prominent_turn_freq)
        combined_freq_acc.append(prominent_accel_freq)

    ####### FIGURE ########
    # Plot frequencies
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 2, width_ratios=[1.1, 0.8], wspace=0.2, hspace=0.3)

    axes = []

    fontsize_label = 13.5
    fontsize_ticks = 11.5

    def to_percent(y, _):
        return f"{y * 100:.0f}%"  # Multiply by 100 and format as a percentage

    # Plot turn frequencies
    ax = fig.add_subplot(gs[0, 0])
    axes.append(ax)
    combined_freq_turn = np.array(combined_freq_turn)
    mean_freq_turn = np.mean(combined_freq_turn, axis=0)
    std_freq_turn = np.std(combined_freq_turn, axis=0)
    ax.bar(
        range(len(bin_labels)),
        mean_freq_turn,
        yerr=std_freq_turn,
        color="tab:gray",
        # capsize=5,
    )
    ax.axhline(
        np.mean(base_freq_turn),
        color="darkred",
        linestyle="-",
        linewidth=2,
    )
    ax.set_xticks(range(len(bin_labels)))
    # set empty xticklabels
    ax.set_xticklabels([])
    ax.set_ylabel(r"frequency of extreme turns (\%)", fontsize=fontsize_label)
    ax.yaxis.set_major_formatter(FuncFormatter(to_percent))

    ax.tick_params(axis="both", which="major", labelsize=fontsize_ticks)

    # Plot before-after relative angle
    ax = fig.add_subplot(gs[0, 1], polar=True)
    axes.append(ax)
    num_bins = 30
    bin_width = (2 * np.pi) / num_bins  # Width of each bin

    pre_angles = [
        value for experiment in angle_data for trial in experiment[0] for value in trial
    ]
    pre_counts, pre_edges = np.histogram(
        pre_angles, bins=num_bins, range=(0, 2 * np.pi)
    )
    pre_densities = pre_counts / (len(pre_angles) * bin_width)

    post_angles = [
        value
        for experiment in angle_data
        for trial in experiment[-1]
        for value in trial
    ]
    post_counts, post_edges = np.histogram(
        post_angles, bins=num_bins, range=(0, 2 * np.pi)
    )
    post_densities = post_counts / (len(post_angles) * bin_width)

    # Binomial test to see if significant bias to right hemisphere

    # modulo 2pi to get the angle in the range [0, 2pi]
    post_angles = np.array(post_angles) % (2 * np.pi)
    post_right = np.sum(np.array(post_angles) < np.pi)
    post_total = post_angles.shape[0]
    post_p = binomtest(post_right, post_total, 0.5)

    print(post_p)

    ax.bar(
        pre_edges[:-1],
        pre_densities,
        width=np.diff(pre_edges),
        edgecolor="black",
        alpha=0.5,
        label="pre (40-35)",
        color="tab:blue",
        zorder=3,
    )
    ax.bar(
        post_edges[:-1],
        post_densities,
        width=np.diff(post_edges),
        edgecolor="black",
        alpha=0.5,
        label="post",
        color="purple",
        zorder=2,
    )

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
    ax.tick_params(axis="both", which="major", labelsize=fontsize_ticks)
    ax.set_xticks(radian_ticks)
    ax.set_xticklabels(tick_labels, fontsize=14)  # Apply custom tick labels
    ax.tick_params(axis="x", which="major", pad=5)

    ax.set_xlabel(r"relative angle $\theta_{RF}$ (rad)", fontsize=fontsize_label)
    ax.set_ylabel(r"density", fontsize=fontsize_label, labelpad=25)

    ax.legend(
        bbox_to_anchor=(0.025, -0.06), #(0.01, -0.075),
        loc="center",
        fancybox=True,
        edgecolor="none",
        fontsize=12,
        facecolor="none",
    )

    # Plot acc
    ax = fig.add_subplot(gs[1, 0])
    axes.append(ax)
    combined_freq_acc = np.array(combined_freq_acc)
    mean_freq_acc = np.mean(combined_freq_acc, axis=0)
    std_freq_acc = np.std(combined_freq_acc, axis=0)
    ax.bar(range(len(bin_labels)), mean_freq_acc, yerr=std_freq_acc, color="tab:gray")
    ax.axhline(
        0.05,
        color="darkred",
        linestyle="-",
        linewidth=2,
    )
    ax.set_xticks(range(len(bin_labels)))
    ax.set_xticklabels(bin_labels, rotation=45)
    ax.set_ylabel(r"frequency of extreme accelerations (\%)", fontsize=fontsize_label)
    ax.yaxis.set_major_formatter(FuncFormatter(to_percent))
    ax.set_xlabel(
        "pre-encounter distances (cm) and initial post-encounter phase (0.4s)",
        fontsize=fontsize_label,
    )

    ax.tick_params(axis="both", which="major", labelsize=fontsize_ticks)

    # Plot before-after speed
    ax = fig.add_subplot(gs[1, 1])
    axes.append(ax)
    pre_speeds = [
        value for experiment in speed_data for trial in experiment[0] for value in trial
    ]
    post_speeds = [
        value
        for experiment in speed_data
        for trial in experiment[-1]
        for value in trial
    ]
    ax.hist(pre_speeds, bins=20, density=True, alpha=0.5, color="tab:blue", label="pre")
    ax.hist(post_speeds, bins=20, density=True, alpha=0.5, color="purple", label="post")
    # pre_kde = gaussian_kde(pre_speeds)
    # post_kde = gaussian_kde(post_speeds)
    # x = np.linspace(0, 100, 1000)
    # ax.plot(x, pre_kde(x), label="Pre", color="blue")
    # ax.plot(x, post_kde(x), label="Post", color="orange")
    ax.set_xlabel("fish speed $v^F$ (cm/s)", fontsize=fontsize_label)
    ax.set_ylabel("density", fontsize=fontsize_label)
    ax.set_xlim(0, 25)
    ax.tick_params(axis="both", which="major", labelsize=fontsize_ticks)
    # ax.legend(loc='upper right', fancybox=True, edgecolor='white', fontsize=12)

    # Annotate with A, B, C, D
    positions = [(-0.025, 1.1), (-0.225, 1.1), 
                 (-0.025, 1.075), (-0.05, 1.075)]
    for i, ax in enumerate(axes):
        ax.text(
            *positions[i],
            r"\textbf{" + f"{chr(65 + i)}." + r"}",
            fontsize=18,
            weight="bold",
            transform=ax.transAxes,
            verticalalignment="center",
            horizontalalignment="center",
        )

    output_folder = "analysis/clean/figs/sub4"
    os.makedirs(output_folder, exist_ok=True)
    filename = "frequencies.pdf"
    fig.savefig(os.path.join(output_folder, filename), bbox_inches="tight")
    plt.close()
    print(f"Figure saved at {output_folder}/{filename}")


if __name__ == "__main__":
    exp_trials = extract_trials()
    distance_bins = list(range(40, -5, -5))  # Generates [40, 35, 30, ..., 5, 0]

    df_baseline = get_baseline_data_subjects()

    speed_data, angle_data, turn_data, accel_data = measure_distance_based(
        exp_trials, distance_bins
    )
    plot_frequencies(
        df_baseline, speed_data, angle_data, accel_data, turn_data, distance_bins
    )
