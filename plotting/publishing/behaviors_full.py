import os
import pickle
from matplotlib import pyplot as plt
import numpy as np

from analysis.clean.freezing_conditions import compute_related_freezing
from core.experiment import Experiment
from core.study import study

import utils.plotter as plotter
from utils.maths.statistical import pearson_confidence_interval

from numpy import polyfit

from scipy.stats import pearsonr, spearmanr


def get_behaviors():
    data = study.data

    behaviors = {
        "slow": [],
        "exploration": [],
        "avoidance": [],
        "wall_hugging": [],
        "total_time": [],
        "freezing_spd": [],
        "freezing_dist": [],
    }

    experiment_ids = data["experiment_id"].unique()

    data = compute_related_freezing(data)

    for experiment_id in experiment_ids:
        df = data[data["experiment_id"] == experiment_id]

        slow = df[df["speed_fish"] < 2].shape[0]
        exploration = df[(df["speed_fish"] > 2) & (df["rev"] < 0)].shape[0]
        avoidance = df[(df["speed_fish"] > 2) & (df["rev"] >= 0)].shape[0]

        wall_hugging = df[df["min_wall_distance_fish"] < 10].shape[0]

        behaviors["slow"].append(slow)
        behaviors["exploration"].append(exploration)
        behaviors["avoidance"].append(avoidance)
        behaviors["wall_hugging"].append(wall_hugging)
        behaviors["total_time"].append(df.shape[0])

        freezing_spd = df[df["freezing_spd"] > 0].shape[0]
        freezing_dist = df[df["freezing_dist"] > 0].shape[0]
        behaviors["freezing_spd"].append(freezing_spd)
        behaviors["freezing_dist"].append(freezing_dist)

    return behaviors


# Function to calculate average percentage and standard deviation
def calc_avg_std_percentages(behavior_times, total_times):
    percentages = [bt / tt for bt, tt in zip(behavior_times, total_times)]
    avg_percentage = np.mean(percentages) * 100
    std_dev = np.std(percentages) * 100
    return avg_percentage, std_dev


def plot_lin_corr_inset(ax, data):
    # Create the inset axes inside your main figure
    # inset_ax = inset_axes(ax, width="40%", height="60%", loc='upper right')  # Adjust the size and location as needed
    inset_ax = ax.inset_axes([0.65, 0.4, 0.325, 0.525])

    # Data from behaviors for the inset plot
    wall_hugging = np.array(data["wall_hugging"])
    avoidance = np.array(data["avoidance"])

    # In percentages
    wall_hugging = wall_hugging / np.array(data["total_time"]) * 100
    avoidance = avoidance / np.array(data["total_time"]) * 100

    # Scatter plot and linear fit inside the inset
    inset_ax.scatter(wall_hugging, avoidance, color="tab:blue", s=10, marker="x")

    # Perform a linear fit
    slope, intercept = polyfit(wall_hugging, avoidance, 1)
    inset_ax.plot(wall_hugging, slope * wall_hugging + intercept, color="black", lw=1)

    # Labels and title for the inset
    inset_ax.set_xlabel(r"Time of thigmotaxis (\%)", fontsize=12)
    inset_ax.set_ylabel(r"Time of avoidance (\%)", fontsize=12)

    # Optional: adjust the ticks and labels for clarity
    inset_ax.tick_params(axis="both", which="major", labelsize=10)

    # Pearson's correlation
    pearson_corr, pearson_p = pearsonr(avoidance, wall_hugging)

    pearson_conf = pearson_confidence_interval(pearson_corr, len(avoidance))

    print(f"Pearson correlation: {pearson_corr:.2f}, p-value: {pearson_p:.2f}")
    print(f"Pearson confidence interval: {pearson_conf}")

    # Text box with correlation coefficient and p-value
    textstr = (
        "Pearson correlation:\n"
        + rf"$r$ = {pearson_corr:.2f}"
        + "\n"
        + rf"$p$ = {pearson_p:.2f}"
    )
    #  "\n"+ rf"$95\%$ CI = [" + f"{pearson_conf[0]:.2f}, {pearson_conf[1]:.2f}]" + \
    props = dict(boxstyle="square", facecolor="wheat", alpha=0.0, edgecolor="black")
    inset_ax.text(
        0.05,
        0.8,
        textstr,
        transform=inset_ax.transAxes,
        fontsize=12,
        verticalalignment="center",
        bbox=props,
    )

    return ax, inset_ax


def plot_avg_std(ax, data, dict_colors):
    # Calculate average percentages and standard deviations
    slow_avg, slow_std = calc_avg_std_percentages(data["slow"], data["total_time"])
    exploration_avg, exploration_std = calc_avg_std_percentages(
        data["exploration"], data["total_time"]
    )
    avoidance_avg, avoidance_std = calc_avg_std_percentages(
        data["avoidance"], data["total_time"]
    )
    wall_hugging_avg, wall_hugging_std = calc_avg_std_percentages(
        data["wall_hugging"], data["total_time"]
    )

    # Plotting the bar chart
    labels = ["Attraction", "Slow motion", "Avoidance", "Thigmotaxis"]
    averages = [exploration_avg, slow_avg, avoidance_avg, wall_hugging_avg]
    std_devs = [exploration_std, slow_std, avoidance_std, wall_hugging_std]

    indices = np.arange(len(labels))
    bar_width = 0.45
    colors = [dict_colors[label] for label in labels]

    bars = ax.bar(
        indices,
        averages,
        bar_width,
        yerr=std_devs,
        capsize=5,
        color=colors,
        edgecolor="black",
    )

    # Freezing
    freezing_dist_avg, freezing_dist_std = calc_avg_std_percentages(
        data["freezing_dist"], data["total_time"]
    )
    freez_indice = 1.3
    freez_barwidth = 0.35
    freez_bar = ax.bar(
        freez_indice,
        freezing_dist_avg,
        freez_barwidth,
        yerr=freezing_dist_std,
        capsize=5,
        color=dict_colors["Freezing"],
        edgecolor="black",
        label="Freezing",
    )

    # Add labels and title
    # ax.set_xlabel('Behavior')
    fontsize_label = 12
    ax.set_ylabel(r"Percentage of time (\%)", fontsize=fontsize_label)
    ax.set_xticks(indices)
    ax.set_xticklabels(labels, fontsize=fontsize_label)
    ax.tick_params(axis="y", which="major", labelsize=12)
    # Adjust x-axis range
    ax.set_xlim(-0.5, len(labels) + 3)  #

    # Add legend with square box and black edgecolor
    fontsize_bar_label = 12
    legend_bars = [bars[0], bars[1], freez_bar, bars[2], bars[3]]
    legend_labels = [
        "Attraction",
        "Slow motion",
        "Freezing",
        "Avoidance",
        "Thigmotaxis",
    ]
    ax.legend(
        legend_bars,
        legend_labels,
        loc="upper left",
        fancybox=True,
        edgecolor="white",
        fontsize=fontsize_bar_label,
    )
    ax.set_ylim(0, 100)

    ax.text(
        0.02,
        1.05,
        r"\textbf{A.}",
        fontsize=14,
        weight="bold",
        transform=ax.transAxes,
        verticalalignment="center",
        horizontalalignment="center",
    )

    # Inset
    ax, inset_ax = plot_lin_corr_inset(ax, data)

    return ax


def plot_subjects(ax, data, dict_colors):
    # Number of subjects
    num_subjects = len(data["slow"])

    # Calculate percentages
    slow_percent = np.array(data["slow"]) / np.array(data["total_time"]) * 100
    avoidance_percent = np.array(data["avoidance"]) / np.array(data["total_time"]) * 100
    exploration_percent = (
        np.array(data["exploration"]) / np.array(data["total_time"]) * 100
    )
    wall_hugging_percent = (
        np.array(data["wall_hugging"]) / np.array(data["total_time"]) * 100
    )

    # Bar width and positions
    bar_width = 0.2
    indices = np.arange(num_subjects)

    # Plotting bars
    bar0 = ax.bar(
        indices - 1.5 * bar_width,
        exploration_percent,
        bar_width,
        label="Attraction",
        color=dict_colors["Attraction"],
        edgecolor="black",
    )  # Purple
    bar1 = ax.bar(
        indices - 0.5 * bar_width,
        slow_percent,
        bar_width,
        label="Slow motion",
        color=dict_colors["Slow motion"],
        edgecolor="black",
    )  # Pink
    bar2 = ax.bar(
        indices + 0.5 * bar_width,
        avoidance_percent,
        bar_width,
        label="Avoidance",
        color=dict_colors["Avoidance"],
        edgecolor="black",
    )  # Blue
    bar3 = ax.bar(
        indices + 1.5 * bar_width,
        wall_hugging_percent,
        bar_width,
        label="Thigmotaxis",
        color=dict_colors["Thigmotaxis"],
        edgecolor="black",
    )  # Green

    # Freezing
    freezing_dist_percent = (
        np.array(data["freezing_dist"]) / np.array(data["total_time"]) * 100
    )
    ax.bar(
        indices - 0.5 * bar_width,
        freezing_dist_percent,
        bar_width,
        label="Freezing",
        color=dict_colors["Freezing"],
        edgecolor="black",
    )

    # Adding labels and title
    # ax.set_xlabel('Subjects')
    fontsize_bar_label = 12
    ax.set_ylabel(r"Percentage of time (\%)", fontsize=fontsize_bar_label)
    ax.set_xticks(indices)
    ax.set_xticklabels(
        [f"Subject {i+1}" for i in range(num_subjects)], fontsize=fontsize_bar_label
    )
    ax.tick_params(axis="y", which="major", labelsize=12)

    # Display vertical dashed line between subjects
    for i in range(num_subjects - 1):
        ax.axvline(x=i + 0.5, color="gray", linestyle="--", linewidth=1.0)

    ax.set_ylim(0, 100)
    ax.set_xlim(-0.5, num_subjects - 0.5)

    ax.text(
        0.02,
        1.06,
        r"\textbf{B.}",
        fontsize=14,
        weight="bold",
        transform=ax.transAxes,
        verticalalignment="center",
        horizontalalignment="center",
    )

    return ax


def plot_behaviors(data):
    fig, axes = plt.subplots(
        2, 1, figsize=(10, 6), gridspec_kw={"height_ratios": [2, 1]}
    )
    fig.subplots_adjust(hspace=0.25)

    dict_colors = {
        "Attraction": "#cc99ff",
        "Slow motion": "#ff9999",
        "Avoidance": "#99ccff",
        "Thigmotaxis": "#99ff99",
        "Freezing": "#ffcc66",
    }
    plot_avg_std(axes[0], data, dict_colors)
    plot_subjects(axes[1], data, dict_colors)

    fig.savefig(f"{study.plot_output_folder}/PUBLISH/behaviors_full.pdf")


if __name__ == "__main__":
    # save behaviors (dict) if not exist otherwise load
    if not os.path.exists(f"{study.plot_output_folder}/PUBLISH/behaviors.pkl"):
        behaviors = get_behaviors()
        with open(f"{study.plot_output_folder}/PUBLISH/behaviors.pkl", "wb") as f:
            pickle.dump(behaviors, f)
    else:
        behaviors = pickle.load(
            open(f"{study.plot_output_folder}/PUBLISH/behaviors.pkl", "rb")
        )

    plot_behaviors(behaviors)
