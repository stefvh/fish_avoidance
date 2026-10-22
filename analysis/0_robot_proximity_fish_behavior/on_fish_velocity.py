import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from core.experiment import Experiment
from core.study import study

import subprocess


def _visualize_y_by_distance(y_var, y_label, experiment: Experiment):
    df = experiment.relative_motion_quantities.df.copy()
    # dump rows where speed_fish or distance is NaN
    df = df.dropna(subset=[y_var, "distance"])
    speeds_fish = df[y_var]
    distances = df["distance"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # [0, 0]
    # Scatter plot to visualize the relationship
    ax = axes[0]
    ax.scatter(distances, speeds_fish, s=5)
    ax.set_title(f"{y_label} vs Robot-Fish Distance (cm)")
    ax.set_xlabel("Robot-Fish Distance (cm)")
    ax.set_ylabel(f"{y_label}")

    # [0, 1]
    # Boxplot speed_fish grouped by categorized distance 
    ax = axes[1]
    max_distance = distances.max()
    min_distance = 0
    width = 5
    bins = np.arange(min_distance, max_distance + width, width)
    df["distance_category"] = pd.cut(df["distance"], bins=bins)
    df.boxplot(column=y_var, by="distance_category", ax=axes[1])
    ax.set_title(f"{y_label} vs Robot-Fish  Distance (cm)")
    ax.set_xlabel("Robot-Fish  Distance Category")
    ax.set_ylabel(f"{y_label}")
    # Count the number of data points in each box
    numbers = []
    for i, bin in enumerate(bins):
        if i == len(bins) - 1:
            continue
        x = (bin + bins[i + 1]) / 2
        y = df[df["distance_category"] == pd.Interval(bin, bins[i + 1])][y_var].count()
        numbers.append(y)
    # Write it below the ticks
    for i, txt in enumerate(numbers):
        axes[1].text(i + 1, 0, txt, ha="center")
    
    # Save the plot
    plt.tight_layout()
    plot_output_folder = study.plot_output_folder + "/robot_proximity_fish_behavior/"
    if not os.path.exists(plot_output_folder):
        os.makedirs(plot_output_folder)
    filename = f"E={experiment.id}_{y_var}_vs_distance.pdf"
    filepath = plot_output_folder + filename
    fig.savefig(filepath)


def visualize_fish_wall_distance_by_distance():
    for i, experiment in enumerate(study.list_experiments):
        _visualize_y_by_distance("min_wall_distance_fish", "Fish Min Wall Distance (cm)", experiment)

def visualize_fish_turn_by_distance():
    for i, experiment in enumerate(study.list_experiments):
        _visualize_y_by_distance("turn_fish", "Fish Turn (deg/s)", experiment)
    
def visualize_fish_speed_by_distance():
    for i, experiment in enumerate(study.list_experiments):
        _visualize_y_by_distance("speed_fish", "Fish Speed (cm/s)", experiment)


def run():
    columns = [
        "experiment_id",
        "frame",
        "speed_fish",
        "turn_fish",
        "min_wall_distance_fish",
        "distance",
        "speed_robot",
    ]
    df = study.rel_mq_df[columns].dropna()
    df["active_robot"] = df["speed_robot"].apply(lambda x: 1 if x > 0.1 else 0)
    df["presence_robot"] = (df["speed_robot"] / df["distance"])
    df["interaction_robot"] = df["speed_robot"] * df["distance"]

    for experiment_id in df["experiment_id"].unique():
        median = np.median(df[df["experiment_id"] == experiment_id]["speed_fish"])
        mad = np.median(np.abs(df[df["experiment_id"] == experiment_id]["speed_fish"] - median))
        df.loc[df["experiment_id"] == experiment_id, "high_speed_fish_outlier"] = df[df["experiment_id"] == experiment_id]["speed_fish"].apply(lambda x: 1 if x > median + 3 * mad else 0)


    # For each experiment_id, reset the frame to start from 1
    df["time"] = df.groupby("experiment_id")["frame"].transform(lambda x: x - x.min() + 1)

    # print first rows of df
    # print(df[df["experiment_id"] == 0])
    # print(df[df["experiment_id"] == 1])
    # print(df[df["experiment_id"] == 2])

    # Save the pandas dataframe to a CSV file
    df.to_csv('analysis/0_robot_proximity_fish_behavior/data.csv', index=False)
    


if __name__ == "__main__":
    run()
