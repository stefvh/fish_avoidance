import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from post_process.feature_extraction import extract_approach_window_frame_by_follow_metric
from core.experiment import Experiment, get_experiment

from utils.parser import parse_args
from utils.plotter import plot_multipdf

from scipy.signal import find_peaks, peak_prominences

def annotate_row(ax, trial_id, min_robot_distance):
    x_textpad = -0.45
    annotate_color = "black"

    annotation_text = r'(' + f'{str(trial_id)}) '
    annotation_text += r'$d_{\min}=$' + f'{min_robot_distance:.1f}cm'
    
    ax.annotate(annotation_text, xy=(x_textpad, 0.45), xycoords="axes fraction",
            va='center', rotation="vertical", fontsize=12, weight='bold', color=annotate_color)


def moving_sum_of_differential(data, window_size):
    diff = np.diff(data)
    return np.convolve(diff, np.ones(window_size), mode="valid")


def moving_average_of_differential(data, window_size):
    diff = np.diff(data)
    diff = abs(diff)
    return np.convolve(diff, np.ones(window_size), mode="valid") / window_size


def plot_trial(ax_row, i, j, trial_id, results):
    k = i + j
    trial_results = results[results["trial_id"] == trial_id]

    relative_frame_numbers = trial_results["frame_number"] - trial_results["frame_number"].min()

    min_robot_distance = trial_results["robot_distance"].min()
    argmin_robot_distance = trial_results["robot_distance"].idxmin()
    x_rel_min = trial_results["frame_number"][argmin_robot_distance] - trial_results["frame_number"].min()
    for ax in ax_row[:2]:
        ax.axvline(x=x_rel_min, color="black", linestyle="--", linewidth=0.3)
    
    window_frame = extract_approach_window_frame_by_follow_metric(trial_results)
    for ax in ax_row[:2]:
        ax.axvline(x=window_frame[0], color="red", linestyle="--", linewidth=0.8)
        ax.axvline(x=window_frame[1], color="red", linestyle="--", linewidth=0.8)

    column = 0

    # Plot robot distance
    ax = ax_row[column]
    ax.plot(relative_frame_numbers, trial_results["robot_distance_batch"])
    if j ==0:
        ax.set_title("Robot Distance")
    ax.set_xlabel("Frame Number")
    ax.set_ylabel("Distance (cm)")

    max_robot_distance = results["robot_distance"].max()
    ax.set_ylim(0, max_robot_distance)

    annotate_row(ax, trial_results["trial_id"].iloc[0], min_robot_distance)

    column +=1

    # Plot fish velocity 
    ax = ax_row[column]
    ax.plot(relative_frame_numbers, trial_results["fish_velocity_batch"])
    if j ==0:
        ax.set_title("Fish Velocity")
    ax.set_xlabel("Frame Number")
    ax.set_ylabel("Velocity (cm/s)")

    max_fish_velocity = results["fish_velocity_batch"].max()
    ax.set_ylim(0, max_fish_velocity)

    ax2 = ax.twinx()
    window_size = 4
    y2 = moving_average_of_differential(trial_results["fish_velocity_batch"], window_size)
    ax2.plot(relative_frame_numbers[window_size:], y2, color="orange")
    ax2.set_ylabel("Sum of Differential")

    column +=1

    # Plot fish velocity slope
    ax = ax_row[column]
    y_slope = np.gradient(trial_results["fish_velocity_batch"])
    ax.plot(relative_frame_numbers, y_slope)
    if j ==0:
        ax.set_title("Fish Velocity Slope")
    ax.set_xlabel("Frame Number")
    ax.set_ylabel("Slope")

    column +=1

    # Plot fish velocity distribution
    ax = ax_row[column]
    ax.hist(trial_results["fish_velocity_batch"], bins=50, density=True)
    if j ==0:
        ax.set_title("Fish Velocity Distribution")
    ax.set_xlabel("Velocity (cm/s)")
    ax.set_ylabel("Density")

    column +=1

    # Plot fish acceleration
    ax = ax_row[column]
    ax.plot(relative_frame_numbers, trial_results["fish_acceleration_batch"])
    if j ==0:
        ax.set_title("Fish Acceleration")
    ax.set_xlabel("Frame Number")
    ax.set_ylabel("Acceleration (cm/s2)")

    max_fish_acceleration = results["fish_acceleration_batch"].max()
    ax.set_ylim(0, max_fish_acceleration)

    column +=1

    # Plot fish wall category
    ax = ax_row[column]
    ax.plot(relative_frame_numbers, trial_results["fish_wall_category"])
    if j ==0:
        ax.set_title("Fish Wall Category")
    ax.set_xlabel("Frame Number")
    ax.set_ylabel("Category")

    ax.set_ylim(-0.1, 3.1)

    column +=1

def run(experiment: Experiment):
    # results is pd.DataFrame
    results = experiment.results
    # create a list of trial_ids that is sorted based on the results.fish_wall_category
    trial_ids = results.trial_id.unique()
    wall_category_list = []
    for x in trial_ids:
        trial_results = results[results.trial_id == x]
        fish_wall_category = trial_results["fish_wall_category"]
        category_c = np.sum(fish_wall_category)
        wall_category_list.append(category_c)
    wall_category_list = [wall_category_list[i] for i in range(len(trial_ids)) if not np.isnan(wall_category_list[i])]
    # sort the trial_ids based on the distance_list
    trial_ids = [x for _, x in sorted(zip(wall_category_list, trial_ids))]

    # select_trials_category_a(results)

    output_folder = experiment.input_folder + "/plots"
    sub_folder = "observation"
    output_filename = "trials_fish_reactions.pdf"
    if not os.path.exists(f"{output_folder}/{sub_folder}"):
        os.makedirs(f"{output_folder}/{sub_folder}")

    plot_multipdf(
        len(trial_ids),
        5,
        6,
        lambda ax_row, i, j: plot_trial(
            ax_row=ax_row,
            i=i, 
            j=j,
            trial_id=trial_ids[i+j],
            results=results,
        ),
        f"{output_folder}/{sub_folder}/{output_filename}",
    )

if __name__ == "__main__":
    args = parse_args()
    variables = vars(args)

    experiment = get_experiment(**variables)
    run(experiment)
