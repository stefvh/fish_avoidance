import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.experiment import Experiment, get_experiment

from utils.parser import get_default_parser
from utils.plotter import plot_multipdf

from scipy.signal import find_peaks, peak_prominences


def parse_args():
    parser = get_default_parser()
    parser.add_argument("--sort", action="store_true")

    args = parser.parse_args()
    return args


def annotate_row(ax, trial_id, min_robot_distance):
    x_textpad = -0.45
    annotate_color = "black"

    annotation_text = r'(' + f'{str(trial_id)}) '
    annotation_text += r'$d_{\min}=$' + f'{min_robot_distance:.1f}cm'
    
    ax.annotate(annotation_text, xy=(x_textpad, 0.45), xycoords="axes fraction",
            va='center', rotation="vertical", fontsize=12, weight='bold', color=annotate_color)


def plot_trial(ax_row, i, j, trial_id, results):
    k = i + j
    trial_results = results[results["trial_id"] == trial_id]

    relative_frame_numbers = trial_results["frame_number"] - trial_results["frame_number"].min()

    min_robot_distance = trial_results["robot_distance"].min()
    argmin_robot_distance = trial_results["robot_distance"].idxmin()
    x_rel_min = trial_results["frame_number"][argmin_robot_distance] - trial_results["frame_number"].min()
    for ax in ax_row:
        ax.axvline(x=x_rel_min, color="red", linestyle="-")

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

    ax.axhline(y=2, color="black", linestyle="--")
    ax.axhline(y=8, color="black", linestyle="--")

    annotate_row(ax, trial_results["trial_id"].iloc[0], min_robot_distance)

    # fps = experiment.frequency
    # peaks, _ = find_peaks(trial_results["robot_distance"], distance=fps, prominence=5, width=fps)
    # dips, dips_props = find_peaks(-trial_results["robot_distance"])

    # ax.plot(peaks, trial_results["robot_distance"].iloc[peaks], "x")
    # ax.plot(dips, trial_results["robot_distance"].iloc[dips], "o")

    # dips_prom = peak_prominences(-trial_results["robot_distance"], dips)[0]
    # print(f"Trial {trial_id} dip props: {dips_prom}")

    # 6 -> 19.5
    # 12 -> 14.8

    column +=1

    # Plot robot velocity
    ax = ax_row[column]
    ax.plot(relative_frame_numbers, trial_results["robot_velocity_batch"])
    if j ==0:
        ax.set_title("Robot Velocity")
    ax.set_xlabel("Frame Number")
    ax.set_ylabel("Velocity (cm/s)")

    max_robot_velocity = results["robot_velocity_batch"].max()
    ax.set_ylim(0, max_robot_velocity)

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

    # fps = experiment.frequency
    # peaks, peak_props = find_peaks(trial_results["fish_velocity_batch"], distance=fps, height=5, prominence=5)
    # print(f"Trial {trial_id} peak props: {peak_props}")
    # ax.plot(peaks, trial_results["fish_velocity_batch"].iloc[peaks], "x")

    column +=1

    # Plot fish orientation
    ax = ax_row[column]
    l_y = trial_results["fish_direction_batch"]
    n_y = []
    for y in l_y:
        if np.isnan(y).any():
            n_y.append(np.nan)
        else:
            n_y.append(np.arctan2(y[1], y[0]))
    ax.plot(relative_frame_numbers, n_y)
    if j ==0:
        ax.set_title("Fish Orientation")
    ax.set_xlabel("Frame Number")
    ax.set_ylabel("Orientation (rad)")

    column +=1

    # Plot robot distance slope
    ax = ax_row[column]
    ax.plot(relative_frame_numbers, trial_results["robot_distance_slopes"])
    if j ==0:
        ax.set_title("Distance Slope")
    ax.set_xlabel("Frame Number")
    ax.set_ylabel("Slope (cm/s)")

    max_robot_distance_slope = results["robot_distance_slopes"].max()
    min_robot_distance_slope = results["robot_distance_slopes"].min()
    ax.set_ylim(min_robot_distance_slope, max_robot_distance_slope)

    column +=1

    # Plot fish velocity slope
    ax = ax_row[column]
    ax.plot(relative_frame_numbers, trial_results["fish_velocity_slopes"])
    if j ==0:
        ax.set_title("Fish Velocity Slope")
    ax.set_xlabel("Frame Number")
    ax.set_ylabel("Slope (cm/s)")

    max_fish_velocity_slope = results["fish_velocity_slopes"].max()
    min_fish_velocity_slope = results["fish_velocity_slopes"].min()
    ax.set_ylim(min_fish_velocity_slope, max_fish_velocity_slope)

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

    # # Plot robot velocity slope
    # ax = ax_row[column]
    # ax.plot(relative_frame_numbers, trial_results["robot_velocity_slopes"])
    # if j ==0:
    #     ax.set_title("Robot Velocity Slope")
    # ax.set_xlabel("Frame Number")
    # ax.set_ylabel("Slope (cm/s)")

    # max_robot_velocity_slope = results["robot_velocity_slopes"].max()
    # min_robot_velocity_slope = results["robot_velocity_slopes"].min()
    # ax.set_ylim(min_robot_velocity_slope, max_robot_velocity_slope)

    # column +=1


def run(experiment: Experiment, b_sort: bool = False):
    # results is pd.DataFrame
    results = experiment.results
    # create a list of trial_ids that is sorted based on the results.robot_distance
    trial_ids = results.trial_id.unique()
    if b_sort:
        distance_list = [results[results.trial_id == x].robot_distance.min() for x in trial_ids]
        # take out nan values
        all_trial_ids = trial_ids
        trial_ids = [trial_ids[i] for i in range(len(trial_ids)) if not np.isnan(distance_list[i])]
        # print trial_ids not in all_trial_ids
        invalid_trials = [x for x in all_trial_ids if x not in trial_ids]
        if len(invalid_trials) > 0:
            print(invalid_trials)
        distance_list = [distance_list[i] for i in range(len(trial_ids)) if not np.isnan(distance_list[i])]
        # sort the trial_ids based on the distance_list
        trial_ids = [x for _, x in sorted(zip(distance_list, trial_ids))]

    output_folder = experiment.input_folder + "/plots"
    sub_folder = "observation"
    output_filename = "trials_overview.pdf"
    if not os.path.exists(f"{output_folder}/{sub_folder}"):
        os.makedirs(f"{output_folder}/{sub_folder}")

    plot_multipdf(
        len(trial_ids),
        5,
        7,
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
    run(experiment, variables["sort"])
