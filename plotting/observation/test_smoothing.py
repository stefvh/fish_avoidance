import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.experiment import Experiment, get_experiment
from post_process.motion_quantities import MotionQuantities

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


def plot_trial(ax_row, i, j, trial_id, results, robot_mq: MotionQuantities, fish_mq: MotionQuantities):
    k = i + j
    trial_results = results[results["trial_id"] == trial_id]

    start_frame = trial_results["frame_number"].min()
    end_frame = trial_results["frame_number"].max()

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

    column +=1

    # Plot robot velocity (results)
    ax = ax_row[column]
    ax.plot(relative_frame_numbers, trial_results["robot_velocity_batch"])
    if j ==0:
        ax.set_title("Robot Velocity")
    ax.set_xlabel("Frame Number")
    ax.set_ylabel("Velocity (cm/s)")

    max_robot_velocity = results["robot_velocity_batch"].max()
    ax.set_ylim(0, max_robot_velocity)

    column +=1

    # Plot robot velocity (motion quantities)
    ax = ax_row[column]
    robot_speed = robot_mq.df[robot_mq.df["frame"].between(start_frame, end_frame)]["speed"]
    ax.plot(relative_frame_numbers, robot_speed)
    if j ==0:
        ax.set_title("Robot Speed")
    ax.set_xlabel("Frame Number")
    ax.set_ylabel("Speed (cm/s)")

    max_robot_speed = robot_mq.df["speed"].max()
    ax.set_ylim(0, max_robot_speed)

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

    column +=1

    # Plot fish velocity (mq)
    ax = ax_row[column]
    fish_speed = fish_mq.df[fish_mq.df["frame"].between(start_frame, end_frame)]["speed"]
    ax.plot(relative_frame_numbers, fish_speed)
    if j ==0:
        ax.set_title("Fish Speed")
    ax.set_xlabel("Frame Number")
    ax.set_ylabel("Speed (cm/s)")

    max_fish_speed = fish_mq.df["speed"].max()
    ax.set_ylim(0, max_fish_speed)

    column +=1

    # Plot fish turn (n=1)
    ax = ax_row[column]
    ax.plot(relative_frame_numbers, trial_results["fish_turn_n=1"])
    if j ==0:
        ax.set_title("Fish turn n=1")
    ax.set_xlabel("Frame Number")
    ax.set_ylabel("Turn (rad/s)")

    max_fish_turn = results["fish_turn_n=1"].max()
    ax.set_ylim(-max_fish_turn, max_fish_turn)

    column +=1

    # Plot fish turn (n=5)
    ax = ax_row[column]
    ax.plot(relative_frame_numbers, trial_results["fish_turn_n=5"])
    if j ==0:
        ax.set_title("Fish turn n=5")
    ax.set_xlabel("Frame Number")
    ax.set_ylabel("Turn (rad/s)")

    max_fish_turn = results["fish_turn_n=5"].max()
    ax.set_ylim(-max_fish_turn, max_fish_turn)

    column +=1

    # Plot fish turn (mq)
    ax = ax_row[column]
    fish_turn = fish_mq.df[fish_mq.df["frame"].between(start_frame, end_frame)]["turn"]
    ax.plot(relative_frame_numbers, fish_turn, label="Turn")
    fish_turn_intervaled = fish_mq.df[fish_mq.df["frame"].between(start_frame, end_frame)]["turn_intervaled"]
    ax.plot(relative_frame_numbers, fish_turn_intervaled, label="Turn Intervaled")
    # fish_turn_rolling = fish_mq.df[fish_mq.df["frame"].between(start_frame, end_frame)]["turn_rolling"]
    # ax.plot(relative_frame_numbers, fish_turn_rolling, label="Turn Rolling")
    if j ==0:
        ax.set_title("Fish Turn")
    ax.set_xlabel("Frame Number")
    ax.set_ylabel("Turn (rad/s)")

    # ax.legend()

    ax.set_ylim(-np.pi, np.pi)

    column +=1


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
    output_filename = "test_smoothing.pdf"
    if not os.path.exists(f"{output_folder}/{sub_folder}"):
        os.makedirs(f"{output_folder}/{sub_folder}")
    
    robot_mq = experiment.robot_motion_quantities
    fish_mq = experiment.fish_motion_quantities

    plot_multipdf(
        len(trial_ids),
        5,
        8,
        lambda ax_row, i, j: plot_trial(
            ax_row=ax_row,
            i=i, 
            j=j,
            trial_id=trial_ids[i+j],
            results=results,
            robot_mq=robot_mq,
            fish_mq=fish_mq
        ),
        f"{output_folder}/{sub_folder}/{output_filename}",
    )

if __name__ == "__main__":
    args = parse_args()
    variables = vars(args)

    experiment = get_experiment(**variables)
    run(experiment, variables["sort"])
