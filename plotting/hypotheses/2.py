import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pre_process.experiment import Experiment, get_experiment

from utils.parser import parse_args
from utils.plotter import plot_multipdf

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

    max_fish_velocity = results["fish_velocity_batch"].max()
    max_robot_distance = results["robot_distance"].max()

    relative_frame_numbers = trial_results["frame_number"] - trial_results["frame_number"].min()

    # Plot robot distance
    ax = ax_row[0]
    ax.plot(relative_frame_numbers, trial_results["robot_distance"])
    if k == 0:
        ax.set_title("Robot Distance")
    ax.set_xlabel("Frame Number")
    ax.set_ylabel("Distance (cm)")

    ax.set_ylim(0, max_robot_distance)

    min_robot_distance = trial_results["robot_distance"].min()
    argmin_robot_distance = trial_results["robot_distance"].idxmin()
    annotate_row(ax, trial_results["trial_id"].iloc[0], min_robot_distance)

    ax.axvline(x=trial_results["frame_number"][argmin_robot_distance] - trial_results["frame_number"].min(), color="red", linestyle="-")

    # Plot fish velocity
    ax = ax_row[1]
    ax.plot(relative_frame_numbers, trial_results["fish_velocity_batch"])
    if k == 0:
        ax.set_title("Fish Velocity")
    ax.set_xlabel("Frame Number")
    ax.set_ylabel("Velocity (cm/s)")

    ax.set_ylim(0, max_fish_velocity)

    ax.axvline(x=trial_results["frame_number"][argmin_robot_distance] - trial_results["frame_number"].min(), color="red", linestyle="-")

    # Plot fish orientation
    ax = ax_row[2]
    l_y = trial_results["fish_direction_batch"]
    n_y = []
    for y in l_y:
        if np.isnan(y).any():
            n_y.append(np.nan)
        else:
            n_y.append(np.arctan2(y[1], y[0]))
    ax.plot(relative_frame_numbers, n_y)
    if k == 0:
        ax.set_title("Fish Orientation")
    ax.set_xlabel("Frame Number")
    ax.set_ylabel("Orientation (rad)")

    ax.axvline(x=trial_results["frame_number"][argmin_robot_distance] - trial_results["frame_number"].min(), color="red", linestyle="-")


def run(experiment: Experiment):
    # results is pd.DataFrame
    results = experiment.results
    # create a list of trial_ids that is sorted based on the results.robot_distance
    trial_ids = results.trial_id.unique()
    distance_list = [results[results.trial_id == x].robot_distance.min() for x in trial_ids]
    # take out nan values
    all_trial_ids = trial_ids
    trial_ids = [trial_ids[i] for i in range(len(trial_ids)) if not np.isnan(distance_list[i])]
    # print trial_ids not in all_trial_ids
    print([x for x in all_trial_ids if x not in trial_ids])
    distance_list = [distance_list[i] for i in range(len(trial_ids)) if not np.isnan(distance_list[i])]
    # sort the trial_ids based on the distance_list
    trial_ids = [x for _, x in sorted(zip(distance_list, trial_ids))]


    plot_multipdf(
        len(trial_ids),
        5,
        3,
        lambda ax_row, i, j: plot_trial(
            ax_row=ax_row,
            i=i, 
            j=j,
            trial_id=trial_ids[i+j],
            results=results,
        ),
        f"plotting/hypotheses/2.pdf",
    )

if __name__ == "__main__":
    args = parse_args()
    variables = vars(args)

    experiment = get_experiment(**variables)
    run(experiment)
