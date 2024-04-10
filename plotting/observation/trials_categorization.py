import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pre_process.experiment import Experiment, get_experiment

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

    # Plot robot state
    ax = ax_row[column]
    ax.plot(relative_frame_numbers, trial_results["state"])
    if j ==0:
        ax.set_title("Robot State")
    ax.set_xlabel("Frame Number")
    ax.set_ylabel("State")

    column +=1

    # Plot robot velocity
    ax = ax_row[column]
    # take each 5th elemnt of relative_frame_numbers
    rfn = relative_frame_numbers[::5]
    tr = trial_results["robot_velocity_batch"][::5]
    ax.plot(rfn, tr)
    if j ==0:
        ax.set_title("Robot Velocity")
    ax.set_xlabel("Frame Number")
    ax.set_ylabel("Velocity (cm/s)")

    max_robot_velocity = results["robot_velocity_batch"].max()
    ax.set_ylim(0, max_robot_velocity)

    column +=1

    # Plot alpha
    ax = ax_row[column]
    ax.plot(relative_frame_numbers, trial_results["alpha_batch"])
    ax.plot(relative_frame_numbers, trial_results["alpha_clockwise_batch"])
    if j ==0:
        ax.set_title("Alpha")
    ax.set_xlabel("Frame Number")
    ax.set_ylabel("Alpha")

    ax.set_ylim(-np.pi, np.pi)

    column +=1

    # Plot beta
    ax = ax_row[column]
    ax.plot(relative_frame_numbers, trial_results["beta_batch"])
    ax.plot(relative_frame_numbers, trial_results["beta_clockwise_batch"])
    if j ==0:
        ax.set_title("Beta")
    ax.set_xlabel("Frame Number")
    ax.set_ylabel("Beta")

    ax.set_ylim(-np.pi, np.pi)

    column +=1

    # Plot robot wall category
    ax = ax_row[column]
    ax.plot(relative_frame_numbers, trial_results["robot_wall_category"])
    if j ==0:
        ax.set_title("Robot Wall Category")
    ax.set_xlabel("Frame Number")
    ax.set_ylabel("Category")

    ax.set_ylim(-0.1, 3.1)

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


# def get_trial_automatic_category(trial_id: int, experiment: Experiment):
#     results = experiment.results
#     trial_results = results[results["trial_id"] == trial_id]

#     robot_distances = trial_results["robot_distance"]
#     frame_numbers = trial_results["frame_number"]

#     fn = frame_numbers[robot_distances >= 35]
#     if fn.empty:
#         return -1
#     rel_fn = fn - frame_numbers.min()
#     rel_fn = rel_fn[rel_fn < 100]
#     rel_fn_start_approach = rel_fn.max()
#     fn_start_approach = rel_fn_start_approach + frame_numbers.min()
    
def get_categories():
    A = [0, 1, 5, 6, 26, 34]
    B = [4, 21, 24, 27, 28, 43]
    C = [32]
    D = [2, 9, 10, 13, 40, 41]
    E = [23, 25, 36]
    F = [33]
    G = [37]
    H = [11, 12]
    I = [3, 7, 8, 20, 29, 38, 39]
    J = [22, 35, 42]

    return (A, B, C, D, E, F, G, H, I, J)


def get_trial_manual_category(trial_id):
    categories = get_categories()
    for i, category in enumerate(categories):
        if trial_id in category:
            return i
    
    return -1

def select_trials_category_a(results: pd.DataFrame):
    selected_trial_ids = []

    for trial_id in results["trial_id"].unique():
        trial_results = results[results["trial_id"] == trial_id]

        # First criteria
        robot_distance = trial_results["robot_distance"]
        robot_wall_category = trial_results["robot_wall_category"]

        # find minimum distance
        argmin_distance = robot_distance.idxmin()

        print(trial_id, argmin_distance, robot_distance[argmin_distance], robot_wall_category[argmin_distance])

        rd_threshold = 15
        tr = trial_results[
            (robot_distance < rd_threshold) & 
            (robot_wall_category == 0)
        ]

        fn_max = tr["frame_number"].max()

        # check if frame number is continuous
        for fn in range(tr["frame_number"].min(), tr["frame_number"].max()):
            if fn not in tr["frame_number"].values:
                fn_max = fn
                print("not continuous")
                break
        
        print(trial_id, tr["frame_number"].min(), fn_max, tr["frame_number"].max())

        print("-----")


    
    return selected_trial_ids

def get_test_trials():
    categories = get_categories()
    category_a = categories[0]

    return category_a

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


    robot_states = experiment.robot_states
    results = results.merge(robot_states, on="frame_number", how="left")
    results = results.dropna(subset=["state"])

    # select_trials_category_a(results)

    output_folder = experiment.input_folder + "/plots"
    sub_folder = "observation"
    output_filename = "trials_categorization.pdf"
    if not os.path.exists(f"{output_folder}/{sub_folder}"):
        os.makedirs(f"{output_folder}/{sub_folder}")

    plot_multipdf(
        len(trial_ids),
        5,
        9,
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
