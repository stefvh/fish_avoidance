import os
from matplotlib.collections import LineCollection
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

def color_line(ax, x, y, c):
    points = np.array([x, y]).T.reshape(-1,1,2) 
    segments = np.concatenate([points[:-1], points[1:]], axis=1)   
    lc = LineCollection(segments, cmap='plasma', linewidth=3) 
    lc.set_array(c)

    ax.plot(x, y, color="black", linewidth=0.5)
    ax.add_collection(lc) 
    ax.autoscale()
    plt.colorbar(lc, ax=ax)

    ax.axhline(y=0, color="black", linestyle="--", linewidth=0.2, zorder=0)

    return ax

def shorten_trial(trial_id, trial_results):
    if trial_id == 1:
        relative_frame_numbers = trial_results["frame_number"] - trial_results["frame_number"].min()
        trial_results = trial_results[relative_frame_numbers < 200]
    return trial_results

def plot_plateaus_by_rolling_window(ax, s, x):
    window_len = 5
    std_thresh = 1

    # Create a boolean mask which marks the less noisy points
    marked = s.rolling(window=window_len).std() < std_thresh
    marked = np.array(marked)
    # Whenever a new point is marked, mark also the other points of the window (see discussion below)
    for i in range(window_len + 1, len(marked)):
        if marked[i] and ~marked[i-1]:
            marked[i - (window_len-1) : i] = True

    # Plot the marked points
    ax.scatter(x[marked], s[marked], color='orange', s=10)

def compute_average_plateau_robot_velocity_before_min_distance(trial_results: pd.DataFrame):
    df = trial_results.copy()
    df.reset_index(inplace=True)
    min_d = df["robot_distance"].min()
    argmin_d = df["robot_distance"].idxmin()

    idx = argmin_d
    values = []
    bar = 1.5
    if min_d < 20:
        # go backwards in time until the velocity is greater than 20
        while df["robot_velocity_batch"].iloc[idx] < 20:
            idx -= 1
        fred = True
        while fred:
            window = df["robot_velocity_batch"].iloc[idx - 5:idx]
            if np.std(window) > bar:
                idx -= 1
            else:
                fred = False
    # go backwards in time until the velocity is less than 20
    # while while df["robot_velocity_batch"].iloc[idx] > 20
    fred = True
    while fred:
        window = df["robot_velocity_batch"].iloc[idx - 5:idx]
        if np.std(window) < bar:
            values.append(df["robot_velocity_batch"].iloc[idx])
            idx -= 1
        else:
            fred = False
    
    print(np.mean(values), np.std(values))
    return np.mean(values), np.std(values)


def plot_trial(ax_row, i, j, trial_id, results):
    k = i + j
    trial_results = results[results["trial_id"] == trial_id]
    trial_results = shorten_trial(trial_id, trial_results)

    relative_frame_numbers = trial_results["frame_number"] - trial_results["frame_number"].min()

    min_robot_distance = trial_results["robot_distance"].min()
    argmin_robot_distance = trial_results["robot_distance"].idxmin()
    x_rel_min = trial_results["frame_number"][argmin_robot_distance] - trial_results["frame_number"].min()
    # for ax in ax_row:
    #     ax.axvline(x=x_rel_min, color="red", linestyle="-")

    annotate_row(ax_row[0], trial_results["trial_id"].iloc[0], min_robot_distance)

    column = 0

    # Plot
    ax = ax_row[column]

    x = relative_frame_numbers
    y = trial_results["robot_distance_batch"]
    c = trial_results["fish_velocity_batch"]

    points = np.array([x, y]).T.reshape(-1,1,2) 
    segments = np.concatenate([points[:-1], points[1:]], axis=1)   
    lc = LineCollection(segments, cmap='plasma', linewidth=3) 
    lc.set_array(c)

    ax.plot(x, y, color="black", linewidth=0.5)
    ax.add_collection(lc) 
    ax.autoscale()
    plt.colorbar(lc, ax=ax)

    max_robot_distance = results["robot_distance"].max()
    ax.set_ylim(0, max_robot_distance)

    ax.axhline(y=2, color="black", linestyle="--", linewidth=0.2, zorder=0)
    ax.axhline(y=8, color="black", linestyle="--", linewidth=0.2, zorder=0)

    y2 = trial_results["robot_velocity_batch"]
    ax2 = ax.twinx()
    ax2.plot(x, y2, color="black", linewidth=0.8)
    max_y = results["robot_velocity_batch"].max()
    ax2.set_ylim(0, max_y)
    ax2.set_yticks([])

    # s = trial_results["robot_velocity_batch"]
    # x = relative_frame_numbers
    # plot_plateaus_by_rolling_window(ax2, s, x)
    prior_mean_r_vel, prior_std_r_vel = compute_average_plateau_robot_velocity_before_min_distance(trial_results)
    ax2.axhline(y=prior_mean_r_vel, color="red", linestyle="--", linewidth=0.6, zorder=0)
    ax2.axhline(y=prior_mean_r_vel + prior_std_r_vel, color="red", linestyle="--", linewidth=0.2, zorder=0)
    ax2.axhline(y=prior_mean_r_vel - prior_std_r_vel, color="red", linestyle="--", linewidth=0.2, zorder=0)

    column +=1

    # Plot
    ax = ax_row[column]

    x = relative_frame_numbers
    y = trial_results["alpha_clockwise_batch"]
    c = trial_results["fish_velocity_batch"]

    color_line(ax, x, y, c)

    column += 1

    # Plot
    ax = ax_row[column]

    x = relative_frame_numbers
    y = trial_results["beta_clockwise_batch"]
    c = trial_results["fish_velocity_batch"]

    color_line(ax, x, y, c)

    column += 1

    # Plot
    ax = ax_row[column]

    x = relative_frame_numbers
    y1 = trial_results["fish_distance_contributions"]
    y2 = trial_results["robot_distance_contributions"]

    ax.plot(x, y1, label="Fish", color="blue", linewidth=0.5)
    ax.plot(x, y2, label="Robot", color="red", linewidth=0.5)

    # ax.set_ylim(0.45, 2.05)
    ax.hlines(y=1, xmin=0, xmax=relative_frame_numbers.max(), color="black", linestyle="--", linewidth=0.2, zorder=0)

    ax.legend(loc="upper right", fontsize=6)

    column +=1

    # Plot
    ax = ax_row[column]

    x = relative_frame_numbers
    y = trial_results["fish_turn_batch"]
    c = trial_results["fish_velocity_batch"]

    color_line(ax, x, y, c)

    column +=1

    # Plot
    ax = ax_row[column]

    x = relative_frame_numbers
    y = trial_results["fish_turn_1hz"]
    y = abs(y)
    c = trial_results["fish_velocity_batch"]

    color_line(ax, x, y, c)

    column +=1

    
def get_categories():
    A = [0, 1, 6, 26, 34] # [0, 1, 5, 6, 26, 34]
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

def get_test_trials():
    categories = get_categories()
    category_a = categories[0]

    return category_a

def run(experiment: Experiment):
    # results is pd.DataFrame
    results = experiment.results

    output_folder = experiment.input_folder + "/plots"
    sub_folder = "category_a"
    output_filename = "trials_results.pdf"
    if not os.path.exists(f"{output_folder}/{sub_folder}"):
        os.makedirs(f"{output_folder}/{sub_folder}")

    trial_ids = get_test_trials()

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
