import matplotlib.pyplot as plt
import pickle as pkl

import numpy as np
import os 
import sys

from analyse_measuring import Results
from utils.slope import compute_transition_speed

plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'text.latex.preamble': r'\usepackage{amsfonts}' r'\usepackage{amsmath}'
})

folder_input = "data/RoboFish_Commander_Data/JUNE 13 (SET 2)/20230613-165310"

folder_output = "plot_trials"
if not os.path.exists(folder_output):
    os.makedirs(folder_output)

with open(folder_input + "/parameters.pkl", "rb") as f:
    parameters = pkl.load(f)

with open(folder_input + "/trials.pkl", "rb") as f:
    trials = pkl.load(f)

metrics = [
    {"result_id": "list_robot_fish_repulsion",
    "name": r"Repulsion (-Follow)", 
    "unit": "",
    "ylim": [-0.5, 0.5],},
    {"result_id": "list_robot_fish_distance",
    "name": r"Distance ($d$)",
    "notation": r"$d_{FR}$", 
    "unit": "cm",
    "ylim": None,},
    {"result_id": "list_fish_velocity", 
    "name": r"Fish Velocity",
    "notation": r"$v_{F}$", 
    "unit": "cm/0.04s",
    "ylim": None,},
    {"result_id": "list_robot_velocity",
    "name": r"Robot Velocity",
    "notation": r"$v_{R}$", 
    "unit": "cm/0.04s",
    "ylim": None,},
    {"result_id": "list_robot_fish_vector_angle",
    "name": r"Robot-Fish Angle",
    "notation": r"$\theta_{FR}$", 
    "unit": "rad",
    "ylim": [0, 3.14],},
    {"result_id": "list_fish_orientation",
    "name": r"Fish Orientation",
    "notation": r"$\theta_{F}$", 
    "unit": "rad",
    "ylim": None,},
]

trials = sorted(trials, key=lambda x: x["results"].min_fish_robot_distance, reverse=False)

n_rows = 5
n_cols = len(metrics) + 1

def annotate_row(ax, trial):
    x_textpad = -0.45
    annotate_color = "black"
    if trial["type"].name == "STRAIGHT":
        annotate_color = "black"
    if trial["type"].name == "SIDEWAYS":
        annotate_color = "red"
    if trial["type"].name == "HALF":
        annotate_color = "orange"
    # annotation_text = r'(' + f'{str(trial["id"])}) {trial["type"].name}'
    annotation_text = r'(' + f'{str(trial["id"])}) '
    
    min_distance = trial["results"].min_fish_robot_distance
    annotation_text += r'$d_{\min}=$' + f'{min_distance:.1f}cm'
    
    ax.annotate(annotation_text, xy=(x_textpad, 0.45), xycoords="axes fraction",
            va='center', rotation="vertical", fontsize=12, weight='bold', color=annotate_color)

def plot_trial(i, j, axes, trial):
    results = trial["results"]
    results_dict = {
        "list_robot_velocity": results.list_robot_velocity,
        "list_fish_orientation": results.list_fish_orientation,
        "list_robot_fish_repulsion": results.list_robot_fish_repulsion,
        "list_fish_velocity": results.list_fish_velocity,
        "list_robot_fish_distance": results.list_robot_fish_distance,
        "list_robot_fish_vector_angle": results.list_robot_fish_angle,
    }
    list_rel_time_sec = results.list_rel_time_sec

    x_values = list_rel_time_sec
    y_values = results_dict["list_robot_fish_distance"]
    # find x-value where y-value is minimum
    min_index = np.argmin(y_values)
    min_x = x_values[min_index]

    for k, metric in enumerate(metrics):
        result_id = metric["result_id"]
        name = metric["name"]
        unit = metric["unit"]
        ylim = metric["ylim"]        

        ax = axes[k]
        x_values = list_rel_time_sec
        y_values = results_dict[result_id]
        start_cut_off = len(x_values) - len(y_values)
        x_values = x_values[start_cut_off:]
        ax.plot(x_values, y_values)

        if j == 0:
            ax.set_title(name)
        if result_id == "list_robot_fish_repulsion":
            ax.hlines(y=0, xmin=0, xmax=x_values[-1], linewidth=1, linestyle="--", color='black')
        if result_id == "list_robot_fish_vector_angle":
            ax.hlines(y=np.pi / 2, xmin=0, xmax=x_values[-1], linewidth=1, linestyle="--", color='black')
        if ylim is not None:
            ax.set_ylim(ylim)
            ax.vlines(x=min_x, ymin=ylim[0], ymax=ylim[1], linewidth=1, linestyle="-", color='red')
        else:
            ax.vlines(x=min_x, ymin=min(y_values), ymax=max(y_values), linewidth=1, linestyle="-", color='red')
        if i == 0:
            if k == 0:
                ax.set_ylim((-0.5, 1.0))
            if k == 1:
                ax.set_ylim((0, 45))
            if k == 2:
                ax.set_ylim((0, 32.5))
            if k == 3:
                ax.set_ylim((0, 30))
            if k == 4:
                ax.set_ylim((0, 3.14))
            if k == 5:
                ax.set_ylim((0, 10.5))
            # get ylim from ax
            ylim = ax.get_ylim()
            ax.vlines(x=min_x, ymin=ylim[0], ymax=ylim[1], linewidth=1, linestyle="-", color='red')
        
        if j == n_rows - 1:
            if k < n_cols - 1:
                ax.set_xlabel("Time (s)")
    
    # Theoretical transition speed
    ax = axes[-1]
    params = parameters.loc[[trial["id"]]]
    x_values = np.linspace(40, 0, 100)
    y_values = []
    for x in x_values:
        y = compute_transition_speed(
            fish_distance=x,
            transition_distance=params["transition_distance"].values[0],
            slow_speed_cm_s=params["slow_speed"].values[0],
            fast_speed_cm_s=params["fast_speed"].values[0],
            sigmoid_slope=params["transition_rate"].values[0],
        )
        y_values.append(y)
    ax.plot(x_values, y_values)
    ax.vlines(x=params["transition_distance"].values[0], ymin=params["slow_speed"].values[0], ymax=params["fast_speed"].values[0], linewidth=1, linestyle="--", color='black')
    ax.hlines(y=params["slow_speed"].values[0], xmin=0, xmax=40, linewidth=1, linestyle="--", color='green')
    ax.hlines(y=params["fast_speed"].values[0], xmin=0, xmax=40, linewidth=1, linestyle="--", color='red')
    ax.set_ylim((1.1, 2.1))
    ax.set_yticks([1.2, 1.6, 2.0])
    ax.invert_xaxis()
    if j == 0:
        ax.set_title("Theoretical robot vel.")
    if j == n_rows - 1:
        ax.set_xlabel("Distance (cm)")

    annotate_row(axes[0], trial)

def plot(**kwargs):
    for i in range(len(trials)):
        if i % n_rows != 0:
            continue
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols + 2, 2 * n_rows))
        # increase horizontal spacing between subplots
        fig.subplots_adjust(wspace=2/6)

        for j in range(n_rows):
            if i + j >= len(trials):
                break
            axs = axes[j]
            trial = trials[i + j]
            plot_trial(i, j, axs, trial)
        
        fig.savefig(f"{folder_output}/" + str(i) + ".pdf", bbox_inches="tight", dpi=300)

if __name__ == "__main__":
    plot()