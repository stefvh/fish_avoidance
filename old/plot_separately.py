import matplotlib.pyplot as plt
import pickle as pkl

import numpy as np
import os 
import sys

from priors.parameters import (
    SlowSpeed,
    FastSpeed,
    TransitionRate,
    TransitionDistance,
)

plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'text.latex.preamble': r'\usepackage{amsfonts}' r'\usepackage{amsmath}'
})

folder_input = "data/RoboFish_Commander_Data/JUNE 13 (SET 2)/20230613-165310"

folder_output = "plot_separately"
if not os.path.exists(folder_output):
    os.makedirs(folder_output)

with open(folder_input + "/results.pkl", "rb") as f:
    results = pkl.load(f)

def plot_default(ax):
    x_values = np.linspace(0, 10, 100)
    y_values = np.sin(x_values)
    ax.plot(x_values, y_values)

def plot_metric(ax, row_results, metric, equalize_x=False):
    if "batched" in metric["result_id"]:
        x_values = row_results["relative_time_batches"]
    else:
        x_values = row_results["relative_time_seconds"]
    y_values = row_results[metric["result_id"]]

    if len(x_values) == 0 or len(y_values) == 0:
        plot_default(ax)
    else:
        x_values = x_values[0]
        y_values = y_values[0]
        start_cut_off = len(x_values) - len(y_values)
        x_values = x_values[start_cut_off:]
        ax.plot(x_values, y_values)
        if equalize_x:
            ax.set_xlim([0, 35])
        ax.set_ylim(metric["ylim"])
        # ax.set_ylabel(metric["name"])
        # ax.set_xlabel("Time (s)")
        # ax.set_title(f"({fast_speed.name}, {transition_rate.name}, {transition_distance.name})")
        # ax.set_ylabel(metric["name"] + " (" + metric["unit"] + ")")

def plot(**kwargs):
    equalize_x = kwargs.get("equalize_x", False)
    equalize_x = bool(equalize_x)

    def_metrics = [
        {"result_id": "robot_velocities",
        "name": r"$v_{R}$",
        "unit": "cm/0.04s",
        "ylim": [0, 50],}
    ]
         
    metrics = [
        {"result_id": "distances",
        "name": r"$d_{FR}$", 
        "unit": "cm",
        "ylim": [0, 45],},
        {"result_id": "follow_vectors",
        "name": r"$\vec{f}_{FR}$", 
        "unit": "cm",
        "ylim": [-30, 30],},
        {"result_id": "velocity_changes",
        "name": r"\vec{v}_{FR}", 
        "unit": "cm/0.04s",
        "ylim": [-30, 30],},
        {"result_id": "acceleration_changes",
        "name": r"\vec{a}_{FR}", 
        "unit": "cm/0.04s",
        "ylim": [-500, 500],},
    ]

    for h, metric in enumerate(metrics):
        for i, slow_speed in enumerate([SlowSpeed.LOW, SlowSpeed.HIGH]):
            fig, axes = plt.subplots(12, 2, figsize=(28, 36))

            row = 0
            for j, fast_speed in enumerate(FastSpeed):
                for k, transition_rate in enumerate(TransitionRate):
                    for l, transition_distance in enumerate(TransitionDistance):
                        row_axes = axes[row]
                        row_results = results[row]

                        for m, def_metric in enumerate(def_metrics):
                            ax = row_axes[m]
                            plot_metric(ax, row_results, def_metric, equalize_x=equalize_x)
                        
                        ax = row_axes[m + 1]
                        plot_metric(ax, row_results, metric, equalize_x=equalize_x)

                        # Annotate transition distance
                        fontsize_row_label = 20
                        ax = row_axes[0]
                        x_textpad = -0.125  # -0.45
                        ax.annotate(r'\boldmath$T_d=' + f'{int(transition_distance.value)}' + r'$', xy=(x_textpad, 0.45), xycoords="axes fraction",
                                    va='center', rotation="vertical",
                                    fontsize=fontsize_row_label, weight='bold')
                        x_linepad = -0.075  # -0.29
                        ax.annotate('', xy=(x_linepad, -0.075), xycoords='axes fraction', xytext=(x_linepad, 1.01),
                                    arrowprops=dict(arrowstyle="-", color='black'))
                        # Annotate transition rate
                        if row % 2 == 0:
                            ax = row_axes[0]
                            x_textpad = -0.225
                            ax.annotate(r'\boldmath$T_r=' + f'{int(transition_rate.value)}' + r'$', xy=(x_textpad, 0.0), xycoords="axes fraction",
                                    va='center', rotation="vertical",
                                    fontsize=fontsize_row_label, weight='bold')
                            x_linepad = -0.175  # -0.29
                            ax.annotate('', xy=(x_linepad, -1.1), xycoords='axes fraction', xytext=(x_linepad, 1.01),
                                        arrowprops=dict(arrowstyle="-", color='black'))
                        # Annotate fast speed
                        if row % 6 == 0:
                            ax = row_axes[0]
                            x_textpad = -0.325
                            ax.annotate(r'\boldmath$V_f=' + f'{fast_speed.value}' + r'$', xy=(x_textpad, -2.5), xycoords="axes fraction",
                                    va='center', rotation="vertical",
                                    fontsize=fontsize_row_label, weight='bold')
                            x_linepad = -0.275
                            ax.annotate('', xy=(x_linepad, -5.85), xycoords='axes fraction', xytext=(x_linepad, 1.01),
                                arrowprops=dict(arrowstyle="-", color='black'))
                        row += 1
            
            metric_id = metric["result_id"]
            fig.savefig(f"{folder_output}/plot_M={metric_id}_SS={slow_speed.name}({slow_speed.value})_EX={equalize_x}.pdf", bbox_inches="tight", dpi=300)

if __name__ == "__main__":
    equalize_x = False
    if len(sys.argv) > 1:
        equalize_x = sys.argv[1]
    plot(equalize_x=equalize_x)