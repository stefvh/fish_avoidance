import matplotlib.pyplot as plt
import pickle as pkl

import numpy as np
import os 
import sys

from parameters import (
    SlowSpeed,
    FastSpeed,
    TransitionRate,
    TransitionDistance,
)

folder_input = "data/RoboFish_Commander_Data/JUNE 13 (SET 2)/20230613-165310"

folder_output = "plot_all_metrics"
if not os.path.exists(folder_output):
    os.makedirs(folder_output)

with open(folder_input + "/results.pkl", "rb") as f:
    results = pkl.load(f)

def plot_default(ax):
    x_values = np.linspace(0, 10, 100)
    y_values = np.sin(x_values)
    ax.plot(x_values, y_values)

def plot(**kwargs):
    equalize_x = kwargs.get("equalize_x", False)
    equalize_x = bool(equalize_x)

    for i, slow_speed in enumerate([SlowSpeed.LOW, SlowSpeed.HIGH]):
        fig, axes = plt.subplots(12, 5, figsize=(36, 20))

        row = 0
        for j, fast_speed in enumerate(FastSpeed):
            for k, transition_rate in enumerate(TransitionRate):
                for l, transition_distance in enumerate(TransitionDistance):
                    row_axes = axes[row]
                    row_results = results[row]

                    metrics = [
                        {"result_id": "robot_velocities", 
                        "name": r"$v_{R}$", 
                        "unit": "cm/0.04s",
                        "ylim": [0, 50],},
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

                    for m, metric in enumerate(metrics):
                        # print(f"Plotting {metric["result_id"]}")
                        ax = row_axes[m]
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

                    
                    row += 1
        
        fig.savefig(f"{folder_output}/plot_EX={equalize_x}_SlowSpeed={slow_speed.name}({slow_speed.value}).pdf", bbox_inches="tight", dpi=300)

if __name__ == "__main__":
    equalize_x = False
    if len(sys.argv) > 1:
        equalize_x = sys.argv[1]
    plot(equalize_x=equalize_x)