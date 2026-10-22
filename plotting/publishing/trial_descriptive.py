from matplotlib import patches, pyplot as plt
import numpy as np
import pandas as pd

from core.experiment import Experiment
from core.study import study

import utils.plotter as plotter

def get_trials():
    remove_exp_trials = [
        [10, 21],
        [2, 11, 12, 13, 15],
        [8],
        [],
        []
    ]
    data = study.data

    exp_trials = []

    for exp in study.list_experiments:
        exp_id = exp.id
        experiment_datetime = exp.experiment_datetime

        trials = []
        remove_trials = remove_exp_trials[exp_id]
        df = pd.read_csv(f"pre_process/manual_trials/{experiment_datetime}.csv")
        for i, row in df.iterrows():
            if row["trial_id"] in remove_trials:
                continue
            trial_data = data[
                (data["experiment_id"] == exp_id)
                & (data["frame"] >= row["start_frame"])
                & (data["frame"] <= row["end_frame"])
            ]
            if trial_data.shape[0] != row["end_frame"] - row["start_frame"] + 1:
                print("Missing frames in trial data")
                print(f"Experiment ID: {exp_id}, Trial ID: {i}")
                print(
                    f"Expected: {row['end_frame'] - row['start_frame'] + 1}, Actual: {trial_data.shape[0]}"
                )
            else:
                trial = {
                    "experiment_id": exp_id,
                    "trial_id": i,
                    "m_trial_id": row["trial_id"],
                    "start_frame": row["start_frame"],
                    "end_frame": row["end_frame"],
                    "data": trial_data,
                }
                trials.append(trial)
        exp_trials.append(trials)
    
    print("-------------")

    return exp_trials


def plot_trajectory(ax, df):
    # ax.plot(trial["data"]["x_fish"], trial["data"]["y_fish"], label="Fish")
    # ax.plot(trial["data"]["x_robot"], trial["data"]["y_robot"], label="Robot")

    r_cmap = plt.cm.get_cmap('Blues')
    n_frames = df.shape[0]
    alphas = np.linspace(0.01, 1, n_frames)
    for i in range(n_frames - 1):
        ax.plot(df['x_robot'].iloc[i:i+2], df['y_robot'].iloc[i:i+2], color=r_cmap(alphas[i]), lw=4, label=r'Robot' if i == int(n_frames/2) else "")

    f_cmap = plt.cm.get_cmap('Greens')
    for i in range(n_frames - 1):
        ax.plot(df['x_fish'].iloc[i:i+2], df['y_fish'].iloc[i:i+2], color=f_cmap(alphas[i]), lw=4, label=r'Fish' if i == int(n_frames/2) else "")

    # Add a legend to differentiate between fish and robot
    # ax.legend(loc='upper left')

    # get the axis limits
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    dx = x_max - x_min
    dy = y_max - y_min

    x_min += 0.7 * dx

    # Add a gradient line in the corner with annotations
    # Define the line positions (relative to axis coordinates)
    bx = 0.11
    ex = 0.25
    gradient_x = np.linspace(x_min + bx * dx, x_min + ex * dx, 100)
    gradient_y = np.full_like(gradient_x, y_min + 0.925 * (y_max - y_min))

    # Use a more visible alpha range (from 0.2 to 0.8) for better contrast in the gradient
    gradient_alphas = np.linspace(0.01, 1.0, len(gradient_x))

    # Plot the gradient line
    for i in range(len(gradient_x) - 1):
        ax.plot(gradient_x[i:i+2], gradient_y[i:i+2], color=r_cmap(gradient_alphas[i]), lw=4)
    
    gradient_x = np.linspace(x_min + bx * dx, x_min + ex * dx, 100)
    gradient_y = np.full_like(gradient_x, y_min + 0.85 * (y_max - y_min))

    # Plot the gradient line
    for i in range(len(gradient_x) - 1):
        ax.plot(gradient_x[i:i+2], gradient_y[i:i+2], color=f_cmap(gradient_alphas[i]), lw=4)
    
    # Annotate the start and end of the gradient line (robot)
    ax.text(x_min + 0.18 * dx, y_min + 0.95 * (y_max - y_min), 'robot', fontsize=10, verticalalignment='center', horizontalalignment='center')
    ax.text(x_min + 0.03 * dx, y_min + 0.925 * (y_max - y_min), 'start', fontsize=10, verticalalignment='center', horizontalalignment='left')
    ax.text(x_min + 0.27 * dx, y_min + 0.925 * (y_max - y_min), 'end', fontsize=10, verticalalignment='center', horizontalalignment='left')

    # Annotate the start and end of the gradient line (fish)
    ax.text(x_min + 0.18 * dx, y_min + 0.87 * (y_max - y_min), 'fish', fontsize=10, verticalalignment='center', horizontalalignment='center')
    ax.text(x_min + 0.03 * dx, y_min + 0.85 * (y_max - y_min), 'start', fontsize=10, verticalalignment='center', horizontalalignment='left')
    ax.text(x_min + 0.27 * dx, y_min + 0.85 * (y_max - y_min), 'end', fontsize=10, verticalalignment='center', horizontalalignment='left')

    # Draw a box over both fish and robot gradient lines and text to look like a legend
    box_x = x_min + 0.01 * dx
    box_y = y_min + 0.82 * (y_max - y_min)
    box_width = 0.33 * dx
    box_height = 0.16 * (y_max - y_min)

    # Create a rectangle patch and add it to the axis
    rect = patches.Rectangle((box_x, box_y), box_width, box_height, linewidth=1, edgecolor='black', facecolor='none')
    ax.add_patch(rect)

    # # Plot rectangle around min distance
    # ef = np.argmin(df["distance"])
    # # Center of the rectangle is the x_fish and y_fish at the minimum distance
    # x_center = df["x_fish"].iloc[ef]
    # y_center = df["y_fish"].iloc[ef]
    # # Width and height of the rectangle
    # width = 5
    # height = 5
    # # Create a rectangle patch and add it to the axis
    # rect = patches.Rectangle((x_center - width / 2, y_center - height / 2), width, height, linewidth=1, edgecolor='black', facecolor='none')
    # ax.add_patch(rect)

    ax.set_xlabel(r'$x$ (cm)', fontsize=18)
    ax.set_ylabel(r'$y$ (cm)', fontsize=18)

    ax.text(0.03, 1.02, r"\textbf{A.}", fontsize=18, weight='bold',
            transform=ax.transAxes, verticalalignment='center', horizontalalignment='center')

    return ax
    

def plot_trial(trial, frame_end=None):
    df = trial["data"]

    if frame_end is not None:
        df = df[df["frame"] <= frame_end]

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    spacing = 0.4
    plt.subplots_adjust(hspace=spacing)
    plt.subplots_adjust(wspace=spacing)

    # merge the top left 4 subplots as one big subplot
    gs = axes[0, 0].get_gridspec()
    for ax in axes[0:2, 0:2].flatten():
        ax.remove()
    axbig = fig.add_subplot(gs[0:2, 0:2])

    # axbig: Plot the trajectory of fish and robot
    axbig = plot_trajectory(axbig, df)

    colors = {
        'distance': 'tab:orange',   # Orange
        'speed_fish': 'tab:brown',            # Magenta
        'rev': 'tab:cyan',    # Dark Gray
        'apv': 'tab:red',  # Red
        'min_wall_distance_fish': 'tab:purple'      # Purple
    }

    fontsize_label = 18

    # Fish min wall distance
    ax = axes[0, 2]
    ax.plot(df["frame"], df["min_wall_distance_fish"], color=colors['min_wall_distance_fish'])
    ax.set_ylabel(r"$d_{FW}$ (cm)", fontsize=fontsize_label)
    ax.text(0.05, 1.06, r"\textbf{B.}", fontsize=18, weight='bold',
            transform=ax.transAxes, verticalalignment='center', horizontalalignment='center')

    # Fish speed
    ax = axes[1, 2]
    ax.plot(df["frame"], df["speed_fish"], color=colors['speed_fish'])
    ax.set_ylabel(r"$v_{F}$ (cm/s)", fontsize=fontsize_label)
    ax.text(0.05, 1.06, r"\textbf{C.}", fontsize=18, weight='bold',
            transform=ax.transAxes, verticalalignment='center', horizontalalignment='center')

    # Distance
    ax = axes[2, 0]
    ax.plot(df["frame"], df["distance"], color=colors['distance'])
    ax.set_ylabel(r"$d_{FR}$ (cm)", fontsize=fontsize_label)
    ax.text(0.05, 1.06, r"\textbf{D.}", fontsize=18, weight='bold',
            transform=ax.transAxes, verticalalignment='center', horizontalalignment='center')

    ef = np.argmin(df["distance"])
    td = 10
    phase = (max(0, ef - td), ef + td + 1)
    # plot vertical lines at the start and end of the minimum distance
    phase_frames = df["frame"].iloc[phase[0]:phase[1]].to_numpy()
    # ax.axvline(phase_frames[0], color='red', linestyle='dashed', linewidth=1)
    # ax.axvline(phase_frames[-1], color='red', linestyle='dashed', linewidth=1)
    y_min, y_max = ax.get_ylim()
    ty = y_min + 0.9 * (y_max - y_min)
    tx = (min(df["frame"]) + phase_frames[0])/2
    ax.text(tx, ty, 'PRE', fontsize=10, verticalalignment='center', horizontalalignment='center')
    tx = (phase_frames[0] + phase_frames[-1])/2
    ax.text(tx, ty, 'AT', fontsize=10, verticalalignment='center', horizontalalignment='center')
    tx = (phase_frames[-1] + max(df["frame"]))/2
    ax.text(tx, ty, 'POST', fontsize=10, verticalalignment='center', horizontalalignment='center')

    for ax in axes.flatten():
        ax.axvspan(phase_frames[0], phase_frames[-1], color='gray', alpha=0.2)

    # Robot approach speed (apv)
    ax = axes[2, 1]
    ax.plot(df["frame"], df["apv"], color=colors['apv'])
    ax.set_ylabel(r"$v_{app}$ (cm/s)", fontsize=fontsize_label)
    ax.text(0.05, 1.06, r"\textbf{E.}", fontsize=18, weight='bold',
        transform=ax.transAxes, verticalalignment='center', horizontalalignment='center')

    # Fish retreat speed (rev)
    ax = axes[2, 2]
    ax.plot(df["frame"], df["rev"], color=colors['rev'])
    ax.set_ylabel(r"$v_{ret}$ (cm/s)", fontsize=fontsize_label)
    ax.text(0.05, 1.06, r"\textbf{F.}", fontsize=18, weight='bold',
            transform=ax.transAxes, verticalalignment='center', horizontalalignment='center')

    for ax in axes.flatten():
        ax.set_xlabel(r"time (25 Hz)", fontsize=fontsize_label)


    fig.savefig(f"{study.plot_output_folder}/PUBLISH/trial_descriptive.pdf")


if __name__ == "__main__":
    # Get trials
    exp_trials = get_trials()
    
    # Print the number of trials in each experiment
    for i, trials in enumerate(exp_trials):
        print(f"Experiment {i}: {len(trials)} trials")
    
    # Arbitrarily select trial
    exp_id = 0
    m_trial_id = 1
    frame_end = 2200
    trial = None
    for t in exp_trials[exp_id]:
        if t["m_trial_id"] == m_trial_id:
            trial = t
            break
    if trial is None:
        print("Trial not found")
        exit(1)

    # Plot the trial
    plot_trial(trial, frame_end=frame_end)
