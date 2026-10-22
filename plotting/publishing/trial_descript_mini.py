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

    # Get the axis limits
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    dx = x_max - x_min
    dy = y_max - y_min

    # Adjust x_min for the larger legend
    x_min += 0.7 * dx

    # Define larger line width and gradient position for better visibility
    bx = 0.11
    ex = 0.25
    line_width = 6  # Increase line width for visibility

    x_min -= 0.1 * dx
    y_min -= 0.5 * dy

    # Define gradient positions for the robot
    gradient_x = np.linspace(x_min + bx * dx, x_min + ex * dx, 100)
    gradient_y_robot = np.full_like(gradient_x, y_min + 0.9 * (y_max - y_min))

    # Use a more visible alpha range (from 0.01 to 1.0) for better contrast in the gradient
    gradient_alphas = np.linspace(0.01, 1.0, len(gradient_x))

    # Plot the gradient line for robot
    for i in range(len(gradient_x) - 1):
        ax.plot(gradient_x[i:i+2], gradient_y_robot[i:i+2], color=r_cmap(gradient_alphas[i]), lw=line_width)

    # Define gradient positions for the fish
    gradient_y_fish = np.full_like(gradient_x, y_min + 0.775 * (y_max - y_min))

    # Plot the gradient line for fish
    for i in range(len(gradient_x) - 1):
        ax.plot(gradient_x[i:i+2], gradient_y_fish[i:i+2], color=f_cmap(gradient_alphas[i]), lw=line_width)

    # Annotate the start and end of the gradient line (robot) with larger font size
    fontsize = 22
    ax.text(x_min + 0.18 * dx, y_min + 0.95 * (y_max - y_min), 'robot', fontsize=fontsize, verticalalignment='center', horizontalalignment='center')
    ax.text(x_min + -0.05 * dx, y_min + 0.9 * (y_max - y_min), 'start', fontsize=fontsize, verticalalignment='center', horizontalalignment='left')
    ax.text(x_min + 0.27 * dx, y_min + 0.9 * (y_max - y_min), 'end', fontsize=fontsize, verticalalignment='center', horizontalalignment='left')

    # Annotate the start and end of the gradient line (fish) with larger font size
    ax.text(x_min + 0.18 * dx, y_min + 0.825 * (y_max - y_min), 'fish', fontsize=fontsize, verticalalignment='center', horizontalalignment='center')
    ax.text(x_min + -0.05 * dx, y_min + 0.775 * (y_max - y_min), 'start', fontsize=fontsize, verticalalignment='center', horizontalalignment='left')
    ax.text(x_min + 0.27 * dx, y_min + 0.775 * (y_max - y_min), 'end', fontsize=fontsize, verticalalignment='center', horizontalalignment='left')

    # Adjust the size of the box to match the larger elements
    # box_x = x_min + -0.07 * dx
    # box_y = y_min + 0.8 * (y_max - y_min)  # Move box down slightly
    # box_width = 0.3 * dx  # Increase box width to accommodate larger text
    # box_height = 0.18 * (y_max - y_min)  # Increase height to fit larger gradient lines

    # # Create a rectangle patch and add it to the axis
    # rect = patches.Rectangle((box_x, box_y), box_width, box_height, linewidth=2, edgecolor='black', facecolor='none')
    # ax.add_patch(rect)

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

    ax.set_xlabel(r'$x$ (cm)', fontsize=28)
    ax.set_ylabel(r'$y$ (cm)', fontsize=28)

    ax.text(0.05, 1.06, r"\textbf{A.}", fontsize=28, weight='bold',
            transform=ax.transAxes, verticalalignment='center', horizontalalignment='center')

    return ax
    

def plot_trial(trial, frame_end=None):
    df = trial["data"]

    if frame_end is not None:
        df = df[df["frame"] <= frame_end]

    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    spacing = 0.25
    # plt.subplots_adjust(hspace=spacing)
    plt.subplots_adjust(wspace=spacing)

    axbig = axes[0]

    # axbig: Plot the trajectory of fish and robot
    axbig = plot_trajectory(axbig, df)

    colors = {
        'distance': 'tab:orange',   # Orange
        'speed_fish': 'tab:brown',            # Magenta
        'rev': 'tab:cyan',    # Dark Gray
        'apv': 'tab:red',  # Red
        'min_wall_distance_fish': 'tab:purple'      # Purple
    }

    fontsize_label = 28
    fontsize_annotation = 28

    # Distance
    ax = axes[1]
    ax.plot(df["frame"], df["distance"], color=colors['distance'])
    ax.set_ylabel(r"$d_{FR}$ (cm)", fontsize=fontsize_label)
    ax.text(0.05, 1.06, r"\textbf{B.}", fontsize=fontsize_annotation, weight='bold',
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
    fontsize_text = 24
    ax.text(tx, ty, 'PRE', fontsize=fontsize_text, verticalalignment='center', horizontalalignment='center')
    tx = (phase_frames[0] + phase_frames[-1])/2
    ax.text(tx, ty, 'AT', fontsize=fontsize_text, verticalalignment='center', horizontalalignment='center')
    tx = (phase_frames[-1] + max(df["frame"]))/2
    ax.text(tx, ty, 'POST', fontsize=fontsize_text, verticalalignment='center', horizontalalignment='center')

    for ax in axes[1:].flatten():
        ax.axvspan(phase_frames[0], phase_frames[-1], color='gray', alpha=0.2)

    # Robot approach speed (apv)
    ax = axes[2]
    ax.plot(df["frame"], df["apv"], color=colors['apv'])
    ax.set_ylabel(r"$v_{app}^R$ (cm/s)", fontsize=fontsize_label)
    ax.text(0.05, 1.06, r"\textbf{C.}", fontsize=fontsize_annotation, weight='bold',
        transform=ax.transAxes, verticalalignment='center', horizontalalignment='center')

    # Fish retreat speed (rev)
    ax = axes[3]
    ax.plot(df["frame"], df["rev"], color=colors['rev'])
    ax.set_ylabel(r"$v_{avoid}^F$ (cm/s)", fontsize=fontsize_label)
    ax.text(0.05, 1.06, r"\textbf{D.}", fontsize=fontsize_annotation, weight='bold',
            transform=ax.transAxes, verticalalignment='center', horizontalalignment='center')

    # for ax in axes[1:].flatten():
    #     ax.set_xlabel(r"time (25 Hz)", fontsize=fontsize_label)
    # Convert frame axis (25 Hz) to seconds for all time-based subplots
    desired_secs = np.array([80, 82, 84, 86, 88])
    desired_frames = desired_secs * 25  # convert seconds → frames

    for ax in axes[1:].flatten():
        ax.set_xticks(desired_frames)
        ax.set_xticklabels([f"{s:.0f}" for s in desired_secs])
        ax.set_xlabel(r"time (s)", fontsize=fontsize_label)
    # for ax in axes[1:].flatten():
    #     # Convert xticks from frames to seconds
    #     xticks = ax.get_xticks()
    #     ax.set_xticks(xticks)
    #     ax.set_xticklabels([f"{x/25:.1f}" for x in xticks])  # display 1 decimal place
    #     ax.set_xlabel(r"time (s)", fontsize=fontsize_label)

    
    # set ticks fontsize
    for ax in axes.flatten():
        ax.tick_params(axis='both', which='major', labelsize=20)
    
    # extend y of the figure so the ylabels show
    fig.tight_layout()


    fig.savefig(f"{study.plot_output_folder}/PUBLISH/trial_descriptive_mini.pdf")


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
