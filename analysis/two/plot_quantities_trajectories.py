import os
import numpy as np
import pandas as pd
from core.experiment import Experiment
from core.study import study
import matplotlib.pyplot as plt

def get_iqr_outliers(df: pd.DataFrame, column_name: str):
    values = df[column_name].to_numpy().astype(float)

    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    lower_outliers = df[df[column_name] < lower_bound]
    upper_outliers = df[df[column_name] > upper_bound]

    return upper_outliers

def get_modz_outliers(df: pd.DataFrame, column_name: str):
    values = df[column_name].to_numpy().astype(float)

    median = np.median(values)
    mad = np.median(np.abs(values - median))

    modz = 0.6745 * (values - median) / mad

    threshold = 3.5
    outliers = df[np.abs(modz) > threshold]

    return outliers

def group_outliers(df_outliers: pd.DataFrame):
    frames = df_outliers['frame'].to_numpy()
    frames = np.sort(frames)
    grouping_threshold = 1
    groups = np.split(frames, np.where(np.diff(frames) > grouping_threshold)[0] + 1)

    return groups

def get_trajectories(data, experiment_id):
    df = data[data['experiment_id'] == experiment_id]

    df_outliers = get_modz_outliers(df, 'acceleration_fish')
    groups = group_outliers(df_outliers)

    print(groups)

    trajectories = []
    frame_buffer = 25
    for group in groups:
        start_frame = group[0] - frame_buffer
        end_frame = group[-1] + frame_buffer
        trajectory = df[(df['frame'] >= start_frame) & (df['frame'] <= end_frame)]
        trajectories.append(trajectory)

        plot_trajectory(trajectory)

    return trajectories


def plot_trajectory(trajectory):
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    min_frame = trajectory['frame'].min()
    max_frame = trajectory['frame'].max()

    ax = axes[1]

    # plot distance
    ax.plot(trajectory['frame'], trajectory['distance'], label='distance', color='tab:blue')
    ax.set_ylim(0, 40)
    # plot speed_robot
    ax2 = ax.twinx()
    ax2.plot(trajectory['frame'], trajectory['speed_robot'], label='speed_robot', color='tab:orange')
    ax2.set_ylim(-2.5, 32.5)
    # plot cos_apv
    ax3 = ax.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    ax3.plot(trajectory['frame'], trajectory['cos_apv'], label='cos_apv', color='tab:green')
    ax3.set_ylim(-1.1, 1.1)
    ax3.hlines(0, min_frame, max_frame, color='tab:green', linestyle='--')
    # plot speed_fish
    ax4 = ax.twinx()
    ax4.spines['right'].set_position(('outward', 120))
    ax4.plot(trajectory['frame'], trajectory['speed_fish'], label='speed_fish', color='tab:red')
    ax4.set_ylim(0, 30)
    # plot acceleration_fish
    ax5 = ax.twinx()
    ax5.spines['right'].set_position(('outward', 180))
    ax5.plot(trajectory['frame'], trajectory['acceleration_fish'], label='acceleration_fish', color='tab:purple')
    ax5.set_ylim(-40, 40)

    # ax.set_title(f"Experiment {experiment_id} - Outliers in Fish Speed")
    label_fontsize = 26
    ax.set_xlabel("Frame", fontsize=label_fontsize)
    ax.set_ylabel("Distance (cm)", color='tab:blue', fontsize=label_fontsize)
    ax2.set_ylabel("Speed Robot (cm/s)", color='tab:orange', fontsize=label_fontsize)
    ax3.set_ylabel("cos(apv)", color='tab:green', fontsize=label_fontsize)
    ax4.set_ylabel("Speed Fish (cm/s)", color='tab:red', fontsize=label_fontsize)
    ax5.set_ylabel("Acceleration Fish (cm/s^2)", color='tab:purple', fontsize=label_fontsize)

    # color yticks
    ax.tick_params(axis='y', colors='tab:blue')
    ax2.tick_params(axis='y', colors='tab:orange')
    ax3.tick_params(axis='y', colors='tab:green')
    ax4.tick_params(axis='y', colors='tab:red')
    ax5.tick_params(axis='y', colors='tab:purple')

    # legend
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    lines4, labels4 = ax4.get_legend_handles_labels()
    lines5, labels5 = ax5.get_legend_handles_labels()
    ax.legend(lines + lines2 + lines3 + lines4 + lines5, labels + labels2 + labels3 + labels4 + labels5, loc='upper left')

    ax = axes[0]

    # plot x,y of robot and fish in 2D
    cmap = plt.cm.get_cmap('Blues')
    n_frames = trajectory.shape[0]
    alphas = np.linspace(0, 1, n_frames)
    for i in range(n_frames - 1):
        ax.plot(trajectory['x_robot'].iloc[i:i+2], trajectory['y_robot'].iloc[i:i+2], color=cmap(alphas[i]), lw=4)

    cmap = plt.cm.get_cmap('Greens')
    for i in range(n_frames - 1):
        ax.plot(trajectory['x_fish'].iloc[i:i+2], trajectory['y_fish'].iloc[i:i+2], color=cmap(alphas[i]), lw=4)

    ax.set_xlabel("X (cm)", fontsize=label_fontsize)
    ax.set_ylabel("Y (cm)", fontsize=label_fontsize)

    # add blue to legend
    ax.plot([], [], color='blue', label='Robot Trajectory')
    ax.plot([], [], color='green', label='Fish Trajectory')
    ax.legend()

    # filename
    filename = f"1_qt_F={min_frame}-{max_frame}.png"
    filefolder = f"{study.plot_output_folder}/1"
    if not os.path.exists(filefolder):
        os.makedirs(filefolder)
    filepath = f"{filefolder}/{filename}"
    fig.savefig(filepath)

if __name__ == "__main__":
    data = pd.read_csv('core/study_data.csv')
    get_trajectories(data, 0)