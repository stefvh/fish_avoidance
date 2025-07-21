import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from analysis.full.get_peak_data import get_peaks_data
from core.experiment import Experiment
from core.study import study



def plot_measurements(df: pd.DataFrame, stp: pd.DataFrame):
    fig, axes = plt.subplots(5, 2)
    fig.set_figheight(20)
    fig.set_figwidth(40)

    predictors = ["distance", "speed_robot", "cos_apv", "cos_rev", "min_wall_distance_fish"]
    responses = ["speed_fish", "acceleration_fish", "apv", "turn_fish", "rev"]

    # colors, 10 of them
    colors = [
        'tab:blue',
        'tab:orange',
        'tab:green',
        'tab:red',
        'tab:purple',
        'tab:brown',
        'tab:pink',
        'tab:gray',
        'tab:olive',
        'tab:cyan',
    ]
    label_fontsize = 26
    linewidth = 2
    for ax in axes.flatten():
        ax.yaxis.label.set_size(label_fontsize)

    for i, predictor in enumerate(predictors):
        ax = axes[i, 0]
        ax.plot(df['frame'], 
                df[predictor],
                color=colors[i],
                linewidth=linewidth)
        ax.set_ylabel(predictor)
        ax.set_xlabel("Frame")

    for i, response in enumerate(responses):
        ax = axes[i, 1]
        ax.plot(df['frame'], 
                df[response],
                color=colors[i+5],
                linewidth=linewidth)
        ax.set_ylabel(response)
        ax.set_xlabel("Frame")
    
    # Mark predictor times
    begin_frame = df['frame'].min()
    end_frame = df['frame'].max()
    p_times = stp[(stp['predictor_time'] >= begin_frame) & (stp['predictor_time'] <= end_frame)]
    for i, predictor in enumerate(predictors):
        ax = axes[i, 0]
        for _, row in p_times.iterrows():
            ax.axvline(x=row['predictor_time'], color='black', linestyle='dashed', linewidth=linewidth)
    
    # Mark response_vel_time points
    p_times = stp[(stp['response_vel_time'] >= begin_frame) & (stp['response_vel_time'] <= end_frame)]
    _axes = [axes[0, 1], axes[4, 1]]
    for _, row in p_times.iterrows():
        for ax in _axes:
            ax.axvline(x=row['response_vel_time'], color='black', linestyle='dashed', linewidth=linewidth)
    
    # Mark response_acc_time points
    p_times = stp[(stp['response_acc_time'] >= begin_frame) & (stp['response_acc_time'] <= end_frame)]
    ax = axes[1, 1]
    for _, row in p_times.iterrows():
        ax.axvline(x=row['response_acc_time'], color='black', linestyle='dashed', linewidth=linewidth)
    
    # Mark frames of df where frame[i+1] - frame[i] > 1
    for i in range(len(df) - 1):
        if df.iloc[i + 1]['frame'] - df.iloc[i]['frame'] > 1:
            for ax in axes.flatten():
                ax.axvline(x=df.iloc[i]['frame'], color='red', linestyle='dashed', linewidth=linewidth)
                ax.axvline(x=df.iloc[i+1]['frame'], color='red', linestyle='dashed', linewidth=linewidth)
    
    return fig, axes

def plot_experiment(experiment_id: int, data: pd.DataFrame, sample_time_points: pd.DataFrame):
    filefolder = f"{study.plot_output_folder}/meas/E={experiment_id}"
    if not os.path.exists(filefolder):
        os.makedirs(filefolder)

    df = data[data['experiment_id'] == experiment_id]
    stp = sample_time_points[sample_time_points['experiment_id'] == experiment_id]

    begin_frame = df['frame'].min()
    end_frame = df['frame'].max()
    n_frames_per_plot = 500

    for i in range(begin_frame, end_frame, n_frames_per_plot):
        sub_df = df[(df['frame'] >= i) & (df['frame'] < i + n_frames_per_plot)]
        fig, axes = plot_measurements(sub_df, stp)
        fig.suptitle(f"Experiment {experiment_id}", fontsize=30)
        fig.tight_layout()
        fig.savefig(f"{filefolder}/meas_{i}.png")

    return

if __name__ == "__main__":
    experiment_ids = [2]
    data = study.data
    sample_time_points, _ = get_peaks_data(
        func_bound=None,
        func_multiplier=0,
        acc_max_height=0,
        vel_max_height=0,
        acc_prominence=17.5,
        do_group=False,
    )

    for experiment_id in experiment_ids:
        print(f"Plotting measurements of experiment {experiment_id}")
        plot_experiment(experiment_id, data, sample_time_points)
