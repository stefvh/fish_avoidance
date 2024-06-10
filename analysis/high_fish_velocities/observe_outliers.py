import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from core.experiment import Experiment
from core.study import study

# For every experiment separately:
# Find the outliers in measured speed_fish
# For those outliers, show (over a given number of frames) the preceding:
# - speed_robot
# - distance
# - cos_apv

def plot_group(experiment_id,
               df,
               frame_group: list):
    fig, ax = plt.subplots(1, 1, figsize=(28, 16))

    max_frame = np.max(frame_group)
    min_frame = np.min(frame_group)

    up_frame = max_frame + 10
    down_frame = min_frame - 10

    df_group = df[(df['frame'] >= down_frame) & (df['frame'] <= up_frame)]

    # plot distance
    ax.plot(df_group['frame'], df_group['distance'], label='distance', color='tab:blue')
    ax.set_ylim(0, 40)
    ax.vlines(min_frame, 0, 40, color='tab:blue', linestyle='--')
    ax.vlines(max_frame, 0, 40, color='tab:blue', linestyle='--')
    # plot speed_robot
    ax2 = ax.twinx()
    ax2.plot(df_group['frame'], df_group['speed_robot'], label='speed_robot', color='tab:orange')
    ax2.set_ylim(-2.5, 32.5)
    # plot cos_apv
    ax3 = ax.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    ax3.plot(df_group['frame'], df_group['cos_apv'], label='cos_apv', color='tab:green')
    ax3.set_ylim(-1.1, 1.1)
    ax3.hlines(0, down_frame, up_frame, color='tab:green', linestyle='--')
    # plot speed_fish
    ax4 = ax.twinx()
    ax4.spines['right'].set_position(('outward', 120))
    ax4.plot(df_group['frame'], df_group['speed_fish'], label='speed_fish', color='tab:red')
    ax4.set_ylim(0, 30)
    # plot acceleration_fish
    ax5 = ax.twinx()
    ax5.spines['right'].set_position(('outward', 180))
    ax5.plot(df_group['frame'], df_group['acceleration_fish'], label='acceleration_fish', color='tab:purple')
    ax5.set_ylim(-40, 40)

    ax.set_title(f"Experiment {experiment_id} - Outliers in Fish Speed")
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

    filename = f"E={experiment_id}_outliers_fish_speed_group={min_frame}-{max_frame}.png"
    filefolder = f"{study.plot_output_folder}/outliers"
    if not os.path.exists(filefolder):
        os.makedirs(filefolder)
    filepath = f"{filefolder}/{filename}"
    fig.savefig(filepath)

def analyse_experiment(experiment: Experiment):
    df = experiment.relative_motion_quantities.df

    fig, axes = plt.subplots(3, 2, figsize=(20, 16))

    #
    rf = df[
        (df['cos_apv'] < 0) &
        (df['speed_fish'] > 15) &
        (df['speed_robot'] > 20) &
        (df['distance'] < 10)
    ][["frame", "speed_robot", "distance", "speed_fish"]]
    print(rf)

    # [0, 0]
    # Hist and Violinplot of fish_speed
    fish_speeds = df["speed_fish"]
    np_fish_speeds = fish_speeds.dropna().to_numpy().astype(float)
    max_speed = np.max(np_fish_speeds)

    ax = axes[0, 0]
    hist = ax.hist(fish_speeds, bins=50, density=True, color="tab:purple")
    ax.set_title("Fish Speeds")
    ax.set_xlabel("Fish Speed (cm/s)")
    ax.set_ylabel("Density")
    ax.set_xlim(0, max_speed)
    hist_max = hist[0].max()
    ylim = hist_max + 0.1 * hist_max
    ax.set_ylim(0, ylim)

    axin = ax.inset_axes([0.65, 0.4, 0.3, 0.55])
    violin = axin.violinplot(np_fish_speeds, 
                             showextrema=True, 
                             showmedians=True,
                             quantiles=[0.25, 0.75],)
    for partname in ("cquantiles",):
        vp = violin[partname]
        vp.set_edgecolor("tab:green")
        vp.set_linewidth(1.)
        vp.set_linestyle("-")
    for partname in ("cmedians",):
        vp = violin[partname]
        vp.set_linewidth(0.75)
        vp.set_linestyle("--")
    axin.set_xticks([])
    axin.set_ylabel("Density")
    # axin.set_xlim(0, 32)
    # axin.set_ylim(0, 1)

    q1 = np.percentile(np_fish_speeds, 25)
    q3 = np.percentile(np_fish_speeds, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    ax.vlines(upper_bound, 0, ylim, color="tab:red", linestyle="--")

    iqr_high_outliers = df[df['speed_fish'] > upper_bound]
    high_outlier_frames = iqr_high_outliers['frame']

    # create groups of frames that are close to each other
    high_outlier_frames = high_outlier_frames.to_numpy()
    frames = np.sort(high_outlier_frames)
    groups = np.split(frames, np.where(np.diff(frames) > 1)[0] + 1)

    # for group in groups:
    #     plot_group(experiment.id, df, group)

    median_y = np.median(np_fish_speeds)
    MAD = np.median(np.abs(np_fish_speeds - median_y))
    x_buffer = np.linspace(0, max_speed, 100)
    modified_z_scores = 0.6745 * (x_buffer - median_y) / MAD

    # plot minimum value of x_buffer where np.abs(modified_z_scores) > threshold
    threshold = 3.5
    min_x = x_buffer[np.where(np.abs(modified_z_scores) > threshold)[0][0]]
    ax.vlines(min_x, 0, ylim, color="tab:orange", linestyle="--")
    
    # [0, 1]
    # Hist and Violinplot of acceleration_fish
    fish_accelerations = df["acceleration_fish"]
    np_fish_accelerations = fish_accelerations.dropna().to_numpy().astype(float)
    max_acceleration = np.max(np_fish_accelerations)

    ax = axes[0, 1]
    hist = ax.hist(np_fish_accelerations, bins=50, density=True, color="tab:orange")
    ax.set_title("Fish Accelerations")
    ax.set_xlabel("Fish Acceleration (cm/s^2)")
    ax.set_ylabel("Density")
    hist_max = hist[0].max()
    ylim = hist_max + 0.1 * hist_max
    ax.set_ylim(0, ylim)

    axin = ax.inset_axes([0.65, 0.4, 0.3, 0.55])
    violin = axin.violinplot(np_fish_accelerations, 
                             showextrema=True, 
                             showmedians=True,
                             quantiles=[0.25, 0.75],)
    for partname in ("cquantiles",):
        vp = violin[partname]
        vp.set_edgecolor("tab:green")
        vp.set_linewidth(1.)
        vp.set_linestyle("-")
    for partname in ("cmedians",):
        vp = violin[partname]
        vp.set_linewidth(0.75)
        vp.set_linestyle("--")
    axin.set_xticks([])
    axin.set_ylabel("Density")
    
    filename = f"E={experiment.id}_outliers_fish_speed.pdf"
    filepath = f"{study.plot_output_folder}/{filename}"
    fig.savefig(filepath)

    plt.close(fig)


def run():
    experiments = study.list_experiments

    for experiment in experiments:
        if experiment.id == 0:
            analyse_experiment(experiment)

if __name__ == "__main__":
    run()