import os
import numpy as np
import pandas as pd

from core.experiment import Experiment
from core.study import study

import matplotlib.pyplot as plt

from analysis.clean.avoidance_mechanisms import get_data
from analysis.clean.reactions import clean_data, compute_end_approach
from plotting.publishing.subset_trials import get_trials_by_data

import utils.plotter



def extract_trials(clean=True):
    df = get_data()
    if clean:
        df = clean_data(df)
    exp_trials = get_trials_by_data(df, study.list_experiments)
    
    for trials in exp_trials:
        for trial in trials:
            trial["end_approach_index"] = compute_end_approach(trial)
    
    return exp_trials

if __name__ == "__main__":
    exp_trials = extract_trials()

    robot_speeds = []

    for exp_idx, trials in enumerate(exp_trials):
        for trial in trials:
            end_idx = trial["end_approach_index"]
            end_data = trial["data"].iloc[:end_idx]
            last_frames = end_data.iloc[-10:]
            speed_robot = last_frames['speed_robot'].values
            robot_speeds.extend(speed_robot)

    # Convert the list of robot speeds to a pandas series or array
    robot_speeds = np.array(robot_speeds)

    # # Optionally, analyze or visualize the speed distribution
    # print(f"Robot Speed Distribution: {robot_speeds}")

    # Assuming robot_speeds is the list or array containing the robot speeds
    plt.figure(figsize=(8, 6))
    plt.hist(robot_speeds, bins=30, edgecolor='black', alpha=0.7, color="tab:purple")
    plt.xlabel(r'approaching robot speed $v^R$ (cm/s)', fontsize=20)
    plt.ylabel(r'density', fontsize=20)
    # set ticks fontsize
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    # plt.grid(True)
    # plt.show()
    plt.savefig('robot_speeds.pdf')

