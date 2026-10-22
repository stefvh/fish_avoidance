import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from analysis.clean.avoidance_mechanisms import get_data
from core.experiment import Experiment
from core.study import study
from plotting.publishing.subset_trials import get_trials_by_data


def transform(data: pd.DataFrame):
    distances = data["distance"].to_numpy()
    ef = np.argmin(distances)
    td = 10

    start_frame = 0
    end_frame = ef - td

    angles = data["angle_rf"].to_numpy()
    start_angle = angles[start_frame]
    end_angle = angles[end_frame]

    return start_angle, end_angle


if __name__ == "__main__":
    df = get_data()
    exp_trials = get_trials_by_data(df, study.list_experiments)

    x, y = [], []

    for exp_id, trials in enumerate(exp_trials):
        for trial in trials:
            data = trial["data"]
            start_angle, end_angle = transform(data)
            x.append(start_angle)
            y.append(end_angle)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.scatter(x, y)
    # labels
    ax.set_xlabel("Start angle")
    ax.set_ylabel("End angle")

    ax.axvline(np.pi, color="black", linestyle="--")
    ax.axhline(np.pi, color="black", linestyle="--")

    # do a y = x line
    ax.plot([0, 2 * np.pi], [0, 2 * np.pi], color="black", linestyle="-")

    output_folder = "analysis/clean/figs/avoidance_trajectories"
    os.makedirs(output_folder, exist_ok=True)
    fig.savefig(os.path.join(output_folder, f"trial_start-end_angles.pdf"))
    plt.close()
