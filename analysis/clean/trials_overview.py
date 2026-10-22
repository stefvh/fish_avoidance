import os
import numpy as np
import pandas as pd

from core.experiment import Experiment
from core.study import study
from plotting.publishing.subset_trials import get_trials_by_data
from analysis.full.by_trials import plot_trials_per_exp


if __name__ == "__main__":
    df = pd.read_csv("analysis/clean/measurements.csv")

    df['turn_test'] = df['turn_heading_fish']
    speed_threshold = 0.5  
    df.loc[df['speed_fish'] < speed_threshold, 'turn_test'] = 0

    exp_trials = get_trials_by_data(df, study.list_experiments)

    output_folder = "analysis/clean/figs/trials"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    columns = [
        "distance",
        "speed_fish",
        "v_app",
        "v_avoid",
        "turn_heading_fish",
    ]

    plot_trials_per_exp(exp_trials, 
                        output_folder=output_folder,
                        columns=columns)