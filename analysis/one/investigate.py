import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn

from core.experiment import Experiment
from core.study import study

from analysis.one.space import plot_trajectory_descriptives

def plot_trajectory_window():
    pass

if __name__ == "__main__":
    input_folder = study.plot_output_folder + "/peak_moments"

    data = pd.read_csv(input_folder + "/regression_data.csv")

    experiment_id = 1
    data = data[data["experiment_id"] == experiment_id]

    # print rows where distance is < 10 and acceleration_fish is < 10
    data = data[(data["distance"] < 10) & (data["acceleration_fish"] < 10)]

    full_data = study.data

    for i, row in data.iterrows():
        fig, axes = plt.subplots(5, 1)
        fig.set_figheight(25)
        fig.set_figwidth(15)

        t_p = row["predictor_time"]
        window = 200

        trajectory = full_data[(full_data["experiment_id"] == experiment_id) &
                       (full_data["frame"] >= t_p - window) &
                       (full_data["frame"] <= t_p + window)]

        plot_trajectory_descriptives(axes=axes, 
                                     trajectory=trajectory,
                                     prominence=18.5)
    
        fig.savefig(input_folder + f"/T_E={experiment_id}_F={t_p}.pdf")
