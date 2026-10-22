import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from core.experiment import Experiment
from core.study import study

from scipy.stats import wilcoxon, zscore, chi2_contingency

def analyse_vf_by_distance(experiment: Experiment):
    df = experiment.relative_motion_quantities.df.copy()
    fish_speeds = df["speed_fish"]
    np_fish_speeds = fish_speeds.dropna().to_numpy().astype(float)

    distance_thresholds = [
        2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30
    ]

    # categorize fish_speed
    # iqr
    q1 = np.percentile(np_fish_speeds, 25)
    q3 = np.percentile(np_fish_speeds, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    # mod z-score
    median = np.median(np_fish_speeds)
    mad = np.median(np.abs(np_fish_speeds - median))
    # categorize
    df["speed_fish_category_iqr"] = df["speed_fish"].apply(lambda x: 1 if x > upper_bound else 0)
    df["speed_fish_category_mod_z"] = df["speed_fish"].apply(lambda x: 1 if (x - median) / mad > 3.5 else 0)

    for dist_th in distance_thresholds:
        group_below = df[df["distance"] < dist_th]
        group_above = df[df["distance"] >= dist_th]
        # iqr groups
        low_velocity_below = np.sum(group_below["speed_fish_category_iqr"] == 0)
        high_velocity_below = np.sum(group_below["speed_fish_category_iqr"] == 1)
        low_velocity_above = np.sum(group_above["speed_fish_category_iqr"] == 0)
        high_velocity_above = np.sum(group_above["speed_fish_category_iqr"] == 1)

        # Create contingency table
        contingency_table = np.array([
            [low_velocity_below, high_velocity_below],
            [low_velocity_above, high_velocity_above]
        ])

        # Perform the chi-square test
        chi2, p, _, _ = chi2_contingency(contingency_table)

        # Print the results
        print("------------------------------------")
        print(f"Distance Threshold: {dist_th}")
        print(f'Chi2 Statistic: {chi2}, p-value: {p}')
        if p < 0.05:
            print("Reject the null hypothesis: There is a significant association between distance and velocity categories.")
        else:
            print("Fail to reject the null hypothesis: There is no significant association between distance and velocity categories.")

        # # Create two groups
        # group1 = df[df["distance"] < dist_th]
        # group2 = df[df["distance"] >= dist_th]

        # # Get the velocities
        # vf_below = group1["speed_fish"].dropna().to_numpy().astype(float)
        # vf_above = group2["speed_fish"].dropna().to_numpy().astype(float)

        # # Perform the Wilcoxon signed-rank test
        # stat, p_value = wilcoxon(vf_below, vf_above)

        # # Print the results
        # print(f'Statistic: {stat}, p-value: {p_value}')
        # if p_value < 0.05:
        #     print("Reject the null hypothesis: There is a significant difference between the velocities.")
        # else:
        #     print("Fail to reject the null hypothesis: There is no significant difference between the velocities.")

def run():
    experiments = study.list_experiments

    # list_vars = [
    #     ["distance"],
    #     ["distance", "speed_robot"],
    #     ["distance", "speed_robot", "cos_apv"],
    # ]

    # for experiment in experiments:
    #     for vars in list_vars:
    #         analyse(experiment, vars)

    for experiment in experiments:
        if experiment.id == 0:
            analyse_vf_by_distance(experiment)

if __name__ == "__main__":
    run()