from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels
import statsmodels.genmod
import statsmodels.genmod.families
import statsmodels.stats
import statsmodels.stats.anova

from core.experiment import Experiment
from core.study import study

from analysis.clean.sub5.speed_distance import get_speed_distance_by_arb
from analysis.one.wrapper_fixed import run_plm_model
from analysis.one.wrapper_fixed_gen import run_pglm_model
from core.experiment import Experiment
from core.study import study

import networkx as nx

import utils.plotter as plotter

import statsmodels.formula.api as smf

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


def get_avg_time(exp_trials):
    avg_times_per_exp = []
    for exp in exp_trials:
        avg_times = []
        for trial in exp:
            # compute end_frame - start_frame and divide by 25 to get seconds
            trial_duration = (trial["end_frame"] - trial["start_frame"]) / 25.0
            avg_times.append(trial_duration)
        avg_times_per_exp.append(avg_times)
    return avg_times_per_exp


def henk():
    exp_durations = []
    for exp in study.list_experiments:
        trials = exp.trials
        # trials is pd.DataFrame
        # loop over it and compute duration of trial by start_frame and end_frame difference divided by 25 hz
        durations = []
        for i, trial in trials.iterrows():
            trial_duration = (trial["end_frame"] - trial["start_frame"]) / 25.0
            durations.append(trial_duration)
        exp_durations.append((np.mean(durations), np.std(durations)))
    print("Experiment durations (mean, std):")
    for i, (mean, std) in enumerate(exp_durations):
        print(f"Experiment {i}: Mean: {mean:.2f}, Std: {std:.2f}")
    # mean of means
    means = [mean for mean, _ in exp_durations]
    stds = [std for _, std in exp_durations]
    print("Mean of means:", np.mean(means))
    print("Std of means:", np.std(means))

if __name__ == "__main__":
    # exp_trials = get_trials()
    # # print number of trials per experiment
    # for i, trials in enumerate(exp_trials):
    #     print(f"Experiment {i}: {len(trials)} trials")
    # avg_times_per_exp = get_avg_time(exp_trials)
    # print("Average times per experiment:")
    # means = []
    # stds = []
    # for i, avg_times in enumerate(avg_times_per_exp):
    #     avg = np.mean(avg_times)
    #     means.append(avg)
    #     std = np.std(avg_times)
    #     stds.append(std)
    #     print(f"Experiment {i}: {avg_times}, Mean: {avg:.2f}, Std: {std:.2f}")
    # print("Mean of means:", np.mean(means))
    # print("Std of means:", np.std(means))
    henk()
