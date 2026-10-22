
import pandas as pd

from core.experiment import Experiment
from core.study import study


def get_trials():
    data = study.data
    experiments = study.list_experiments
    trials = get_trials_by_data(data, experiments)
    return trials

def get_trials_by_data(data, experiments):
    remove_exp_trials = [
        [10, 21],
        [2, 11, 12, 13, 15],
        [8],
        [],
        []
    ]

    exp_trials = []

    for exp in experiments:
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
                # print("Missing frames in trial data")
                # print(f"Experiment ID: {exp_id}, Trial ID: {i}")
                # print(
                #     f"Expected: {row['end_frame'] - row['start_frame'] + 1}, Actual: {trial_data.shape[0]}"
                # )
                pass
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

    return exp_trials