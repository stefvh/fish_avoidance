from typing import List
import numpy as np
import pandas as pd


def sort_trials_by_distance(results: pd.DataFrame) -> List[int]:
    trial_ids = results.trial_id.unique()
    distance_list = [results[results.trial_id == x].robot_distance.min() for x in trial_ids]
    all_trial_ids = trial_ids
    trial_ids = [trial_ids[i] for i in range(len(trial_ids)) if not np.isnan(distance_list[i])]
    invalid_trials = [x for x in all_trial_ids if x not in trial_ids]
    if len(invalid_trials) > 0:
        print(invalid_trials)
    distance_list = [distance_list[i] for i in range(len(trial_ids)) if not np.isnan(distance_list[i])]
    trial_ids = [x for _, x in sorted(zip(distance_list, trial_ids))]
    return trial_ids