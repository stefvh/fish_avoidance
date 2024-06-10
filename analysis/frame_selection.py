import numpy as np
import pandas as pd
from core.experiment import Experiment

def get_filtered_frames(experiment: Experiment, trials_filter: str):
    if trials_filter == "all":
        frame_numbers = range(
            experiment.trials.iloc[0]["start_frame"],
            experiment.trials.iloc[-1]["end_frame"],
        )
    elif trials_filter == "peak_repulsion":
        filepath = f"plotting/observation/peak_repulsion_ids/{experiment.experiment_datetime}.csv"
        cv_input = pd.read_csv(filepath)
        trial_ids = list(cv_input["trial_id"])

        frame_numbers = []
        for trial_id in trial_ids:
            trial = experiment.trials[experiment.trials["trial_id"] == trial_id].iloc[0]
            frame_numbers += list(range(trial["start_frame"], trial["end_frame"]))
    else:
        raise ValueError(f"Invalid trials_filter: {trials_filter}")
    return frame_numbers


def get_distance_dip_frames(results, frame_window):
    # For each trial_id in results, find the highest frame number where state in "APPROACHING*"
    dip_frames = []
    dip_values = []

    for trial_id in results["trial_id"].unique():
        trial_results = results[results["trial_id"] == trial_id]
        dip_distance = trial_results["robot_distance"].min()
        dip_idx = trial_results["robot_distance"].idxmin()
        dip_frame = trial_results["frame_number"][dip_idx]

        t_frames = [dip_frame]

        threshold = 2.0
        # add frames going back from dip_frame until the robot_distance is above or below threshold
        for i in range(1, 100):
            if dip_idx - i < 0 or dip_idx + i >= len(trial_results):
                break

            dist = trial_results["robot_distance"][dip_idx - i]
            if dist > dip_distance + threshold or dist < dip_distance - threshold:
                t_frames.append(trial_results["frame_number"][dip_idx - i])
                break
        # add frames going forward from dip_frame until the robot_distance is above or below threshold
        for i in range(1, 100):
            if dip_idx - i < 0 or dip_idx + i >= len(trial_results):
                break

            dist = trial_results["robot_distance"][dip_idx + i]
            if dist > dip_distance + threshold or dist < dip_distance - threshold:
                t_frames.append(trial_results["frame_number"][dip_idx + i])
                break

        frame_window = frame_window   
        t_min = min(t_frames)
        t_max = max(t_frames)

        dip_frames.append(list(range(t_min - frame_window, t_max + frame_window)))
        dip_values.append(dip_distance)

    return dip_frames, dip_values