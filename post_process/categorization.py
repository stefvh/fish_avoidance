import numpy as np
import pandas as pd

import post_process.feature_extraction as fe


def categorize_approach_velocity(trial_results: pd.DataFrame,
                                 window_frame: tuple,):
    start_frame, end_frame = window_frame

    df = trial_results.copy()
    df.reset_index(inplace=True)

    values = []
    bar = 1.5
    idx = end_frame
    cont = True
    while cont:
        if idx <= start_frame:
            cont = False
        window = df["robot_velocity_batch"].iloc[idx - 5:idx]
        if np.std(window) < bar:
            values.append(df["robot_velocity_batch"].iloc[idx])
            idx -= 1
        else:
            cont = False
    
    if values == []:
        values.append(df["robot_velocity_batch"].iloc[end_frame])
    
    mean_velocity = np.mean(values)
    std_velocity = np.std(values)
    duration = len(values)

    if duration >= min(1, end_frame - start_frame) and mean_velocity > 15:
        return "STEADY_APPROACH"
    else:
        return "VARYING_APPROACH"


def categorize_stop_distance(trial_results: pd.DataFrame):
    min_d = trial_results["robot_distance"].min()

    if min_d < 5:
        return "CLOSE_STOP"
    else:
        return "PREMATURE_STOP"


def categorize_stop_location(trial_results: pd.DataFrame,
                             window_frame: tuple,):
    start_frame, end_frame = window_frame

    df = trial_results.copy()
    df.reset_index(inplace=True)

    betas = trial_results["beta_clockwise_batch"].iloc[end_frame - 5: end_frame]
    mean_beta = np.mean(betas)

    if abs(mean_beta) < 0.25:
        return "INTERSECT_STOP"
    else:
        return "SIDEWAY_STOP"


def categorize_post_behavior(trial_results: pd.DataFrame,
                             window_frame: tuple,):
    _, end_frame = window_frame

    df = trial_results.copy()
    df.reset_index(inplace=True)

    df_window = df[(df["frame_number"] >= end_frame) & (df["frame_number"] <= end_frame + 5*25)]
    wall_avoidance = df_window[df_window["state"].str.contains("WALL_AVOIDANCE")]
    high_velocity = df_window[df_window["robot_velocity_batch"] > 5]

    if len(wall_avoidance) > 5 and len(high_velocity) > 5:
        return "DISTURBED_POST"
    else:
        return "NORMAL_POST"


def categorize_wall_avoidance(trial_results: pd.DataFrame,
                              window_frame: tuple,):
    start_frame, end_frame = window_frame
    end_frame += 5 * 25

    df = trial_results.copy()
    df.reset_index(inplace=True)

    df_window = df[(df["frame_number"] >= start_frame) & (df["frame_number"] <= end_frame)]
    wall_avoidance = df_window[df_window["state"].str.contains("WALL_AVOIDANCE")]
    
    if wall_avoidance.empty:
        return "NO_WALL_AVOIDANCE"
    else:
        return "WALL_AVOIDANCE"
