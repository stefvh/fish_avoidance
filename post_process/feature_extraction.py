import numpy as np
import pandas as pd


def extract_approach_window_frame_by_state(trial_results: pd.DataFrame):
    df = trial_results.copy()
    df.reset_index(inplace=True)

    # flags
    b_approach_start = False
    # frames
    start_frame_approach = None
    end_start_approach = None
    # loop
    for frame_number in range(len(df)):
        robot_state = df["state"].iloc[frame_number]
        if not b_approach_start and "APPROACH" in robot_state:
            b_approach_start = True
            start_frame_approach = frame_number
            continue
        if b_approach_start and "APPROACH" not in robot_state:
            end_start_approach = frame_number
            break
    
    return (start_frame_approach, end_start_approach)


def extract_approach_window_frame_by_follow_metric(trial_results: pd.DataFrame):
    df = trial_results.copy()
    df.reset_index(inplace=True)

    argmin_d = df["robot_distance"].idxmin()

    # window start
    idx = 10 # arbitrary
    window_start = None

    flags = df["robot_fish_follow_metric"] > 0.05
    flags = flags.to_list()
    c = 0
    c_flags = []
    for x in reversed(flags):
        if x == False:
            c = 0
        else:
            c += 1
        c_flags.append(c)
    c_flags.reverse()
    c_flags = np.array(c_flags)

    while df["robot_fish_follow_metric"].iloc[idx] < 0.05 or c_flags[idx] < 20:
        idx += 1
        window_start = idx
        if idx == argmin_d:
            break

    # window end
    # idx = window_start
    # window_end = None

    flags = df["robot_fish_follow_metric"] < 0.05
    flags = flags.to_list()
    c = 0
    c_flags = []
    for x in reversed(flags):
        if x == False:
            c = 0
        else:
            c += 1
        c_flags.append(c)
    c_flags.reverse()
    c_flags = np.array(c_flags)

    # while df["robot_fish_follow_metric"].iloc[idx] > 0.05 or c_flags[idx] > 5:
    #     idx += 1
    #     window_end = idx
    #     if idx == len(df) - 1:
    #         break
    
    idx = argmin_d
    window_end = None

    while df["robot_fish_follow_metric"].iloc[idx] > 0.05 or c_flags[idx] < 10:
        idx += 1
        window_end = idx
        if idx >= len(df):
            break
    if window_end is None:
        idx = argmin_d
        while df["robot_fish_follow_metric"].iloc[idx] < 0.05:
            idx -= 1
            window_end = idx
            if idx >= len(df):
                break

    # while df["robot_fish_follow_metric"].iloc[idx] < 0.05:
    #     idx -= 1
    #     window_end = idx
    #     if idx < 0:
    #         break
    # if window_end is None:
    #     idx = argmin_d
    #     while df["robot_fish_follow_metric"].iloc[idx] > 0.05:
    #         idx += 1
    #         window_end = idx
    #         if idx >= len(df):
    #             break

    return (window_start, window_end)
