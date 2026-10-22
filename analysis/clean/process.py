
import os
import numpy as np
import pandas as pd
from core.experiment import Experiment, get_experiment_by_id
from pre_process.environment import Environment
from scipy.ndimage import gaussian_filter1d

FREQUENCY = 25
EXPERIMENT_IDS = [0, 1, 2, 3, 4]

def get_measurements():
    if os.path.exists("analysis/clean/measurements.csv"):
        return pd.read_csv("analysis/clean/measurements.csv")
    else:
        return process_experiments()

def process_experiments():
    experiments = [get_experiment_by_id(exp_id) for exp_id in EXPERIMENT_IDS]

    dfs = []
    for exp in experiments:
        df = process_experiment(exp)
        dfs.append(df)
    df = pd.concat(dfs)

    df.to_csv("analysis/clean/measurements.csv", index=False)

    return df

def process_experiment(exp: Experiment):
    # Get the tracks
    robot_poses, fish_poses = exp.get_tracks_limited_correction()
    robot_poses, fish_poses = np.copy(robot_poses), np.copy(fish_poses)
    # Pre-process
    df_robot = pre_process(exp, robot_poses)
    df_fish = pre_process(exp, fish_poses)
    # Process the dataframes
    df_robot = process_df(exp, df_robot)
    df_fish = process_df(exp, df_fish)
    # Process both dataframes together
    df = combined_processing(exp, df_robot, df_fish)
    # Mark
    df['experiment_id'] = exp.id

    return df

def pre_process(exp, poses):
    df = np_to_df(poses)
    df = filter_df(df, exp.valid_frame_start, exp.valid_frame_end)
    return df

def np_to_df(np_array):
    df = pd.DataFrame(np_array, columns=["x", "y", "z"])
    df["frame"] = range(len(df))
    return df

def filter_df(df, valid_frame_start, valid_frame_end):
    return df.loc[(df["frame"] > valid_frame_start) & (df["frame"] < valid_frame_end)]

def process_df(exp: Experiment, df: pd.DataFrame):
    new_dfs = []

    time_step = exp.period
    smoothing_sigma = exp.rolling_window_size / 2
    environment = Environment(exp.corners)

    segments = segmentize_df(df, ["x", "y"])

    for segment_df in segments:
        if not segment_df.empty:
            new_df = process_df_segment(
                segment_df,
                time_step,
                smoothing_sigma,
                environment
            )
            new_dfs.append(new_df)

    result_df = pd.concat(new_dfs)
    return result_df

def segmentize_df(df, subset):
    # Step 1: Filter rows where x_robot or x_fish is NaN
    valid_df = df.dropna(subset=subset)
    
    # Step 2: Identify segments where 'frame' increases consecutively by +1
    valid_df['frame_diff'] = valid_df['frame'].diff()  # Compute the difference between consecutive frames
    valid_df['segment'] = (valid_df['frame_diff'] != 1).cumsum()  # Create segment IDs for consecutive frames
    
    # Step 3: Group by segments and filter out segments with less than 25 frames
    segments = []
    for segment_id, segment_df in valid_df.groupby('segment'):
        if len(segment_df) >= 25:
            segments.append(segment_df.drop(columns=['frame_diff', 'segment']))  # Remove helper columns
    
    return segments

def process_df_segment(segment_df: pd.DataFrame, time_step, smoothing_sigma, environment: Environment):
    # Smooth poses
    segment_df['x'] = gaussian_filter1d(segment_df['x'], sigma=smoothing_sigma)
    segment_df['y'] = gaussian_filter1d(segment_df['y'], sigma=smoothing_sigma)
    sin_z = gaussian_filter1d(np.sin(segment_df['z']), sigma=smoothing_sigma)
    cos_z = gaussian_filter1d(np.cos(segment_df['z']), sigma=smoothing_sigma)
    segment_df['z'] = np.arctan2(sin_z, cos_z)

    # Velocity vector
    segment_df['dx'] = segment_df['x'].diff() / time_step
    segment_df['dy'] = segment_df['y'].diff() / time_step

    # Speed
    segment_df['speed'] = np.sqrt(segment_df['dx']**2 + segment_df['dy']**2)

    # Acceleration vector
    segment_df['ax'] = segment_df['dx'].diff() / time_step
    segment_df['ay'] = segment_df['dy'].diff() / time_step

    # Acceleration
    segment_df['acceleration'] = segment_df['speed'].diff() / time_step

    # Direction of motion (heading) based on velocity vector
    segment_df['heading'] = np.arctan2(segment_df['dy'], segment_df['dx'])

    # Turning angle (heading)
    h_diff = np.diff(segment_df['heading'].values)
    h_diff_adjusted = np.array([
        h_diff,
        h_diff + 2 * np.pi,
        h_diff - 2 * np.pi
    ])
    turn = h_diff_adjusted[np.abs(h_diff_adjusted).argmin(axis=0), np.arange(h_diff.size)]
    segment_df['turn_heading'] = np.concatenate(([np.nan], turn))

    # Turning angle (z)
    z_diff = np.diff(segment_df['z'].values)
    z_diff_adjusted = np.array([
        z_diff,
        z_diff + 2 * np.pi,
        z_diff - 2 * np.pi
    ])
    turn = z_diff_adjusted[np.abs(z_diff_adjusted).argmin(axis=0), np.arange(z_diff.size)]
    segment_df['turn_z'] = np.concatenate(([np.nan], turn))

    # Wall distances
    segment_df['wall_distances'] = segment_df.apply(
        lambda row: environment.get_wall_distances(row[['x', 'y']]), axis=1
    )
    segment_df['min_wall_distance'] = segment_df['wall_distances'].apply(np.min)
    nearest_wall_idx = segment_df['wall_distances'].apply(np.argmin)
    
    # Wall orientations
    # wall_orientation = []
    # np_x = segment_df['x'].to_numpy()
    # np_y = segment_df['y'].to_numpy()
    # np_z = segment_df['z'].to_numpy()
    # for i, wall_index in enumerate(nearest_wall_idx):
    #     x, y, z = np_x[i], np_y[i], np_z[i]
    #     if np.isnan(x) or np.isnan(y) or np.isnan(z):
    #         wall_orientation.append(np.nan)
    #         continue
    #     wall_orientation = environment.get_orientation_to_wall(
    #         x, y, z, wall_index
    #     )
    #     wall_orientation.append(wall_orientation)
    # segment_df['wall_orientation'] = wall_orientation

    return segment_df

def combined_processing(exp, df_robot, df_fish):
    df = combine_dfs(df_robot, df_fish)
    df = process_combined_df(exp, df)
    return df

def combine_dfs(df_robot, df_fish):
    df = pd.merge(
        df_robot, df_fish, on="frame", suffixes=("_robot", "_fish")
    )
    return df

def process_combined_df(exp: Experiment, df: pd.DataFrame):
    new_dfs = []

    time_step = exp.period

    segments = segmentize_df(df, ["x_robot", "x_fish"])

    for segment_df in segments:
        if not segment_df.empty:
            new_df = process_combined_df_segment(
                segment_df,
            )
            new_dfs.append(new_df)

    df = pd.concat(new_dfs)

    df["time"] = df["frame"] - df["frame"].min() + 1
    df["time_sec"] = df["time"] * time_step

    df = df.dropna(subset=["distance", "speed_robot", "speed_fish"])

    return df

def process_combined_df_segment(df: pd.DataFrame):
    # Relative position
    df["x_fr"] = df["x_fish"] - df["x_robot"]
    df["y_fr"] = df["y_fish"] - df["y_robot"]

    # Distance
    df["distance"] = np.sqrt(df["x_fr"] ** 2 + df["y_fr"] ** 2)

    # Normalized relative positions
    df["x_fr_n"] = df["x_fr"] / df["distance"]
    df["y_fr_n"] = df["y_fr"] / df["distance"]

    # Robot approach speed (projection of robot's velocity onto the relative position vector)
    df["v_app"] = df["dx_robot"] * df["x_fr_n"] + df["dy_robot"] * df["y_fr_n"]
    df["apv"] = df["v_app"]
    # Robot approach angle (cosinus of the angle between the robot's velocity and the relative position vector)
    df["cos_app"] = df["v_app"] / df["speed_robot"]

    # Fish avoidance speed(projection of fish's velocity onto the relative position vector)
    df["v_avoid"] = df["dx_fish"] * df["x_fr_n"] + df["dy_fish"] * df["y_fr_n"]
    df["rev"] = df["v_avoid"]
    # Fish avoidance acceleration (projection of fish's acceleration onto the relative position vector)
    df["a_avoid"] = df["ax_fish"] * df["x_fr_n"] + df["ay_fish"] * df["y_fr_n"]
    df["rea"] = df["a_avoid"]
    # Fish avoidance angle (cosinus of the angle between the fish's velocity and the relative position vector)
    df["cos_avoid"] = df["v_avoid"] / df["speed_fish"]

    return df


if __name__ == "__main__":
    process_experiments()