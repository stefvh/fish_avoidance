
import numpy as np
import pandas as pd
from analysis.clean.process import EXPERIMENT_IDS, segmentize_df

def compute_related_freezing(df_measurements: pd.DataFrame):
    new_dfs = []
    for exp_id in EXPERIMENT_IDS:
        df_exp = df_measurements[df_measurements["experiment_id"] == exp_id]
        new_df = _process_experiment(df_exp)
        new_dfs.append(new_df)
    df = pd.concat(new_dfs)
    return df

def _process_experiment(df: pd.DataFrame):
    new_dfs = []

    segments = segmentize_df(df, ["x_robot", "x_fish"])
    for segment_df in segments:
        if not segment_df.empty:
            new_df = _process_segment(segment_df)
            new_dfs.append(new_df)

    df = pd.concat(new_dfs)
    return df

def _process_segment(df: pd.DataFrame, delta_frames: int = 50):
    # Number of frames corresponding to 2 seconds (50 frames at 25Hz)

    # Shift x_fish and y_fish by delta_frames to compare the position delta_frames ahead
    df['x_future'] = df['x_fish'].shift(-delta_frames)
    df['y_future'] = df['y_fish'].shift(-delta_frames)

    # Calculate the distance between current and future positions
    df['fish_displ'] = euclidean_distance(df['x_fish'], df['y_fish'], df['x_future'], df['y_future'])

    # Mark as freezing if the distance is less than or equal to 2 cm
    df['freezing_related'] = df['fish_displ'] <= 2

    # Clean up the extra columns we don't need
    df.drop(columns=['x_future', 'y_future',], inplace=True)

    # TODO
    # freezing_cont = np.zeros(len(df))  # Initialize the new column as 0 (False)
    # radius = 6

    # for i, row_i in df.iterrows():
    #     # Get the position at time t (i-th row)
    #     p_t = np.array([row_i['x_fish'], row_i['y_fish']])
        
    #     # Check subsequent positions
    #     continuous_time = 0
    #     for j in range(i, len(df)):
    #         row_j = df.iloc[j]
    #         p_j = np.array([row_j['x_fish'], row_j['y_fish']])
    #         distance = euclidean_distance(p_t[0], p_t[1], p_j[0], p_j[1])
            
    #         if distance <= radius:
    #             # Accumulate continuous time within the radius
    #             continuous_time += 1
    #             # If continuous time exceeds the required frames, mark as 1 (True) in 'freezing_cont'
    #             if continuous_time >= delta_frames:
    #                 freezing_cont[i] = 1
    #                 break
    #         else:
    #             break  # Stop checking once the fish moves outside the radius
                
    # df['freezing_cont'] = freezing_cont

    # Check for freezing based on speed
    duration_frames = 50
    speed_threshold = 2
    df['freezing_spd'] = 0  # Initialize the new column as 0 (False)
    for i in range(len(df) - duration_frames):
        # Get the speed over the next `duration_frames`
        if np.all(df.iloc[i:i + duration_frames, df.columns.get_loc('speed_fish')] < speed_threshold):
            df.iloc[i, df.columns.get_loc('freezing_spd')] = 1  # Mark as freezing at the end of the duration

    # duration_frames = 50
    # distance_threshold = 3  # Threshold for distance traveled in cm
    # # Initialize the 'freezing_new' column as 0 (False)
    # df['freezing_dist'] = 0

    # df['distance_travelled'] = df['speed_fish'] / 25.0

    # # Calculate the cumulative sum of distances over the duration window
    # for i in range(len(df) - duration_frames):
    #     # Check if the sum of distance traveled in the next `duration_frames` is below the threshold
    #     total_distance = df['distance_travelled'].iloc[i:i + duration_frames].sum()
        
    #     # If the total distance traveled is below the threshold, mark as freezing
    #     if total_distance < distance_threshold:
    #         df.iloc[i + duration_frames, df.columns.get_loc('freezing_dist')] = 1

    duration_frames = 50
    distance_threshold = 2  # Threshold for distance traveled in cm
    # Initialize the 'freezing_new' column as 0 (False)
    df['freezing_dist'] = 0

    # Compute the change in position (displacement) for each frame based on smoothed positions
    df['dx_cm'] = df['x_fish'].diff()  # Change in x
    df['dy_cm'] = df['y_fish'].diff()  # Change in y

    # Calculate the distance traveled in each frame (Euclidean distance between consecutive positions)
    df['distance_travelled'] = np.sqrt(df['dx_cm']**2 + df['dy_cm']**2)

    # Now we will use a sliding window to sum the distances over the `duration_frames` (e.g., 50 frames)
    for i in range(len(df) - duration_frames):
        # Calculate the total distance traveled within the next `duration_frames`
        total_distance = df['distance_travelled'].iloc[i:i + duration_frames].sum()
        
        # If the total distance traveled is below the threshold, mark as freezing
        if total_distance < distance_threshold:
            df.iloc[i + duration_frames, df.columns.get_loc('freezing_dist')] = 1

    return df

def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
