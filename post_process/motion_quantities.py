import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from pre_process.environment import Environment


class MotionQuantities:
    def __init__(self, 
                 input : pd.DataFrame,
                 wall_corners : np.ndarray,
                 time_step : float = 0.04, 
                 window_size : int = 5):
        self.input = input

        self.time_step = time_step
        self.smoothing_sigma = window_size / 2

        self.environment = Environment(wall_corners)

        self.input['segment'] = self._identify_segments()
        self.df, self.df_segments = self.compute()
    
    def _identify_segments(self):
        valid_points = self.input['x'].notna() & self.input['y'].notna()
        segment_numbers = (valid_points != valid_points.shift()).cumsum() * valid_points
        return segment_numbers
    
    def _apply_gaussian_smoothing(self, series):
        return gaussian_filter1d(series, sigma=self.smoothing_sigma)
    
    def _process_segment(self, segment_df):
        # Apply Gaussian smoothing to positions
        segment_df['x'] = self._apply_gaussian_smoothing(segment_df['x'])
        segment_df['y'] = self._apply_gaussian_smoothing(segment_df['y'])

        # Calculate smoothed velocities
        segment_df['dx'] = segment_df['x'].diff() / self.time_step
        segment_df['dy'] = segment_df['y'].diff() / self.time_step

        # Calculate smoothed speed
        segment_df['speed'] = np.sqrt(segment_df['dx']**2 + segment_df['dy']**2)

        # Calculate smoothed accelerations
        segment_df['ax'] = segment_df['dx'].diff() / self.time_step
        segment_df['ay'] = segment_df['dy'].diff() / self.time_step

        # Calculate smoothed acceleration magnitude
        # segment_df['acceleration'] = np.sqrt(segment_df['ax']**2 + segment_df['ay']**2)
        segment_df['acceleration'] = segment_df['speed'].diff() / self.time_step

        # Calculate direction of motion (heading)
        segment_df['heading'] = np.arctan2(segment_df['dy'], segment_df['dx'])

        # Calculate turn rate (change in heading)
        delta_heading = np.diff(segment_df['heading'].values)
        turn = (delta_heading + np.pi) % (2 * np.pi) - np.pi
        segment_df['turn'] = np.concatenate(([np.nan], turn))
        segment_df['turn_rate'] = segment_df['turn'] / (self.time_step)

        # Calculate turn rate interval
        heading = segment_df['heading'].values
        turn_interval = 5
        delta_heading = heading[turn_interval:] - heading[:-turn_interval]
        turn = (delta_heading + np.pi) % (2 * np.pi) - np.pi
        turns = np.full(len(segment_df), np.nan)
        turns[turn_interval:] = turn
        segment_df['turn_intervaled'] = turns

        # Calculate distance from walls
        segment_df['wall_distances'] = segment_df.apply(
            lambda row: self.environment.get_wall_distances(row[['x', 'y']]), axis=1
        )
        segment_df['min_wall_distance'] = segment_df['wall_distances'].apply(np.min)

        return segment_df
    
    def compute(self):
        segments = self.input['segment'].dropna().unique()
        smoothed_dfs = []

        for segment in segments:
            segment_df = self.input[self.input['segment'] == segment]
            if not segment_df.empty:
                smoothed_segment_df = self._process_segment(segment_df)
                smoothed_dfs.append(smoothed_segment_df)

        result_df = pd.concat(smoothed_dfs)
        return result_df, smoothed_dfs
