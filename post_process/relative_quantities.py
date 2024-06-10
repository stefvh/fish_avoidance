import numpy as np
import pandas as pd


class RelativeQuantities:
    def __init__(
        self,
        robot_df: pd.DataFrame,
        fish_df: pd.DataFrame,
        time_step: float = 0.04,
        window_size: int = 5,
    ):
        self.robot_df = robot_df
        self.fish_df = fish_df

        self.time_step = time_step
        self.window_size = window_size

        self.df, self.df_segments = self.compute()

    def _process_segment(self, df):
        # Relative position
        df["x_fr"] = df["x_fish"] - df["x_robot"]
        df["y_fr"] = df["y_fish"] - df["y_robot"]

        # Calculate distance
        df["distance"] = np.sqrt(df["x_fr"] ** 2 + df["y_fr"] ** 2)

        # Normalize relative positions
        df["x_fr_n"] = df["x_fr"] / df["distance"]
        df["y_fr_n"] = df["y_fr"] / df["distance"]

        # Calculate approach velocity (projection of robot's velocity onto the relative position vector)
        df["apv"] = df["dx_robot"] * df["x_fr_n"] + df["dy_robot"] * df["y_fr_n"]
        df["cos_apv"] = df["apv"] / df["speed_robot"]

        # Calculate approach velocity over multiple frames
        df[f"apv_n={self.window_size}"] = (
            df["apv"].rolling(window=self.window_size, min_periods=1).mean()
        )
        df[f"cos_apv_n={self.window_size}"] = (
            df["cos_apv"].rolling(window=self.window_size, min_periods=1).mean()
        )

        return df

    def compute(self):
        input = pd.merge(
            self.robot_df, self.fish_df, on="frame", suffixes=("_robot", "_fish")
        )
        input["segment"] = input["segment_robot"]
        input = input.drop(columns=["segment_fish", "segment_robot"])

        segments = input["segment"].dropna().unique()
        segment_dfs = []

        for segment in segments:
            segment_df = input[input["segment"] == segment]
            if not segment_df.empty:
                processed_segment_df = self._process_segment(segment_df)
                segment_dfs.append(processed_segment_df)

        result_df = pd.concat(segment_dfs)
        return result_df, segment_dfs
