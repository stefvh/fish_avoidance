

import numpy as np
import pandas as pd


if __name__ == "__main__":
    df = pd.read_csv("analysis/clean/measurements.csv")

    # Calculate relative position of the robot to the fish
    df["x_rf"] = df["x_robot"] - df["x_fish"]
    df["y_rf"] = df["y_robot"] - df["y_fish"]

    # Calculate relative angle and adjust by fish's heading
    df["angle_rf"] = np.arctan2(df["y_rf"], df["x_rf"]) - df["heading_fish"]
    df["angle_rf"] = np.mod(df["angle_rf"], 2 * np.pi)

    # Count rows where angle_rf > pi
    print(df[df["angle_rf"] > np.pi].shape)