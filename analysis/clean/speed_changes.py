import os
from matplotlib import patches, pyplot as plt
import pandas as pd
import numpy as np

from analysis.clean.avoidance_mechanisms import get_data
from analysis.clean.produce_trajectories import find_trajectories
from core.experiment import Experiment
from core.study import study

from plotting.publishing.subset_trials import get_trials_by_data
import utils.plotter as plotter


def get_trajectories(
    df: pd.DataFrame, v_app_threshold: float = 2, reaction_delay: int = 8, min_length: int = 10
):
    """
    Find robot approach trajectories based on v_app > threshold, and compute metrics for the fish's reaction.

    Parameters:
    df (pd.DataFrame): Input dataframe containing fish and robot metrics for each frame.
    v_app_threshold (float): Threshold for v_app to define robot approach.
    frame_delay (int): Number of additional frames after the trajectory ends to account for fish reaction.

    Returns:
    pd.DataFrame: A dataframe with one row per trajectory and calculated metrics.
    """

    # Identify frames where v_app > threshold
    above_threshold = df["v_app"] > v_app_threshold

    # Find the start and end indices of each trajectory (consecutive frames where v_app > threshold)
    df["trajectory_id"] = (
        above_threshold != above_threshold.shift(1)
    ).cumsum() * above_threshold
    trajectories = df[df["trajectory_id"] > 0]["trajectory_id"].unique()

    # Prepare output list for storing trajectory data
    output_rows = []

    # Loop over each trajectory to compute metrics
    for traj_id in trajectories:
        traj_df = df[df["trajectory_id"] == traj_id]

        # Check if the trajectory meets the minimum length and consecutive frame requirements based on "frame" column
        if len(traj_df) < min_length or not (traj_df["frame"].diff().iloc[1:] == 1).all():
            continue  # Skip this trajectory if it's too short or not consecutive


        # Extend the trajectory by frame_delay
        last_frame = traj_df.index[-1]
        post_traj_df = df[
            (df.index > last_frame) & (df.index <= last_frame + reaction_delay)
        ]

        full_traj_df = pd.concat([traj_df, post_traj_df])

        # Calculate metrics only for the original trajectory
        min_distance = traj_df["distance"].min()
        avg_robot_speed = traj_df["speed_robot"].mean()
        angles_rad = traj_df["angle_rf"].to_numpy()
        mean_sin = np.mean(np.sin(angles_rad))
        mean_cos = np.mean(np.cos(angles_rad))
        avg_angle = np.arctan2(mean_sin, mean_cos)
        avg_angle = np.mod(avg_angle, 2 * np.pi)

        # Calculate metrics for the full trajectory (original + post-trajectory delay)
        avg_acceleration = full_traj_df["acceleration_fish"].mean()
        max_acceleration = full_traj_df["acceleration_fish"].max()
        max_accel_frame = full_traj_df["acceleration_fish"].idxmax()

        max_turn = full_traj_df["turn_heading_fish"].abs().max()
        max_turn_frame = full_traj_df["turn_heading_fish"].idxmax()

        # distance at max acceleration and max turn (-4)
        min_frame = full_traj_df.index[0]
        max_accel_frame_pre = max(min_frame, max_accel_frame - 4)
        max_turn_frame_pre = max(min_frame, max_turn_frame - 4)
        distance_max_accel = full_traj_df.loc[max_accel_frame_pre : max_accel_frame, "distance"].min()
        distance_max_turn = full_traj_df.loc[max_turn_frame_pre : max_turn_frame, "distance"].min()

        # robot speed at max acceleration and max turn (-4)
        speed_robot_max_accel = full_traj_df.loc[max_accel_frame_pre : max_accel_frame, "speed_robot"].max()
        speed_robot_max_turn = full_traj_df.loc[max_turn_frame_pre : max_turn_frame, "speed_robot"].max()

        # angle_rf at max acceleration and max turn
        angle_rf_max_accel = full_traj_df.loc[
            max_accel_frame_pre, "angle_rf"
        ]
        angle_rf_max_turn = full_traj_df.loc[
            max_turn_frame_pre, "angle_rf"
        ]

        # cos_rev at max acceleration and max turn, and delayed frames

        cos_rev_max_accel = df.loc[
            max_accel_frame + 4, "cos_avoid"
        ]
        cos_rev_max_turn = df.loc[
            max_turn_frame + 4, "cos_avoid"
        ]

        # cos_rev at the end of full_traj_df
        cos_rev_end = full_traj_df["cos_avoid"].iloc[-1]

        # start and end angle
        start_angle = traj_df["angle_rf"].iloc[0]
        end_angle = traj_df["angle_rf"].iloc[-1]

        # Changes in speed
        duration = full_traj_df["frame"].iloc[-1] - full_traj_df["frame"].iloc[0]
        mean_speed = full_traj_df["speed_fish"].mean()
        std_speed = full_traj_df["speed_fish"].std()
        max_speed = full_traj_df["speed_fish"].max()
        min_speed = full_traj_df["speed_fish"].min()
        range_speed = max_speed - min_speed
        mean_acceleration = full_traj_df[full_traj_df["acceleration_fish"] > 0]["acceleration_fish"].mean()
        mean_deceleration = full_traj_df[full_traj_df["acceleration_fish"] < 0]["acceleration_fish"].mean()
        time_accelerating = full_traj_df[full_traj_df["acceleration_fish"] > 0].shape[0] / full_traj_df.shape[0]
        time_decelerating = full_traj_df[full_traj_df["acceleration_fish"] < 0].shape[0] / full_traj_df.shape[0]
        max_acceleration = full_traj_df["acceleration_fish"].max()
        max_acceleration_frame = full_traj_df["acceleration_fish"].idxmax()
        max_speed_frame = full_traj_df["speed_fish"].idxmax()
        distance_max_acc = full_traj_df.loc[max_acceleration_frame, "distance"]
        distance_max_speed = full_traj_df.loc[max_speed_frame, "distance"]
        start_distance = full_traj_df["distance"].iloc[0]
        end_distance = full_traj_df["distance"].iloc[-1]

        # Build output row
        row = {
            "experiment_id": traj_df["experiment_id"].iloc[0],
            "trajectory_id": traj_id,
            "start_frame": full_traj_df.index[0],
            "end_frame": full_traj_df.index[-1],
            "start_distance": start_distance,
            "end_distance": end_distance,
            "duration": duration,
            "mean_speed": mean_speed,
            "std_speed": std_speed,
            "max_speed": max_speed,
            "min_speed": min_speed,
            "range_speed": range_speed,
            "mean_acceleration": mean_acceleration,
            "mean_deceleration": mean_deceleration,
            "time_accelerating": time_accelerating,
            "time_decelerating": time_decelerating,
            "distance_max_acc": distance_max_acc,
            "distance_max_speed": distance_max_speed,
            "avg_acceleration": avg_acceleration,
            "max_acceleration": max_acceleration,
            "max_accel_frame": max_accel_frame,
            "max_turn": max_turn,
            "max_turn_frame": max_turn_frame,
            "min_distance": min_distance,
            "avg_robot_speed": avg_robot_speed,
            "avg_angle": avg_angle,
            "distance_max_accel": distance_max_accel,
            "distance_max_turn": distance_max_turn,
            "speed_robot_max_accel": speed_robot_max_accel,
            "speed_robot_max_turn": speed_robot_max_turn,
            "angle_rf_max_accel": angle_rf_max_accel,
            "angle_rf_max_turn": angle_rf_max_turn,
            "cos_rev_max_accel": cos_rev_max_accel,
            "cos_rev_max_turn": cos_rev_max_turn,
            "cos_rev_end": cos_rev_end,
            "start_angle": start_angle,
            "end_angle": end_angle,
        }

        # Append the row to the output list
        output_rows.append(row)

    # Convert output rows to a dataframe
    output_df = pd.DataFrame(output_rows)

    return output_df

def plot_variable(df, variable):
    # plot x-axis: start_distance
    # plot y-axis: end_distance
    # color (z): variable

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    scatter = ax.scatter(df["start_distance"], df["end_distance"], c=df[variable], cmap="coolwarm", edgecolor="k")
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(variable)
    ax.set_xlabel("Start distance")
    ax.set_ylabel("End distance")
    ax.set_title(f"Start and end distance by {variable}")

    # Save the plot
    output_folder = "analysis/clean/figs/speed_changes"
    os.makedirs(output_folder, exist_ok=True)
    fig.savefig(os.path.join(output_folder, f"{variable}.pdf"))
    plt.close()


if __name__ == "__main__":
    df = get_data()
    df_traj = get_trajectories(df)

    # Don't consider trajectories where end_distance > 40
    df_traj = df_traj[df_traj["end_distance"] <= 40]
    df_traj = df_traj[df_traj["start_distance"] <= 60]

    # Don't consider where mean_speed > 10
    df_traj = df_traj[df_traj["mean_speed"] <= 10]

    variables = [
        "mean_speed",
        "std_speed",
        "max_speed",
        "min_speed",
        "range_speed",
        "mean_acceleration",
        "mean_deceleration",
        "time_accelerating",
        "time_decelerating",
        "distance_max_acc",
        "distance_max_speed",
    ]

    for variable in variables:
        plot_variable(df_traj, variable)