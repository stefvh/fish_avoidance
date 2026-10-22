
import pandas as pd

from analysis.frame_selection import get_distance_dip_frames


def by_robot_state(results: pd.DataFrame, robot_states: pd.DataFrame):
    # Add robot state
    results = results.merge(robot_states, on="frame_number", how="left")

    # Drop NaN values
    results = results[
        [
            "frame_number", 
            "fish_velocity_batch", 
            "fish_turn_batch", 
            "state"
        ]
    ]
    results = results.dropna(subset=["fish_velocity_batch", "fish_turn_batch", "state"])

    # Create condition
    results = results[
        results["state"].isin(["APPROACHING_SLOW", "APPROACHING_FAST", "INITIALIZING"])
    ]
    results["b_state"] = results["state"].apply(
        lambda x: 1 if "APPROACHING" in x else 0
    )

    return results, ["INITIALIZING", "APPROACHING"]

def by_time_near_distance_dip(results: pd.DataFrame, window: int = 10):
    results = results[
        [
            "trial_id",
            "frame_number",
            "fish_velocity_batch",
            "fish_turn_batch",
            "robot_distance",
        ]
    ]
    results = results.dropna(subset=["fish_velocity_batch", "fish_turn_batch"])

    dip_frames, _ = get_distance_dip_frames(results, window)
    dip_frames = [item for sublist in dip_frames for item in sublist]

    results["b_state"] = results["frame_number"].apply(
        lambda x: 1 if x in dip_frames else 0
    )

    return results, ["REST", "PEAK"]