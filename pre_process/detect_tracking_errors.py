from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.experiment import Experiment

import os

import numpy as np
import pandas as pd

from pre_process.manual_poses import generate_suggestion

from utils.parser import get_default_parser


def parse_args():
    parser = get_default_parser()

    parser.add_argument("-sf", "--start_frame", type=int, default=13250)
    parser.add_argument("-ef", "--end_frame", type=int, default=13290)

    args = parser.parse_args()
    return args


def detect_identity_swaps(
    experiment: Experiment, start_frame: int, end_frame: int, save_folder: str
):
    # Loop through the frames from start_frame to end_frame
    # When the robot_pose at frame i+1 is seemingly at the position of the fish_pose at frame i, we have an identity swap
    identity_swaps = []

    for frame in range(start_frame, end_frame):
        i_robot_pose = experiment.robot_poses[frame]
        j_robot_pose = experiment.robot_poses[frame + 1]

        i_fish_pose = experiment.fish_poses[frame]
        j_fish_pose = experiment.fish_poses[frame + 1]

        if (
            np.isnan(i_robot_pose).any()
            or np.isnan(i_fish_pose).any()
            or np.isnan(j_robot_pose).any()
            or np.isnan(j_fish_pose).any()
        ):
            continue

        threshold = 5

        c1 = np.linalg.norm(i_robot_pose[:2] - j_robot_pose[:2]) > threshold
        c2 = np.linalg.norm(i_fish_pose[:2] - j_fish_pose[:2]) > threshold

        c3 = np.linalg.norm(i_robot_pose[:2] - j_fish_pose[:2]) < threshold
        c4 = np.linalg.norm(i_fish_pose[:2] - j_robot_pose[:2]) < threshold

        if (c1 and c2) and (c3 or c4):
            print(f"Identity swap detected at frame {frame}.")
            identity_swaps.append(frame)

    with open(save_folder + "/detected_identity_issues.csv", "w") as f:
        f.write("frame_number\n")
        for frame in identity_swaps:
            f.write(f"{frame}\n")

    return


def _detect_missing_poses(poses, start_frame: int, end_frame: int, type: str):
    last_tracked_pose = None
    last_tracked_frame = None

    data = {
        "frame_number": [],
        "type": [],
        "x": [],
        "y": [],
        "z": [],
    }
    frame_windows = {
        "frame_number": [],
        "window_length": [],
    }
    for frame in range(start_frame, end_frame):
        pose = poses[frame]

        if last_tracked_pose is None:
            if np.isnan(pose).any():
                continue
            else:
                last_tracked_pose = pose
                last_tracked_frame = frame

        if np.isnan(pose).any():
            continue
        else:
            if last_tracked_frame + 1 != frame:
                new_poses, _ = generate_suggestion.suggest_poses_from_to_frame(
                    poses, last_tracked_frame, frame, ""
                )
                new_poses = new_poses[1:-1]
                for i, new_pose in enumerate(new_poses):
                    data["frame_number"].append(last_tracked_frame + i + 1)
                    data["type"].append('"' + type + '"')
                    data["x"].append(new_pose[0])
                    data["y"].append(new_pose[1])
                    data["z"].append(new_pose[2])
                if len(new_poses) > 0:
                    frame_windows["frame_number"].append(last_tracked_frame + 1)
                    frame_windows["window_length"].append(len(new_poses))

            last_tracked_pose = pose
            last_tracked_frame = frame
    return pd.DataFrame(data), pd.DataFrame(frame_windows)


def detect_missing_poses(
    experiment: Experiment, start_frame: int, end_frame: int, save_folder: str
):
    df_suggested_robot, df_frame_windows_robot = _detect_missing_poses(
        experiment.robot_poses, start_frame, end_frame, "robot"
    )
    df_suggested_fish, df_frame_windows_fish = _detect_missing_poses(
        experiment.fish_poses, start_frame, end_frame, "fish"
    )

    df_suggested = pd.concat([df_suggested_robot, df_suggested_fish])
    df_suggested = df_suggested.sort_values(by=["frame_number", "type"])
    df_suggested[["frame_number"]] = df_suggested[["frame_number"]].astype(int)

    df_suggested.to_csv(
        save_folder + "/detected_missing_poses.csv", index=False, quotechar="'"
    )

    df_frame_windows_robot.to_csv(
        save_folder + "/detected_missing_frame_windows_robot.csv", index=False
    )
    df_frame_windows_fish.to_csv(
        save_folder + "/detected_missing_frame_windows_fish.csv", index=False
    )


def _run(experiment: Experiment, start_frame: int, end_frame: int, save_folder: str):
    detect_identity_swaps(experiment, start_frame, end_frame, save_folder)
    detect_missing_poses(experiment, start_frame, end_frame, save_folder)


def run(experiment: Experiment, start_frame: int, end_frame: int):
    output_folder = experiment.output_folder
    # with open(output_folder + "/experiment.pkl", "rb") as f:
    #     experiment = pickle.load(f)

    save_folder = output_folder + "/error_detection"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    _run(experiment, start_frame, end_frame, save_folder)

if __name__ == "__main__":
    args = parse_args()
    variables = vars(args)

    from core.experiment import Experiment, get_experiment
    experiment = get_experiment(**variables)

    run(
        experiment,
        variables["start_frame"],
        variables["end_frame"],
    )
