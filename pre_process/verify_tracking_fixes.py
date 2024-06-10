from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.experiment import Experiment

import os

import numpy as np
import pandas as pd

from utils.parser import get_default_parser


def parse_args():
    parser = get_default_parser()

    parser.add_argument("-sf", "--start_frame", type=int, default=13250)
    parser.add_argument("-ef", "--end_frame", type=int, default=13290)

    args = parser.parse_args()
    return args


def _detect_manual_frame_windows(poses: np.ndarray, corrections: np.ndarray, start_frame: int, end_frame: int):
    last_tracked_pose = None
    last_tracked_frame = None

    frame_windows = {
        "frame_number": [],
        "window_length": [],
    }
    for frame in range(start_frame, end_frame):
        pose = poses[frame]
        corrected = corrections[frame]

        if last_tracked_pose is None:
            if np.isnan(pose).any() or corrected:
                continue
            else:
                last_tracked_pose = pose
                last_tracked_frame = frame

        if np.isnan(pose).any() or corrected:
            continue
        else:
            if last_tracked_frame + 1 != frame:
                frame_windows["frame_number"].append(last_tracked_frame + 1)
                frame_windows["window_length"].append(frame - last_tracked_frame - 1)

            last_tracked_pose = pose
            last_tracked_frame = frame
    return pd.DataFrame(frame_windows)


def detect_manual_frame_windows(
    experiment: Experiment, start_frame: int, end_frame: int, save_folder: str
):
    df_frame_windows_robot = _detect_manual_frame_windows(
        experiment.robot_poses, 
        experiment.robot_manual_corrections,
        start_frame, 
        end_frame,
    )
    df_frame_windows_fish = _detect_manual_frame_windows(
        experiment.fish_poses, 
        experiment.fish_manual_corrections,
        start_frame, 
        end_frame,
    )

    df_frame_windows_robot.to_csv(
        save_folder + "/detected_man_fw_robot.csv", index=False
    )
    df_frame_windows_fish.to_csv(
        save_folder + "/detected_man_fw_fish.csv", index=False
    )


def run(experiment: Experiment, start_frame: int, end_frame: int):
    output_folder = experiment.output_folder
    # with open(output_folder + "/experiment.pkl", "rb") as f:
    #     experiment = pickle.load(f)

    save_folder = output_folder + "/fix_verification"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    detect_manual_frame_windows(
        experiment,
        start_frame,
        end_frame,
        save_folder,
    )

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
