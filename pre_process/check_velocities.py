from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.experiment import Experiment

import os

import numpy as np
import pandas as pd

from utils.parser import parse_args


def _run(experiment: Experiment, save_folder: str):
    try:
        robot_poses, fish_poses = experiment.active_limited_tracks
    except UserWarning:
        return
    except:
        raise

    n_frames = fish_poses.shape[0]
    period = 1/25

    batch = 5
    batch_fish_poses = np.full_like(fish_poses, np.nan)
    for i in range(batch, fish_poses.shape[0]):
        batch_fish_poses[i] = np.nanmean(fish_poses[i - batch : i], axis=0)
    
    fish_velocity_batch = np.full(n_frames, np.nan)
    step = batch
    for i in range(batch, n_frames - batch):
        if (
            np.isnan(batch_fish_poses[i]).any()
            or np.isnan(batch_fish_poses[i + step]).any()
        ):
            continue
        else:
            fish_velocity_batch[i] = np.linalg.norm(
                batch_fish_poses[i + step, :2] - batch_fish_poses[i, :2]
            ) / (step * period)
    
    # create dataframe with frame_number and fish_velocity where fish_velocity > threshold
    start_frame = experiment.trials["start_frame"].min()
    df = pd.DataFrame(
        {
            "frame_number": np.arange(start_frame, start_frame + n_frames, 1),
            "fish_velocity": fish_velocity_batch,
        }
    )
    df = df[df["fish_velocity"] > 20]

    # for each frame_number find the related trial_id
    trial_ids = []
    for i in df["frame_number"]:
        trial_id = experiment.trials[
                (experiment.trials["start_frame"] <= i)
                & (experiment.trials["end_frame"] >= i)
            ]
        if not trial_id.empty:
            trial_id = int(trial_id["trial_id"].values[0])
        else:
            trial_id = np.nan
        trial_ids.append(trial_id)
    df["trial_id"] = trial_ids

    df.to_csv(f"{save_folder}/fish_velocity.csv", index=False)


def run(experiment: Experiment):
    output_folder = experiment.output_folder

    save_folder = output_folder + "/error_detection"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    _run(experiment, save_folder)


if __name__ == "__main__":
    args = parse_args()
    variables = vars(args)

    from core.experiment import get_experiment
    experiment = get_experiment(**variables)

    run(experiment)

