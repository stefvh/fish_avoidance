import os

import pandas as pd
from core.experiment import Experiment
from core.study import study

import numpy as np

if __name__ == '__main__':
    full_data = study.data.copy()

    og_fcs = []
    new_fcs = []
    fin_fcs = []

    for exp in study.list_experiments:
        print("--------------------")
        print(f"Experiment {exp.id}")

        # Original frame count
        og_fc = exp.original_frame_count
        print(f"Original number of frames {og_fc}")

        # Number of frames after removing milling process
        filepath = "core/experiments.csv"
        if not os.path.exists(filepath):
            raise FileNotFoundError("No experiments.csv file found")
        else:
            df = pd.read_csv(filepath)
            row = df.loc[df["id"] == exp.id]
            start_frame = row["start_frame"].iloc[0]
            end_frame = row["end_frame"].iloc[0]
            new_fc = end_frame - start_frame
            print(f"Number of frames after removing milling process: {new_fc}")
            print(f"Number of frames removed: {og_fc - new_fc}")

        # Number of frames removed by detecting long gaps
        q_df = exp.q_df.dropna()
        fin_fc = q_df.shape[0]
        print(f"Number of frames after removing long gaps: {fin_fc}")
        print(f"Number of frames removed: {new_fc - fin_fc}")
        percentage = (new_fc - fin_fc) / new_fc * 100
        print(f"Percentage of frames removed: {percentage:.2f}%")

        # Number of frames corrected in new_fc range
        fish_corrections = exp.fish_manual_corrections # numpy array
        fish_corrections = fish_corrections[start_frame:end_frame]
        robot_corrections = exp.robot_manual_corrections # numpy array
        robot_corrections = robot_corrections[start_frame:end_frame]

        corrections = np.logical_or(fish_corrections, robot_corrections)

        nr_corrections = np.sum(corrections) - (new_fc - fin_fc)

        print(f"Number of corrections: {nr_corrections}")

        percentage = nr_corrections / fin_fc * 100
        print(f"Percentage of corrections: {percentage:.2f}%")

        n_removed = og_fc - fin_fc
        percentage = n_removed / og_fc * 100
        print(f"Number of frames removed: {n_removed}")
        print(f"Percentage of frames removed: {percentage:.2f}%")

        og_fcs.append(og_fc)
        new_fcs.append(new_fc)
        fin_fcs.append(fin_fc)

        # robot_poses, fish_poses = exp.get_tracks_limited_correction()
        # df_robot_positions = exp._transform_poses_to_valid_df(robot_poses)
        # df_fish_positions = exp._transform_poses_to_valid_df(fish_poses)
    
    tot_og_fc = sum(og_fcs)
    tot_new_fc = sum(new_fcs)
    tot_fin_fc = sum(fin_fcs)

    print("--------------------")

    print(f"Total number of frames originally: {tot_og_fc}")
    print(f"Total number of frames after removing milling process: {tot_new_fc}")
    print(f"Total number of frames after removing long gaps: {tot_fin_fc}")
    print(f"Total number of frames used in analysis: {full_data.shape[0]}")
    f_data = full_data.dropna()
    print(f"Total number of frames used in analysis after removing NaN values: {f_data.shape[0]}")