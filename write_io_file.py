import datetime
import pandas as pd
import os
import robofish.io
import numpy as np

import pickle as pkl

def write_io_file(path, df_robot_poses, df_fish_poses):
    calendar_time_points = df_robot_poses['time']

    io_robot_poses = format_io_poses(df_robot_poses, calendar_time_points, type="robot")
    io_fish_poses = format_io_poses(df_fish_poses, calendar_time_points, type="fish")

    f = robofish.io.File(
        path + "/test_tracks_io.hdf5",
        mode="w",
        world_size_cm=[100., 100.],
        frequency_hz=25,
        calendar_time_points=calendar_time_points,
    )
    robots = f.create_multiple_entities(category="robot", poses=io_robot_poses)
    fish = f.create_multiple_entities(category="fish", poses=io_fish_poses)

    # print(f"Robot positions with orientations:\n{robots.entity_poses_rad}")
    # print(f"Fish positions with orientations:\n{fish.entity_poses_rad}")
    print(
        f"Poses Shape:\t{f.entity_poses_rad.shape}.\t"
        + "Representing(entities, timesteps, pose dimensions (x, y, ori)"
    )


    f.validate()
    f.close()

def format_io_poses(df_poses, time_points, type="robot"):
    ids = np.unique(df_poses['id'])
    print(f"IDs: {ids}, type: {type}")
    if type == "robot":
        k_dim = 4
    elif type == "fish":
        k_dim = 4
    else:
        raise ValueError("Invalid type")
    io_poses = np.zeros((1, len(time_points), k_dim))
    for i, time in enumerate(time_points):
        pose = np.float32([np.nan] * k_dim)
        poses_by_id = {}
        previous_id = None
        for j, identifier in enumerate(ids):
            df = df_poses.loc[
                (df_poses['time'] == time) & 
                (df_poses['id'] == identifier)
            ]
            if not df.empty:
                pose[0] = max(min(float(df['x'].iloc[0] - 50.), 50.45), -50.45)
                pose[1] = max(min(float(df['y'].iloc[0] - 50.), 50.45), -50.45)
                if k_dim == 4:
                    pose[2] = np.cos((df['z'].iloc[0]))
                    pose[3] = np.sin((df['z'].iloc[0]))
                elif type == 3:
                    pose[2] = float(df['z'].iloc[0]) % (2 * np.pi)
                else:
                    raise ValueError("Invalid type")
                poses_by_id[identifier] = pose
        if len(poses_by_id) == 1:
            io_poses[0, i] = pose
            previous_id = list(poses_by_id.keys())[0]
        elif len(poses_by_id) > 1:
            print(f"Multiple IDs at same time point: {time}")
            if previous_id is None:
                # Choose pose closest to the left corner
                left_corner = np.array([-50.45, -50.45])
                distances = np.zeros(len(poses_by_id))
                for j, identifier in enumerate(poses_by_id.keys()):
                    distances[j] = np.linalg.norm(left_corner - poses_by_id[identifier][:2])
                previous_id = list(poses_by_id.keys())[np.argmin(distances)]
            io_poses[0, i] = poses_by_id[previous_id]
            # raise ValueError("Multiple IDs at same time point")
        else:
            if i == 0:
                raise ValueError("First pose is NaN")
            else:
                io_poses[0, i] = io_poses[0, i-1]
    return io_poses

if __name__ == "__main__":
    folder = "data/RoboFish_Commander_Data/JUNE 13 (SET 2)/20230613-165310"
    with open(folder + "/robot_poses.pkl", "rb") as f:
        robot_poses = pkl.load(f)
    with open(folder + "/robot_directions.pkl", "rb") as f:
        robot_directions = pkl.load(f)
    with open(folder + "/robot_states.pkl", "rb") as f:
        robot_states = pkl.load(f)
    with open(folder + "/fish_poses.pkl", "rb") as f:
        fish_poses = pkl.load(f)
    with open(folder + "/parameters.pkl", "rb") as f:
        parameters = pkl.load(f)
    write_io_file(folder, robot_poses, fish_poses)


