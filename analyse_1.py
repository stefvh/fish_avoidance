import datetime
import pickle as pkl
from enum import Enum
import pandas as pd

import numpy as np

from analyse_trials import get_valid_trials
from analyse_measuring import Results
from metric import normalize_series
import maths


folder = "data/RoboFish_Commander_Data/JUNE 13 (SET 2)/20230613-165310"

### Open data
with open(folder + "/robot_poses.pkl", "rb") as f:
    robot_poses = pkl.load(f)
# with open(folder + "/robot_directions.pkl", "rb") as f:
#     robot_directions = pkl.load(f)
with open(folder + "/robot_states.pkl", "rb") as f:
    robot_states = pkl.load(f)
with open(folder + "/fish_poses.pkl", "rb") as f:
    fish_poses = pkl.load(f)
# with open(folder + "/parameters.pkl", "rb") as f:
#     parameters = pkl.load(f)

calendar_time_points = robot_poses["time"]
init_time = calendar_time_points[0]
init_time = init_time.to_pydatetime()
# io_robot_poses = np.load(folder + "/io_robot_poses.npy")
# io_fish_poses = np.load(folder + "/io_fish_poses.npy")

### Get valid trials
trials = get_valid_trials()
# for trial in trials:
#     print(trial['id'], trial['start_time'], trial['end_time'])

### Get measurements
list_rel_time_sec = []

time_windows = [1]
for time_window in time_windows:
    print("Time window:", time_window)
    for trial in trials:
        # Compute time
        start_time = datetime.timedelta(
            hours=trial["start_time"].hour,
            minutes=trial["start_time"].minute,
            seconds=trial["start_time"].second,
            microseconds=trial["start_time"].microsecond,
        )
        end_time = datetime.timedelta(
            hours=trial["end_time"].hour,
            minutes=trial["end_time"].minute,
            seconds=trial["end_time"].second,
            microseconds=trial["end_time"].microsecond,
        )
        trial_start_time = init_time + start_time
        trial_end_time = init_time + end_time
        print("Trial:", trial["id"], trial_start_time, trial_end_time)

        # Filter data
        trial_robot_poses = robot_poses[
            (robot_poses["time"] >= trial_start_time)
            & (robot_poses["time"] <= trial_end_time)
        ]
        trial_robot_poses = trial_robot_poses.reset_index(drop=True)
        trial_fish_poses = fish_poses[
            (fish_poses["time"] >= trial_start_time)
            & (fish_poses["time"] <= trial_end_time)
        ]
        trial_fish_poses = trial_fish_poses.reset_index(drop=True)
        trial_robot_states = robot_states[
            (robot_states["time"] >= trial_start_time)
            & (robot_states["time"] <= trial_end_time)
        ]
        trial_robot_states = trial_robot_states.reset_index(drop=True)

        # Measure
        trial_robot_poses["relative_time_seconds"] = trial_robot_poses.apply(
            lambda row: (row["time"] - trial_start_time).total_seconds(), axis=1
        )

        list_fish_velocity = []
        batch = min(5, len(trial_robot_poses))
        for index, row in trial_robot_poses.iterrows():
            if index > batch - 1:
                list_fish_velocity.append(
                    np.linalg.norm(
                        trial_fish_poses.loc[index, ["x", "y"]]
                        - trial_fish_poses.loc[index - batch, ["x", "y"]]
                    )
                    / (row["relative_time_seconds"] - trial_robot_poses.loc[index - batch, "relative_time_seconds"])
                )
        
        list_fish_orientation = []
        batch = min(5, len(trial_robot_poses))
        previous_fish_orientation = None
        range_fish_orientation = np.array([0, 2 * np.pi])
        for index, row in trial_robot_poses.iterrows():
            if index > batch - 1:
                fish_vector = trial_fish_poses.loc[index, ["x", "y"]] - trial_fish_poses.loc[index - batch, ["x", "y"]]
                fish_orientation = np.arctan2(fish_vector["y"], fish_vector["x"]) % (2 * np.pi)
                fish_orientation += range_fish_orientation[0]
                if previous_fish_orientation is not None:
                    if abs(fish_orientation - previous_fish_orientation) > np.pi:
                        if previous_fish_orientation > np.mean(range_fish_orientation):
                            fish_orientation += 2 * np.pi
                            range_fish_orientation += 2 * np.pi
                        else:
                            fish_orientation -= 2 * np.pi
                            range_fish_orientation -= 2 * np.pi
                previous_fish_orientation = fish_orientation
                list_fish_orientation.append(fish_orientation)
        
        list_robot_velocity = []
        batch = min(5, len(trial_robot_poses))
        for index, row in trial_robot_poses.iterrows():
            if index > batch - 1:
                list_robot_velocity.append(
                    np.linalg.norm(
                        trial_robot_poses.loc[index, ["x", "y"]]
                        - trial_robot_poses.loc[index - batch, ["x", "y"]]
                    )
                    / (row["relative_time_seconds"] - trial_robot_poses.loc[index - batch, "relative_time_seconds"])
                )
        
        list_robot_fish_distance = []
        for index, row in trial_robot_poses.iterrows():
            list_robot_fish_distance.append(
                np.linalg.norm(
                    trial_robot_poses.loc[index, ["x", "y"]]
                    - trial_fish_poses.loc[index, ["x", "y"]]
                )
            )
        
        list_robot_fish_vector_angle = []
        batch = min(5, len(trial_robot_poses))
        previous_angle = 0
        for index, row in trial_robot_poses.iterrows():
            if index > batch - 1:
                fish_vector = trial_fish_poses.loc[index, ["x", "y"]] - trial_fish_poses.loc[index - batch, ["x", "y"]]
                tfp = trial_fish_poses.loc[list(range(index - batch, index)), ["x", "y"]]
                tfp = tfp.to_numpy()
                trp = trial_robot_poses.loc[list(range(index - batch, index)), ["x", "y"]]
                trp = trp.to_numpy()
                robot_fish_vectors = tfp[1:, :2] - trp[:-1, :2]
                mean_robot_fish_vector = np.mean(robot_fish_vectors, axis=0)
                if np.linalg.norm(mean_robot_fish_vector) * np.linalg.norm(fish_vector) == 0:
                    angle = previous_angle
                else:
                    angle = np.arccos(np.dot(mean_robot_fish_vector, fish_vector) / (np.linalg.norm(mean_robot_fish_vector) * np.linalg.norm(fish_vector)))
                    previous_angle = angle
                list_robot_fish_vector_angle.append(angle)
        
        list_robot_angle_diff = []
        batch = min(5, len(trial_robot_poses))
        previous_angle = 0
        for index, row in trial_robot_poses.iterrows():
            if index > batch - 1:
                robot_vector = trial_robot_poses.loc[index, ["x", "y"]] - trial_robot_poses.loc[index - batch, ["x", "y"]]
                tfp = trial_fish_poses.loc[list(range(index - batch, index)), ["x", "y"]]
                tfp = tfp.to_numpy()
                trp = trial_robot_poses.loc[list(range(index - batch, index)), ["x", "y"]]
                trp = trp.to_numpy()
                robot_fish_vectors = tfp[1:, :2] - trp[:-1, :2]
                mean_robot_fish_vector = np.mean(robot_fish_vectors, axis=0)

                r_angle = np.arctan2(robot_vector["y"], robot_vector["x"]) % (2 * np.pi)
                rfv_angle = np.arctan2(mean_robot_fish_vector[1], mean_robot_fish_vector[0]) % (2 * np.pi)
                angle_diff = maths.angles_absolute_difference(r_angle, rfv_angle)
                
                list_robot_angle_diff.append(angle_diff)
        
        list_robot_fish_repulsion = []
        batch = min(5, len(trial_robot_poses))
        for index, row in trial_robot_poses.iterrows():
            if index > batch - 1:
                tfp = trial_fish_poses.loc[list(range(index - batch, index)), ["x", "y"]]
                tfp = tfp.to_numpy()
                f_v = tfp[1:, :2] - tfp[:-1, :2]

                trp = trial_robot_poses.loc[list(range(index - batch, index)), ["x", "y"]]
                trp = trp.to_numpy()
                b_p = tfp[1:, :2] - trp[:-1, :2]
                b_p_n = normalize_series(b_p)

                repulsion = np.sum(f_v * b_p_n, axis=1).mean()
                list_robot_fish_repulsion.append(repulsion)
        
        index_min_frd = np.argmin(list_robot_fish_distance)
        min_fish_robot_distance = list_robot_fish_distance[index_min_frd]

        time_window = 5
        rep_peak_increase = 0
        for i in range(index_min_frd - 10, min(index_min_frd + 10, len(list_robot_fish_repulsion) - time_window)):
            increase = list_robot_fish_repulsion[i + time_window] - list_robot_fish_repulsion[i]
            rep_peak_increase = max(rep_peak_increase, increase)
        
        time_window = 25
        dist_peak_increase = 0
        for i in range(index_min_frd - 10, min(index_min_frd + 25, len(list_robot_fish_distance) - time_window)):
            increase = list_robot_fish_distance[i + time_window] - list_robot_fish_distance[i]
            dist_peak_increase = max(dist_peak_increase, increase)
        
        robot_fish_angle_peak = list_robot_fish_vector_angle[index_min_frd - 10]

        time_window = 5
        fish_orientation_peak_increase = 0
        for i in range(index_min_frd - 5, min(index_min_frd + 5, len(list_fish_orientation) - time_window)):
            increase = maths.angles_absolute_difference(list_fish_orientation[i + time_window], list_fish_orientation[i])
            fish_orientation_peak_increase = max(fish_orientation_peak_increase, increase)
        
        rob_vel_peak = list_robot_velocity[index_min_frd - 10]
        
        trial["results"] = Results(
            list_rel_time_sec=trial_robot_poses["relative_time_seconds"].to_list(),
            list_fish_velocity=list_fish_velocity,
            list_fish_orientation=list_fish_orientation,
            list_robot_velocity=list_robot_velocity,
            list_robot_fish_distance=list_robot_fish_distance,
            list_robot_fish_angle=list_robot_fish_vector_angle,
            list_robot_angle_diff=list_robot_angle_diff,
            list_robot_fish_repulsion=list_robot_fish_repulsion,
            min_fish_robot_distance=min_fish_robot_distance,
            max_increase_repulsion=rep_peak_increase,
            max_increase_distance=dist_peak_increase,
            max_increase_fish_ori=fish_orientation_peak_increase,
            hit_robot_velocity=rob_vel_peak,
            hit_robot_fish_angle=robot_fish_angle_peak,
            poses_robot=trial_robot_poses,
            poses_fish=trial_fish_poses,
        )
        

# save trials
with open(folder + "/trials.pkl", "wb") as f:
    pkl.dump(trials, f)
