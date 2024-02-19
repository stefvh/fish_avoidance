import datetime
import pandas as pd
import os
import robofish.io
import numpy as np

import pickle as pkl

from write_io_file import format_io_poses

from metric import calculate_follow

from parameters import get_parameter_set

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


list_results = get_parameter_set("FISH_2")
for i in range(len(list_results)):
    list_results[i]["id"] = i

    list_results[i]["relative_time_seconds"] = []
    list_results[i]["distances"] = []
    list_results[i]["follow_vectors"] = []
    list_results[i]["velocity_changes"] = []
    list_results[i]["acceleration_changes"] = []
    list_results[i]["states"] = []
    list_results[i]["robot_velocities"] = []

    list_results[i]["relative_time_batches"] = []
    list_results[i]["follow_vectors_batched"] = []
    list_results[i]["velocity_changes_batched"] = []
    list_results[i]["acceleration_changes_batched"] = []
    list_results[i]["robot_velocities_batched"] = []

def get_dict_result(params):
    for i, p in enumerate(list_results):
        if (
            p["dummy"] == params["dummy"].values[0]
            and p["slow_speed"] == params["slow_speed"].values[0]
            and p["fast_speed"] == params["fast_speed"].values[0]
            and p["transition_rate"] == params["transition_rate"].values[0]
            and p["transition_distance"] == params["transition_distance"].values[0]
            and p["starting_distance"] == params["starting_distance"].values[0]
        ):
            return p["id"]
    raise ValueError("Parameter not found")


calendar_time_points = robot_poses["time"]
# io_robot_poses = format_io_poses(robot_poses, calendar_time_points, type="robot")
# io_fish_poses = format_io_poses(fish_poses, calendar_time_points, type="fish")

# np.save("io_robot_poses.npy", io_robot_poses)
# np.save("io_fish_poses.npy", io_fish_poses)

io_robot_poses = np.load(folder + "/io_robot_poses.npy")
io_fish_poses = np.load(folder + "/io_fish_poses.npy")

trial_params_index = 0
trial_inner_timer = 0
trial_start_time = calendar_time_points[0]
do_measure = False

relative_time_seconds = []
distances = []
follow_vectors = []
velocity_changes = []
acceleration_changes = []
states = []
robot_velocities = []

relative_time_batches = []
follow_vectors_batched = []
velocity_changes_batched = []
acceleration_changes_batched = []
robot_velocities_batched = []
batch = 25

results_id = None
previous_state = "MILLING"
init_counter = 0

# print(parameters)
# print(parameters.loc[[trial_index]])

# # print robot_states of INITIALIZING
# print(robot_states.loc[robot_states["state"] == "INITIALIZING"])

# print(np.unique(robot_states["state"]))

for k, time_point in enumerate(calendar_time_points):
    robot_state = robot_states.loc[robot_states["time"] == time_point]
    robot_state = robot_state["state"].iloc[0]
    if robot_state == "MILLING":
        continue

    # print(robot_state)

    if previous_state in ["APPROACHING_SLOW", "APPROACHING_FAST"] and robot_state == "INITIALIZING":
        init_counter += 1

    if previous_state == "INITIALIZING" and robot_state in ["APPROACHING_SLOW", "APPROACHING_FAST"]:
        if init_counter > 1:
            # Get the trial parameters
            params = parameters.loc[[trial_params_index]]
            results_id = get_dict_result(params)
            # print(time_point)
            # print(params)
            # Save the trial
            list_results[results_id]["relative_time_seconds"].append(relative_time_seconds)
            list_results[results_id]["distances"].append(distances)
            list_results[results_id]["follow_vectors"].append(follow_vectors)
            list_results[results_id]["velocity_changes"].append(velocity_changes)
            list_results[results_id]["acceleration_changes"].append(acceleration_changes)
            list_results[results_id]["states"].append(states)
            list_results[results_id]["robot_velocities"].append(robot_velocities)
            list_results[results_id]["relative_time_batches"].append(relative_time_batches)
            list_results[results_id]["follow_vectors_batched"].append(follow_vectors_batched)
            list_results[results_id]["velocity_changes_batched"].append(velocity_changes_batched)
            list_results[results_id]["acceleration_changes_batched"].append(acceleration_changes_batched)
            list_results[results_id]["robot_velocities_batched"].append(robot_velocities_batched)
            # Reset lists
            relative_time_seconds = []
            distances = []
            follow_vectors = []
            velocity_changes = []
            acceleration_changes = []
            states = []
            robot_velocities = []
            # Increment trial index
            trial_params_index += 1
        # Setup measurement of new trial
        trial_inner_timer = 0
        trial_start_time = time_point
        do_measure = True

    if do_measure:
        # Measure relative time
        rel_time = time_point - trial_start_time
        relative_time_seconds.append(rel_time.total_seconds())

        # Measure relative distance
        distance = np.linalg.norm(io_robot_poses[0, k, :2] - io_fish_poses[0, k, :2])
        distances.append(distance)

        # Measure fish direction relative to robot
        if trial_inner_timer > 0:
            f_v = io_fish_poses[0, k, :2] - io_fish_poses[0, k - 1, :2]
            b_p = io_robot_poses[0, k, :2] - io_fish_poses[0, k, :2]
            follow_vectors.append((f_v * b_p).sum())

        # Measure velocity change
        if trial_inner_timer > 0:
            velocity_change = (distances[-1] - distances[-2]) / (relative_time_seconds[-1] - relative_time_seconds[-2])
            velocity_changes.append(velocity_change)

        # Measure acceleration change
        if trial_inner_timer > 1:
            acceleration_change = (velocity_changes[-1] - velocity_changes[-2]) / (relative_time_seconds[-1] - relative_time_seconds[-2])
            acceleration_changes.append(acceleration_change)
        
        # Measure state
        states.append(robot_state)

        # Measure robot velocity
        if trial_inner_timer > 0:
            robot_velocity = np.linalg.norm(io_robot_poses[0, k, :2] - io_robot_poses[0, k - 1, :2]) / (relative_time_seconds[-1] - relative_time_seconds[-2])
            robot_velocities.append(robot_velocity)
        
        # Measure relative time in batches
        if trial_inner_timer % batch == 0:
            relative_time_batches.append(rel_time.total_seconds())
        
        # Measure fish direction relative to robot in batches
        if trial_inner_timer >= batch and trial_inner_timer % batch == 0:
            follow = calculate_follow(io_fish_poses[0, k - batch:k, :], io_robot_poses[0, k - batch:k, :])
            follow_vectors_batched.append(follow)
        
        # Measure velocity change in batches
        if trial_inner_timer >= batch and trial_inner_timer % batch == 0:
            velocity_change = np.linalg.norm(io_fish_poses[0, k, :2] - io_fish_poses[0, k - batch, :2]) / (relative_time_seconds[-1] - relative_time_seconds[-batch])
            velocity_changes_batched.append(velocity_change)
        
        # Measure acceleration change in batches
        if trial_inner_timer >= 2*batch and trial_inner_timer % batch == 0:
            acceleration_change = (velocity_changes_batched[-1] - velocity_changes_batched[-2]) / (relative_time_seconds[-1] - relative_time_seconds[-batch])
            acceleration_changes_batched.append(acceleration_change)
        
        # Measure robot velocity in batches
        if trial_inner_timer >= batch and trial_inner_timer % batch == 0:
            robot_velocity = np.linalg.norm(io_robot_poses[0, k, :2] - io_robot_poses[0, k - batch, :2]) / (relative_time_seconds[-1] - relative_time_seconds[-batch])
            robot_velocities_batched.append(robot_velocity)
        
        trial_inner_timer += 1
    
    previous_state = robot_state

print(list_results)

with open(folder + "/results.pkl", "wb") as f:
    pkl.dump(list_results, f)
