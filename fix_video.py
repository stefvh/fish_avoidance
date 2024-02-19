import pickle as pkl
import pandas as pd

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

# print(parameters['time'])

init_counter = 0
trial_params_index = 0
calendar_time_points = robot_poses["time"]
list_trial_times = []

previous_state = "MILLING"

date_zero = robot_states["time"].iloc[0]

for k, time_point in enumerate(calendar_time_points):
    robot_state = robot_states.loc[robot_states["time"] == time_point]
    robot_state = robot_state["state"].iloc[0]
    if robot_state == "MILLING":
        continue

    if previous_state == "MILLING" and robot_state == "INITIALIZING":
        init_counter += 1

    if previous_state == "INITIALIZING" and robot_state in ["APPROACHING_SLOW", "APPROACHING_FAST"]:
        if init_counter > 0:
            list_trial_times.append(time_point)
    
    previous_state = robot_state

start_times = []
for start_time in list_trial_times:
    print(start_time - date_zero)
    start_times.append(start_time - date_zero)
init_times = []
for init_time in parameters['time']:
    print(init_time - date_zero)
    init_times.append(init_time - date_zero)

d = {
    "id": list(range(0, len(start_times))),
    "start_times": start_times,
    "init_times": init_times,
}
df_trial_times = pd.DataFrame.from_dict(d,orient='index').transpose()
print(df_trial_times)