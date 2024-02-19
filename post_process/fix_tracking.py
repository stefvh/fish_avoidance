import datetime
import pickle

import numpy as np
from analyse_trials import get_valid_trials
from pre_process.experiment import Experiment

import matplotlib.pyplot as plt

trials = get_valid_trials()
valid_ids = [trial["id"] for trial in trials]

n_trials = 45
invalid_ids = [i for i in range(0, n_trials) if i not in valid_ids]

print(invalid_ids)

invalid_id = 31

with open("data/RoboFish_Commander_Data/JUNE 13 (SET 2)/20230613-165310/output/experiment.pkl", 'rb') as f:
    experiment = pickle.load(f)

trial = experiment.trials[invalid_id]

print(trial)

new_fish_poses = trial.fish_poses.copy()

print(new_fish_poses)

buffer_len = 10
buffer_pos = np.zeros((buffer_len, 2))
buffer_time_s = []

start_datetime = trial.start_datetime

max_valid_velocity = 15
valid_labels = [True]
for index, row in trial.fish_poses.iterrows():
    if index < buffer_len:
        buffer_pos[index] = np.array([row["x"], row["y"]])
        buffer_time_s.append((row["time"] - start_datetime).total_seconds())
        continue
    if index == buffer_len:
        b_v = buffer_pos[1:] - buffer_pos[:-1]
        b_d = np.linalg.norm(b_v, axis=1)
        b_velocity = b_d / np.diff(buffer_time_s)
        valid_labels.extend(b_velocity < max_valid_velocity)
        if not np.all(valid_labels):
            break
    new_pos = np.array([row["x"], row["y"]])
    new_v = new_pos - buffer_pos[-1]
    new_d = np.linalg.norm(new_v)
    new_t = (row["time"] - start_datetime).total_seconds()
    new_velocity = new_d / (new_t - buffer_time_s[-1])
    if new_velocity < max_valid_velocity:
        valid_labels.append(True)
        buffer_pos[:-1] = buffer_pos[1:]
        buffer_pos[-1] = new_pos
        buffer_time_s[:-1] = buffer_time_s[1:]
        buffer_time_s[-1] = new_t
    else:
        valid_labels.append(False)
    
new_fish_poses["valid"] = valid_labels
experiment_start_datetime = experiment.robot_poses["time"].iloc[0]
new_fish_poses["rel_time_s"] = new_fish_poses["time"].apply(lambda x: (x - experiment_start_datetime).total_seconds() / 60)
print(new_fish_poses)
print(new_fish_poses.loc[new_fish_poses["valid"] == False])

print(new_fish_poses.iloc[70:80])

print(experiment_start_datetime)
camera_start_datetime = datetime.datetime(2023, 6, 13, 16, 53, 9, 349405, tzinfo=datetime.timezone.utc)
print(camera_start_datetime)
experiment_end_datetime = experiment.robot_poses["time"].iloc[-1]
print(experiment_end_datetime)
print(camera_start_datetime - experiment_start_datetime)

# fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# ax = axes[0]
# # plot buffer quiver
# ax.quiver(buffer[:-1, 0], buffer[:-1, 1], np.cos(buffer[1:, 2]), np.sin(buffer[1:, 2]), scale=5)
# # plot positions and show chronolic order
# ax.plot(buffer[:, 0], buffer[:, 1], "ro")
# for i in range(buffer_len):
#     ax.text(buffer[i, 0], buffer[i, 1], f"{i}", fontsize=12)
# ax.set_xlabel(r"x")
# ax.set_ylabel(r"y")
# ax.set_xlim(67, 67.6)
# ax.set_ylim(31.4, 32.4)
# ax.set_title("Buffer trajectory")

# ax = axes[1]
# # plot initial quiver fish_poses of buffer_len
# ax.quiver(trial.fish_poses["x"][:buffer_len], trial.fish_poses["y"][:buffer_len], np.cos(trial.fish_poses["z"][:buffer_len]), np.sin(trial.fish_poses["z"][:buffer_len]), scale=5)
# # plot positions and show chronolic order
# ax.plot(trial.fish_poses["x"][:buffer_len], trial.fish_poses["y"][:buffer_len], "ro")
# for i in range(buffer_len):
#     ax.text(trial.fish_poses["x"][i], trial.fish_poses["y"][i], f"{i}", fontsize=12)
# ax.set_xlabel(r"x")
# ax.set_ylabel(r"y")
# ax.set_xlim(67, 67.6)
# ax.set_ylim(31.4, 32.4)

# plt.savefig("buffer_trajectory.pdf", bbox_inches="tight", dpi=300)


# for index, row in trial.fish_poses.iterrows():
#     if index < buffer_len:
#         buffer[index] = np.array([row["x"], row["y"], row["z"]])
#         continue
#     else:
#         if index == buffer_len:
#             # find if the fish made a turn based on the buffer
#             for i in range(1, buffer_len):
#                 b_v = buffer[i] - buffer[i-1]
#                 b_d = np.linalg.norm(b_v)
#                 b_d_max = max(b_d_max, b_d)

#     b_v = buffer[1:, :2] - buffer[:-1, :2]
#     b_d = np.linalg.norm(b_v, axis=1)
#     # print(b_d)
#     b_d_max = max(b_d_max, np.max(b_d))

#     # buffer[:-1] = buffer[1:]
#     # buffer[-1] = [row["x"], row["y"]]

#     # new_fish_poses.at[index, "x"] = np.mean(buffer[:, 0])
#     # new_fish_poses.at[index, "y"] = np.mean(buffer[:, 1])

#     # check if there any discripancies in the starting buffer
#     b_v = buffer[1:, :2] - buffer[:-1, :2]
#     b_d = np.linalg.norm(b_v, axis=1)
#     b_d_max_id = np.argmax(b_d)
#     # based on velocity
#     # if b_d[b_d_max_id] > 7.5:
#         # print("Discrepancy in the starting buffer")
#         # buffer = buffer[b_d_max_id:]
#         # buffer[-1] = np.array([row["x"], row["y"]])
#         # continue
# buffer[:-1] = buffer[1:]
# buffer[-1] = np.array([row["x"], row["y"]])