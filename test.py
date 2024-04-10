import pickle
import numpy as np

from pre_process.experiment import Experiment
from priors.foldering import get_data_folders

# a1 = np.array([5., 5.])
# a2 = np.array([6., 6.])
# a = [a1, a2]

# b1 = np.array([10., 7.])
# b2 = np.array([12., 8.])
# b = [b1, b2]

# mean_a = np.mean(a, axis=0)
# mean_b = np.mean(b, axis=0)

# mean_d = np.linalg.norm(mean_b - mean_a)

# full_d = 0
# for i in range(len(a)):
#     full_d += np.linalg.norm(b[i] - a[i])
# full_d /= len(a)

# print(mean_d)
# print(full_d)

# read experiment

# parent_folder = "June13_2"

# experiment_datetime = "20230613-165310"

# folder, exp_folder = get_data_folders(parent_folder, experiment_datetime)
# experiment = pickle.load(open(exp_folder + "/experiment.pkl", "rb"))

# # print(experiment.trials)

# print(experiment.robot_poses.shape)

import numpy as np
import matplotlib.pyplot as plt

# Example turn angle data (replace this with your actual data)
turn_angles = np.random.uniform(-180, 180, size=1000)

# Convert turn angles to radians
angles_rad = np.radians(turn_angles)

# Plot turn angles in polar coordinates
plt.figure(figsize=(8, 8))
ax = plt.subplot(111, projection='polar')
ax.hist(angles_rad, bins=50, color='skyblue', alpha=0.7)
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
plt.title('Turn Angle Distribution (Polar Plot)')
plt.show()