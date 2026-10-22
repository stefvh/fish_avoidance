# 165310:
# 3210 -> 3320

# 165310:
# 11620 -> 11730

# 145122:
# 6710 -> 6752

from matplotlib import pyplot as plt
import numpy as np
from core.experiment import Experiment, get_experiment_by_id
from scipy.spatial import ConvexHull

def get_data():
    # 145122:
    # 6654 -> 6690
    experiment = get_experiment_by_id(4)

    robot_poses = experiment.robot_poses
    fish_poses = experiment.fish_poses

    start_frame = 6710 # 6654
    end_frame = 6752 # 6690

    robot_poses = robot_poses[start_frame:end_frame]
    fish_poses = fish_poses[start_frame:end_frame]

    return robot_poses, fish_poses


robot_poses, fish_poses = get_data()
robot_positions = robot_poses[:, :2]
fish_positions = fish_poses[:, :2]
time_point = int(robot_poses.shape[0] / 2)
fish_position = fish_poses[time_point, :2]

n_frames = robot_positions.shape[0]
generators = robot_positions.copy()
generators = generators[:time_point]
generators = np.append(generators, [fish_position], axis=0)
qhull_option = f'QG{generators.shape[0] - 1}'

hull = ConvexHull(points=generators,
                  qhull_options=qhull_option)

# Plot robot_positions and fish_position
plt.figure(figsize=(10, 10))

cmap = plt.cm.get_cmap('Blues')
alphas = np.linspace(0, 1, n_frames)
for i in range(n_frames - 1):
    plt.plot(robot_positions[i:i+2, 0], robot_positions[i:i+2, 1], color=cmap(alphas[i]), lw=4)
# plt.plot(robot_positions[:, 0], robot_positions[:, 1], cmap=cmap, label='Robot Trajectory', color='blue')

plt.scatter(fish_position[0], fish_position[1], color='orange', label='Fish Position', zorder=10, s=60)

cmap = plt.cm.get_cmap('Greens')
for i in range(n_frames - 1):
    plt.plot(fish_positions[i:i+2, 0], fish_positions[i:i+2, 1], color=cmap(alphas[i]), lw=4)
# plt.plot(fish_positions[:, 0], fish_positions[:, 1], label='Fish Trajectory', color='green')

for visible_facet in hull.simplices[hull.good]:
    plt.plot(hull.points[visible_facet, 0],
            hull.points[visible_facet, 1],
            color='violet',
            lw=6)
# plt.plot(generators[hull.vertices,0], generators[hull.vertices,1], 'r-', lw=2, zorder=10)
# # plot the final linesegment
# plt.plot([generators[hull.vertices[-1],0], generators[hull.vertices[0],0]],
#         [generators[hull.vertices[-1],1], generators[hull.vertices[0],1]],
#         'r-', lw=2, zorder=10)
plt.fill(generators[hull.vertices, 0], generators[hull.vertices, 1], 'orange', alpha=0.3, label='Convex Hull')

# add blue to legend
plt.plot([], [], color='blue', label='Robot Trajectory')
# add green to legend
plt.plot([], [], color='green', label='Fish Trajectory')
plt.legend()


plt.savefig('trajectory_partitioning_2.pdf', dpi=300)