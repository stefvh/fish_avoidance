import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, Voronoi, voronoi_plot_2d

# Parameters
num_time_steps = 300
tank_size = 1.0  # 1m x 1m tank

# Generate curved trajectory for robot (circular motion)
t = np.linspace(0, 4 * np.pi, num_time_steps)
robot_x = 0.5 + 0.3 * np.cos(t)
robot_y = 0.5 + 0.3 * np.sin(t)

# Generate random movement for fish
np.random.seed(0)
fish_x = np.random.uniform(0, tank_size, num_time_steps)
fish_y = np.random.uniform(0, tank_size, num_time_steps)

# Time windows for recent and projected trajectories
T_recent = 60  # 2 seconds at 30 Hz
T_project = 30  # 1 second at 30 Hz

# Function to get recent and projected trajectory points
def get_trajectory_points(time, recent_window, project_window, x, y):
    start_recent = max(0, time - recent_window)
    end_project = min(len(x), time + project_window)
    recent_points = np.array(list(zip(x[start_recent:time], y[start_recent:time])))
    project_points = np.array(list(zip(x[time:end_project], y[time:end_project])))
    return recent_points, project_points

# Get recent and projected trajectories at a specific time
current_time = 150
recent_points, project_points = get_trajectory_points(current_time, T_recent, T_project, robot_x, robot_y)

# Combine recent and projected points
trajectory_points = np.vstack((recent_points, project_points))

# Compute Convex Hull
hull = ConvexHull(trajectory_points)

# Compute Voronoi Diagram
points = np.column_stack((robot_x, robot_y))
vor = Voronoi(points)

# Get the Voronoi region corresponding to the current robot position
region_index = vor.point_region[current_time]
region = vor.regions[region_index]

# Filter out infinite regions and construct the polygon
if -1 in region:
    region = [i for i in region if i != -1]

voronoi_polygon = np.array([vor.vertices[i] for i in region])

# Plotting the results
plt.figure(figsize=(10, 10))
plt.plot(robot_x, robot_y, label='Robot Trajectory', color='blue')
plt.plot(fish_x, fish_y, label='Fish Trajectory', color='green')
plt.scatter(robot_x[current_time], robot_y[current_time], color='red', label='Current Robot Position')
plt.plot(trajectory_points[:, 0], trajectory_points[:, 1], 'ko', markersize=2)
plt.fill(trajectory_points[hull.vertices, 0], trajectory_points[hull.vertices, 1], 'orange', alpha=0.3, label='Convex Hull')

# Voronoi region
plt.plot(voronoi_polygon[:, 0], voronoi_polygon[:, 1], 'r--', lw=2, label='Voronoi Cell')
plt.fill(voronoi_polygon[:, 0], voronoi_polygon[:, 1], 'red', alpha=0.2)

plt.xlim(0, tank_size)
plt.ylim(0, tank_size)
plt.legend()
plt.title('Trajectories and Space Partitioning')
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.grid(True)
plt.savefig('trajectory_partitioning.pdf', dpi=300)
