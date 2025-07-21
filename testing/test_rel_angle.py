import numpy as np
from matplotlib import pyplot as plt

# Fish and robot positions
fish_positions = np.array([
    [0, 0],
    [-3, 3]
])

robot1_positions = np.array([
    [2, 6],
    [-6, 2]
])

robot2_positions = np.array([
    [2, 3],
    [-3, 0]
])

# Function to compute cross product to check if vectors are parallel
def cross_product_2d(v1, v2):
    return v1[0] * v2[1] - v1[1] * v2[0]

# Function to compute the angle between two vectors
def angle_between_vectors(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    cos_theta = dot_product / (norm_v1 * norm_v2)
    angle_radians = np.arccos(cos_theta)
    angle_degrees = np.degrees(angle_radians)
    return angle_degrees

# Calculate direction vectors
v_F = fish_positions[1] - fish_positions[0]
v_R1 = robot1_positions[1] - robot1_positions[0]
v_R2 = robot2_positions[1] - robot2_positions[0]

# Check if robot1's trajectory is parallel to the fish's
cross_prod1 = cross_product_2d(v_F, v_R1)
if cross_prod1 != 0:
    angle1 = angle_between_vectors(v_F, v_R1)
    parallel1 = False
    print(f"The angle between the fish and robot1 trajectories is {angle1:.2f} degrees.")
else:
    parallel1 = True
    print("The fish and robot1 trajectories are parallel and will never intersect.")

# Check if robot2's trajectory is parallel to the fish's
cross_prod2 = cross_product_2d(v_F, v_R2)
if cross_prod2 != 0:
    angle2 = angle_between_vectors(v_F, v_R2)
    parallel2 = False
    print(f"The angle between the fish and robot2 trajectories is {angle2:.2f} degrees.")
else:
    parallel2 = True
    print("The fish and robot2 trajectories are parallel and will never intersect.")

# Plot the trajectories
plt.figure()
plt.plot([x[0] for x in fish_positions], [x[1] for x in fish_positions], 'r', label='Fish Trajectory')
plt.plot([x[0] for x in robot1_positions], [x[1] for x in robot1_positions], 'b', label='Robot1 Trajectory')
plt.plot([x[0] for x in robot2_positions], [x[1] for x in robot2_positions], 'g', label='Robot2 Trajectory')

# Add arrows to represent the direction vectors
plt.quiver(fish_positions[0, 0], fish_positions[0, 1], v_F[0], v_F[1], angles='xy', scale_units='xy', scale=1, color='r', label='Fish Direction')
plt.quiver(robot1_positions[0, 0], robot1_positions[0, 1], v_R1[0], v_R1[1], angles='xy', scale_units='xy', scale=1, color='b', label='Robot1 Direction')
plt.quiver(robot2_positions[0, 0], robot2_positions[0, 1], v_R2[0], v_R2[1], angles='xy', scale_units='xy', scale=1, color='g', label='Robot2 Direction')

# Add titles, labels, and legend
plt.title("Trajectories of Fish and Robots with Angle Annotations")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()

# Annotate the angles between the fish's and robots' trajectories
if not parallel1:
    mid_x1 = (fish_positions[1, 0] + robot1_positions[1, 0]) / 2
    mid_y1 = (fish_positions[1, 1] + robot1_positions[1, 1]) / 2
    plt.text(mid_x1, mid_y1, f'{angle1:.2f}°', color='b', fontsize=12)

if not parallel2:
    mid_x2 = (fish_positions[1, 0] + robot2_positions[1, 0]) / 2
    mid_y2 = (fish_positions[1, 1] + robot2_positions[1, 1]) / 2
    plt.text(mid_x2, mid_y2, f'{angle2:.2f}°', color='g', fontsize=12)

# Modified function to compute the signed angle between two vectors
def signed_angle_between_vectors(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    cos_theta = dot_product / (norm_v1 * norm_v2)
    angle_radians = np.arccos(cos_theta)
    angle_degrees = np.degrees(angle_radians)

    # Use cross product to determine if angle should be negative or positive
    cross_prod = np.cross(v1, v2)
    print(cross_prod)
    if cross_prod < 0:
        angle_degrees = -angle_degrees  # Adjust sign based on relative direction
    return angle_degrees

# Calculate the signed angles instead of absolute angles
if cross_prod1 != 0:
    angle1 = signed_angle_between_vectors(v_F, v_R1)
    parallel1 = False
    print(f"The signed angle between the fish and robot1 trajectories is {angle1:.2f} degrees.")

if cross_prod2 != 0:
    angle2 = signed_angle_between_vectors(v_F, v_R2)
    parallel2 = False
    print(f"The signed angle between the fish and robot2 trajectories is {angle2:.2f} degrees.")

# Modify annotation logic to display positive/negative angles accordingly
if not parallel1:
    mid_x1 = (fish_positions[1, 0] + robot1_positions[1, 0]) / 2
    mid_y1 = (fish_positions[1, 1] + robot1_positions[1, 1]) / 2
    plt.text(mid_x1, mid_y1, f'{angle1:.2f}°', color='b', fontsize=12)

if not parallel2:
    mid_x2 = (fish_positions[1, 0] + robot2_positions[1, 0]) / 2
    mid_y2 = (fish_positions[1, 1] + robot2_positions[1, 1]) / 2
    plt.text(mid_x2, mid_y2, f'{angle2:.2f}°', color='g', fontsize=12)

# plot line perpendicular to the fish trajectory
slope = v_F[1] / v_F[0]
perpendicular_slope = -1 / slope
x_vals = np.linspace(-10, 10, 100)
y_vals = perpendicular_slope * x_vals
# move through the end fish position
y_vals = y_vals + fish_positions[1, 1] - perpendicular_slope * fish_positions[1, 0]
plt.plot(x_vals, y_vals, 'r--')

def get_avoidance_speed(fish_position, robot_position, fish_velocity):
    # Relative position
    x_fr = fish_position[0] - robot_position[0]
    y_fr = fish_position[1] - robot_position[1]

    # Calculate distance
    distance = np.sqrt(x_fr ** 2 + y_fr ** 2)

    # Normalize relative positions
    x_fr_n = x_fr / distance
    y_fr_n = y_fr / distance

    # Calculate fish retreat velocity (projection of fish's velocity onto the relative position vector)
    rev = x_fr * x_fr_n + y_fr * y_fr_n
    cos_rev = rev / np.linalg.norm(fish_velocity)

    return rev, cos_rev

avo1 = get_avoidance_speed(fish_positions[0], robot1_positions[0], v_F)
avo2 = get_avoidance_speed(fish_positions[0], robot2_positions[0], v_F)

# Add annotations for avoidance speed
offsets = [0.5, -0.5]
plt.text(robot1_positions[1, 0] + offsets[0], robot1_positions[1, 1] + offsets[1], f'{avo1[0]:.2f}', color='b', fontsize=12)
plt.text(robot2_positions[1, 0] + offsets[0], robot2_positions[1, 1] + offsets[1], f'{avo2[0]:.2f}', color='g', fontsize=12)


# Save the plot
plt.savefig('testing/test_rel_angle_with_annotations.pdf')
