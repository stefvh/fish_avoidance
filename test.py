import numpy as np
import matplotlib.pyplot as plt


import numpy as np
import matplotlib.pyplot as plt

def compute_transition_speed(
    fish_distance,
    transition_distance=30,
    sigmoid_slope=0.5,
    sigmoid_cutoff=35,
    slow_speed_cm_s=15,
    fast_speed_cm_s=30,
):
    # Apply sigmoid function to transition speed smoothly
    # The transition happens between 30cm (higher speed) and 40cm (lower speed)
    x = transition_distance - min(transition_distance, fish_distance) - sigmoid_cutoff
    sigmoid = 1.0 / (1.0 + np.exp(-sigmoid_slope * x))
    
    # Smooth transition of speed between slow_speed_cm_s and fast_speed_cm_s
    speed = slow_speed_cm_s + (fast_speed_cm_s - slow_speed_cm_s) * sigmoid
    return speed

# Generate fish distances from 30 to 40 cm (adjusting the range to ensure the transition from 15 cm/s at 40 cm)
fish_distances = np.linspace(0, 40, 100)

# Compute speeds for the transition
speeds_slope_05 = [
    compute_transition_speed(d, slow_speed_cm_s=15, fast_speed_cm_s=30)  # Transition between 15cm/s and 30cm/s
    for d in fish_distances
]
speeds_slope_10 = [
    compute_transition_speed(d, slow_speed_cm_s=15, fast_speed_cm_s=20)  # Transition between 15cm/s and 20cm/s
    for d in fish_distances
]

# # Plot the results
# plt.figure(figsize=(10, 6))
# plt.plot(fish_distances, speeds_slope_05, label='Sigmoid Slope = 0.5, 15 cm/s to 30 cm/s')
# plt.plot(fish_distances, speeds_slope_10, label='Sigmoid Slope = 10, 15 cm/s to 20 cm/s')
# plt.xlabel('Fish Distance (cm)')
# plt.ylabel('Speed (cm/s)')
# plt.title('Transition Speed with Different Sigmoid Slopes and Cutoffs')
# plt.legend()
# plt.grid(True)



import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x, a=10, b=20):
    x = 35 - x
    return 15 + (a - 15) / (1 + np.exp(-(x - b)))

def speed_at_distance(x, high_speed=30):
    # Ensure the speed is 15 cm/s at distances greater than or equal to 30 cm
    if x >= 30:
        return 15
    
    # Use the sigmoid function to gradually increase speed from 15 cm/s to high_speed
    return sigmoid(x, high_speed, 10)

# Create an array of distances from 0 to 40 cm
distances = np.linspace(0, 40, 400)

# Calculate the corresponding speeds for the distances
speeds = [speed_at_distance(d, high_speed=25) for d in distances]

# Plot the results
plt.plot(distances, speeds, label="Speed vs Distance")
plt.xlabel("Distance (cm)")
plt.ylabel("Speed (cm/s)")
plt.title("Speed vs Distance Plot")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig("test.pdf")
