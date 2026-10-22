import numpy as np

def compute_transition_speed(fish_distance,
                             transition_distance,
                             slow_speed_cm_s,
                             fast_speed_cm_s,
                             sigmoid_slope,
                             sigmoid_cutoff=-0.5,
                             ):
    # A sigmoid function has its minimum approximately at -6 and maximum at +6
    # The transition of slow->fast should happen during the entirety of the "transition_distance" (e.g. 30cm)
    # Note however that the maximum can be reached sooner (based on the slope) (e.g. at 15cm)
    # x = 12 * ((self.transition_distance - min(self.transition_distance, fish_distance)) / self.transition_distance) - 6
    x = transition_distance - \
        min(transition_distance, fish_distance) - sigmoid_cutoff
    # Transition from slow to fast by sigmoid -> output in [0, 1]
    sigmoid = 1.0 / (1.0 + np.exp(-sigmoid_slope * x))
    # Adjust the value to the range of lowest and highest speeds -> output in [slow, fast]
    speed = slow_speed_cm_s + \
        (fast_speed_cm_s - slow_speed_cm_s) * sigmoid
    return speed
