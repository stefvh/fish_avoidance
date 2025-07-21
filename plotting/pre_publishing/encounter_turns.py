from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from core.experiment import Experiment
from core.study import study

from analysis.full.by_trials import get_trials

import utils.plotter as plotter

from matplotlib.patches import FancyArrow

def find_frame_turn(data, frame_encounter, turn_threshold):
    turn_fish = data['turn_fish'].to_numpy()
    """Find the nearest earlier frame where fish makes a significant turn."""
    for i in range(frame_encounter - 5, -1, -1):  # Go backward from frame_encounter
        if abs(turn_fish[i]) > turn_threshold:
            return i
    return None  # No significant turn found before encounter

def rotate_to_fish_frame(fish_heading, x, y):
    """Rotate (x, y) coordinates such that fish is at origin and heading to the right (0 radians)."""
    cos_theta = np.cos(-fish_heading)
    sin_theta = np.sin(-fish_heading)
    x_rotated = x * cos_theta - y * sin_theta
    y_rotated = x * sin_theta + y * cos_theta
    return x_rotated, y_rotated

def foo(exp_trials):
    turn_threshold = np.pi / 6

    # Initialize a list to store the significant turns (turn_fish values at frame_turn)
    turn_fish_values_at_frame_turn = []

    # Loop through trials to find frame_turn and collect the corresponding turn_fish values
    for trials in exp_trials:
        for trial in trials:
            data = trial["data"]
            distances = data["distance"].to_numpy()
            ef = np.argmin(distances)
            td = 10
            frame_encounter = ef + td + 1
            frame_turn = find_frame_turn(data, frame_encounter, turn_threshold)
            if frame_turn is not None:
                np_turn_fish = data['turn_fish'].to_numpy()
                turn_fish_values_at_frame_turn.append(abs(np_turn_fish[frame_turn]))

    # Calculate the maximum turn_fish from the collected significant turns
    max_turn_fish = max(turn_fish_values_at_frame_turn) if turn_fish_values_at_frame_turn else 1  # Handle empty case

    # Prepare the plot
    fig, ax = plt.subplots()

    # Create a colormap normalization based on the max_turn_fish
    norm = plt.Normalize(vmin=0, vmax=max_turn_fish)

    # Loop through trials of the subject
    for exp_id, trials in enumerate(exp_trials):
        for trial_id, trial in enumerate(trials):
            data = trial["data"]
            distances = data["distance"].to_numpy()
            ef = np.argmin(distances)
            td = 10
            frame_encounter = ef + td + 1
            
            # Find the significant turn frame
            frame_turn = find_frame_turn(data, frame_encounter, turn_threshold)
            if frame_turn is None:
                continue  # Skip if no significant turn is found before the encounter
            # Manually remove because this was with manual changes
            if exp_id == 2 and trial["m_trial_id"] == 10:
                continue
            # Fish was not moving, incorrect orientation/turns
            if exp_id == 3 and trial["m_trial_id"] == 5:
                continue

            # Get the fish and robot positions and orientations at frame_turn
            np_heading_fish = data['heading_fish'].to_numpy()
            np_heading_robot = data['heading_robot'].to_numpy()
            np_speed_robot = data['speed_robot'].to_numpy()
            np_x_rel = data['x_robot'].to_numpy() - data['x_fish'].to_numpy()
            np_y_rel = data['y_robot'].to_numpy() - data['y_fish'].to_numpy()

            fish_heading = np_heading_fish[frame_turn]
            robot_heading = np_heading_robot[frame_turn]
            robot_speed = np_speed_robot[frame_turn]
            
            # Calculate the relative robot position to the fish
            x_rel = np_x_rel[frame_turn]
            y_rel = np_y_rel[frame_turn]
            
            # Rotate the relative position to align with the fish's orientation
            x_rel_rotated, y_rel_rotated = rotate_to_fish_frame(fish_heading, x_rel, y_rel)
            
            # Calculate the robot's orientation relative to the fish
            relative_robot_heading = robot_heading - fish_heading

            # Normalize the robot's heading (make sure it's between 0 and 2pi)
            relative_robot_heading = (relative_robot_heading + 2 * np.pi) % (2 * np.pi)
            
            # Plot the robot's relative position with an arrow indicating its orientation
            np_turns_fish = data['turn_fish'].to_numpy()
            arrow_length = 2.5  # Adjust the length of the arrow
            ax.add_patch(FancyArrow(x_rel_rotated, y_rel_rotated,
                                    arrow_length * np.cos(relative_robot_heading),
                                    arrow_length * np.sin(relative_robot_heading),
                                    width=0.5, color=plt.cm.viridis(norm(abs(np_turns_fish[frame_turn])))))
        

            # Add text annotation with frame difference and robot's speed
            frame_diff = frame_encounter - frame_turn
            identifier = exp_id  # Get the subject's identifier
            trial_identifier = trial["m_trial_id"] # Get the trial's identifier
            # ax.text(x_rel_rotated, y_rel_rotated, f"{identifier},{trial_identifier},{frame_diff}", fontsize=6)
            # ax.text(x_rel_rotated, y_rel_rotated, f"F={frame_diff}\nS={robot_speed:.2f}\nID={identifier}", fontsize=8)

    # Add colorbar based on the magnitude of turn_fish
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])  # This is necessary for the colorbar
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Turn Magnitude (radians)')

    # Set plot labels and limits
    ax.set_xlabel('Relative X Distance to Robot (cm)')
    ax.set_ylabel('Relative Y Distance to Robot (cm)')
    ax.set_title('Relative Robot Position and Orientation during Fish Turns')
    ax.set_xlim([-40, 40])
    ax.set_ylim([-40, 40])
    # ax.set_aspect('equal')

    fig.savefig(f'{study.plot_output_folder}/TEST-PUBLISH/encounter_turns.pdf')

if __name__ == "__main__":
    exp_trials = get_trials()

    foo(exp_trials)