import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.experiment import Experiment, get_experiment

from plotting.general import sorting
from pre_process import frequency_conversion
from utils.maths import geometry
from utils.parser import get_default_parser
from utils.plotter import plot_multipdf

import post_process.predictive_fish_movement as fish_models

from scipy.signal import find_peaks, peak_prominences


def parse_args():
    parser = get_default_parser()

    parser.add_argument("--renew_pred", default=False, action="store_true")

    return parser.parse_args()


def annotate_row(ax, trial_id, min_robot_distance):
    x_textpad = -0.45
    annotate_color = "black"

    annotation_text = r"(" + f"{str(trial_id)}) "
    annotation_text += r"$d_{\min}=$" + f"{min_robot_distance:.1f}cm"

    ax.annotate(
        annotation_text,
        xy=(x_textpad, 0.45),
        xycoords="axes fraction",
        va="center",
        rotation="vertical",
        fontsize=12,
        weight="bold",
        color=annotate_color,
    )


def plot_png_predictions(
    robot_poses,
    fish_poses,
    batch_robot_directions,
    batch_fish_directions,
    p_speeds,
    p_turns,
):
    output_folder = experiment.input_folder + "/plots"
    sub_folder = "observation/predictions"
    if not os.path.exists(f"{output_folder}/{sub_folder}"):
        os.makedirs(f"{output_folder}/{sub_folder}")

    color_robot = "tab:orange"
    color_fish = "tab:blue"

    step = 5

    for rel_fr_nr in range(0, fish_poses.shape[0] - step, step):
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        ax.scatter(
            robot_poses[rel_fr_nr, 0], robot_poses[rel_fr_nr, 1], color=color_robot
        )
        ax.scatter(fish_poses[rel_fr_nr, 0], fish_poses[rel_fr_nr, 1], color=color_fish)

        # quiver robot direction
        r_direction = batch_robot_directions[rel_fr_nr]
        ax.quiver(
            robot_poses[rel_fr_nr, 0],
            robot_poses[rel_fr_nr, 1],
            r_direction[0],
            r_direction[1],
            color=color_robot,
        )

        # quiver real direction
        f_direction = batch_fish_directions[rel_fr_nr]
        ax.quiver(
            fish_poses[rel_fr_nr, 0],
            fish_poses[rel_fr_nr, 1],
            f_direction[0],
            f_direction[1],
            color=color_fish,
        )
        # quiver predictions
        p_index = rel_fr_nr // step
        speeds = p_speeds[p_index]
        turns = p_turns[p_index]
        for speed, turn in zip(speeds, turns):
            f_pose = fish_poses[rel_fr_nr]
            f_angle = f_pose[2] + turn
            f_direction = np.array([np.cos(f_angle), np.sin(f_angle)]) * speed
            ax.quiver(
                fish_poses[rel_fr_nr, 0],
                fish_poses[rel_fr_nr, 1],
                f_direction[0],
                f_direction[1],
                color="forestgreen",
            )

        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)

        plt.savefig(f"{output_folder}/{sub_folder}/prediction_{rel_fr_nr}.png")


def plot_trial(ax_row, i, j, trial_id, results, pred_speeds, pred_turns):
    k = i + j
    trial_results = results[results["trial_id"] == trial_id]

    relative_frame_numbers = (
        trial_results["frame_number"] - trial_results["frame_number"].min()
    )

    min_robot_distance = trial_results["robot_distance"].min()
    argmin_robot_distance = trial_results["robot_distance"].idxmin()
    x_rel_min = trial_results["frame_number"][argmin_robot_distance] - trial_results["frame_number"].min()
    for ax in ax_row:
        ax.axvline(x=x_rel_min, color="red", linestyle="-")

    column = 0

    # Plot robot distance
    ax = ax_row[column]
    ax.plot(relative_frame_numbers, trial_results["robot_distance_batch"])
    if j == 0:
        ax.set_title("Robot Distance")
    ax.set_xlabel("Frame Number")
    ax.set_ylabel("Distance (cm)")

    max_robot_distance = results["robot_distance"].max()
    ax.set_ylim(0, max_robot_distance)

    ax.axhline(y=2, color="black", linestyle="--")
    ax.axhline(y=8, color="black", linestyle="--")

    annotate_row(ax, trial_results["trial_id"].iloc[0], min_robot_distance)

    column += 1

    # Plot MSE of speed and orientation predictions
    ax_speed = ax_row[column] 
    ax_orientation = ax_row[column + 1]

    if j == 0:
        ax_speed.set_title("(R)MSE Pred. Speed")
        ax_orientation.set_title("(R)MSE Pred. Orientation")
    
    if trial_id in pred_speeds:
        trial_pred_speeds = pred_speeds[trial_id]
        trial_pred_turns = pred_turns[trial_id]

        p_speeds = np.array(trial_pred_speeds)
        p_turns = np.array(trial_pred_turns)

        _, trial_fish_poses = experiment.get_poses_of_trial(trial_id)
        batch_size = 5
        batch_fish_poses, batch_fish_directions = (
            frequency_conversion.batch_positions_directions(batch_size, trial_fish_poses)
        )

        y_speed = []
        y_orientation = []
        step = 5
        for rel_fr_nr in range(0, trial_fish_poses.shape[0] - step, step):
            f_pose = batch_fish_poses[rel_fr_nr]
            f_direction = batch_fish_directions[rel_fr_nr]
            real_speed = np.linalg.norm(f_direction)
            real_orientation = np.arctan2(f_direction[1], f_direction[0])

            p_index = rel_fr_nr // step
            predicted_speeds = p_speeds[p_index]
            predicted_turns = p_turns[p_index]
            predicted_orientations = []
            for turn in predicted_turns:
                predicted_orientations.append(f_pose[2] + turn)
            
            mse_speed = np.mean((predicted_speeds - real_speed) ** 2)
            se_orientation = []
            for predicted_orientation in predicted_orientations:
                err = geometry.diff_angles(predicted_orientation, real_orientation)
                se_orientation.append(err ** 2)
            mse_orientation = np.mean(se_orientation)

            y_speed.append(mse_speed)
            y_orientation.append(mse_orientation)
        
        ax_speed.plot(np.arange(0, len(y_speed) * step, step), y_speed)
        ax_speed.plot(np.arange(0, len(y_speed) * step, step), np.sqrt(y_speed))
        ax_orientation.plot(np.arange(0, len(y_orientation) * step, step), y_orientation)
        ax_orientation.plot(np.arange(0, len(y_orientation) * step, step), np.sqrt(y_orientation))

    column += 2

    # Plot fish velocity
    ax = ax_row[column]
    ax.plot(relative_frame_numbers, trial_results["fish_velocity_batch"])
    if j == 0:
        ax.set_title("Fish Velocity")
    ax.set_xlabel("Frame Number")

    max_fish_velocity = results["fish_velocity_batch"].max()
    ax.set_ylim(0, max_fish_velocity)

    column += 1

    # Plot normalized fish velocity and MSE of speed predictions
    ax = ax_row[column]
    y = trial_results["fish_velocity_batch"] / trial_results["fish_velocity_batch"].max()
    ax.plot(relative_frame_numbers, y, linewidth=0.90)
    if j ==0:
        ax.set_title("Norm. Fish Vel. and Pred. Error")
    ax.set_xlabel("Frame Number")
    ax.set_ylabel("Norm. Velocity ", color="tab:blue")
    ax.set_ylim(0, 1.05)

    ax2 = ax.twinx()
    if trial_id in pred_speeds:
        y_normaled_speed = np.array(y_speed) / np.nanmax(y_speed)
        ax2.plot(np.arange(0, len(y_speed) * step, step), y_normaled_speed, color="forestgreen", linewidth=0.75)
        ax2.set_ylabel("Norm. MSE", color="forestgreen")
        ax2.set_ylim(0, 1.05)

        fps = experiment.frequency
        peaks, _ = find_peaks(y_normaled_speed, height=0.5)
        indices = np.arange(0, len(y_speed) * step, step)
        peak_indices = indices[peaks]
        ax2.plot(peak_indices, y_normaled_speed[peaks], "x", color="orange", linewidth=1.0)

    column += 1


def plot_trajectories(ax_row, i, j, trial_id, results, pred_speeds, pred_turns):
    column = 0
    color_fish = "blue"

    # # Trajectories of robot and fish
    ax = ax_row[column]
    if j == 0:
        ax.set_title("Trajectories")
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")

    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_aspect("equal")
    if trial_id in pred_speeds:
        trial_pred_speeds = pred_speeds[trial_id]
        trial_pred_turns = pred_turns[trial_id]

        p_speeds = np.array(trial_pred_speeds)
        p_turns = np.array(trial_pred_turns)

        _, trial_fish_poses = experiment.get_poses_of_trial(trial_id)
        batch_size = 5
        batch_fish_poses, batch_fish_directions = (
            frequency_conversion.batch_positions_directions(batch_size, trial_fish_poses)
        )

        # ax.plot(trial_robot_poses[:, 0], trial_robot_poses[:, 1], color=color_robot)
        # ax.plot(trial_fish_poses[:, 0], trial_fish_poses[:, 1], color=color_fish)

        step = 50
        for rel_fr_nr in range(0, trial_fish_poses.shape[0] - step, step):
            # quiver real direction
            f_direction = batch_fish_directions[rel_fr_nr]
            ax.quiver(
                trial_fish_poses[rel_fr_nr, 0],
                trial_fish_poses[rel_fr_nr, 1],
                f_direction[0],
                f_direction[1],
                color=color_fish,
                scale=10,
            )
            # quiver predictions
            p_index = rel_fr_nr // step
            speeds = p_speeds[p_index]
            turns = p_turns[p_index]
            for speed, turn in zip(speeds, turns):
                f_pose = trial_fish_poses[rel_fr_nr]
                f_angle = f_pose[2] + turn
                f_direction = np.array([np.cos(f_angle), np.sin(f_angle)]) * speed
                ax.quiver(
                    trial_fish_poses[rel_fr_nr, 0],
                    trial_fish_poses[rel_fr_nr, 1],
                    f_direction[0],
                    f_direction[1],
                    color="forestgreen",
                    scale=10,
                )

        # plot_png_predictions(
        #     batch_robot_poses,
        #     batch_fish_poses,
        #     batch_robot_directions,
        #     batch_fish_directions,
        #     trial_pred_speeds,
        #     trial_pred_turns,
        # )



def generate_trial_predictions(trial_id):
    l_speeds, l_turns = [], []

    _, trial_fish_poses = experiment.get_poses_of_trial(trial_id)
    batch_size = 5
    batch_fish_poses, batch_fish_directions = (
        frequency_conversion.batch_positions_directions(batch_size, trial_fish_poses)
    )

    step = 5
    for frame_number in range(0, trial_fish_poses.shape[0] - step, step):
        fish_pose = batch_fish_poses[frame_number]
        fish_direction = batch_fish_directions[frame_number]
        speeds, turns = [], []
        n_seeds = 10
        for _ in range(n_seeds):
            speed, turn = fish_models.run(
                model_name="ClusterModel",
                fish_pose=fish_pose,
                fish_direction=fish_direction,
                freq=25 / step,
            )
            speeds.append(speed)
            turns.append(turn)
        l_speeds.append(speeds)
        l_turns.append(turns)

    return l_speeds, l_turns


def run(
    experiment: Experiment,
    renew_pred: bool = False,
):
    results = experiment.results
    trial_ids = sorting.sort_trials_by_distance(results)

    output_folder = experiment.input_folder + "/plots"
    sub_folder = "observation"
    output_filename = "test_prediction_variance.pdf"
    if not os.path.exists(f"{output_folder}/{sub_folder}"):
        os.makedirs(f"{output_folder}/{sub_folder}")

    # generate predictions and save or load them
    if renew_pred or not os.path.exists(
        f"{output_folder}/{sub_folder}/predictions_speed.pkl"
    ):
        print("Generating predictions...")
        pred_speeds, pred_turns = {}, {}
        for trial_id in trial_ids:
            l_speeds, l_turns = generate_trial_predictions(trial_id)
            pred_speeds[trial_id] = l_speeds
            pred_turns[trial_id] = l_turns
        pickle.dump(
            pred_speeds,
            open(f"{output_folder}/{sub_folder}/predictions_speed.pkl", "wb"),
        )
        pickle.dump(
            pred_turns, open(f"{output_folder}/{sub_folder}/predictions_turn.pkl", "wb")
        )
    else:
        pred_speeds = pickle.load(
            open(f"{output_folder}/{sub_folder}/predictions_speed.pkl", "rb")
        )
        pred_turns = pickle.load(
            open(f"{output_folder}/{sub_folder}/predictions_turn.pkl", "rb")
        )

    plot_multipdf(
        len(trial_ids),
        5,
        5,
        lambda ax_row, i, j: plot_trial(
            ax_row=ax_row,
            i=i,
            j=j,
            trial_id=trial_ids[i + j],
            results=results,
            pred_speeds=pred_speeds,
            pred_turns=pred_turns,
        ),
        f"{output_folder}/{sub_folder}/{output_filename}",
    )


if __name__ == "__main__":
    args = parse_args()
    variables = vars(args)

    experiment = get_experiment(**variables)
    run(experiment, variables["renew_pred"])
