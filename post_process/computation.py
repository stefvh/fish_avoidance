import os

import numpy as np
import pandas as pd
import utils.maths.geometry as geometry

from pre_process.environment import Environment
import post_process.scenarios as scenarios
import post_process.metrics as metrics


# def n_step_nan_check(arr : np.ndarray, i: int, n_step : int):
#     b = np.isnan(arr[i + n_step]).any()
#     return b, n_step


def n_step_nan_check(arr: np.ndarray, i: int, n_step: int):
    """
    Check if there are any index n steps ahead that are not nan
    """
    n = n_step
    if n > 0:
        while i + n < len(arr) and n > 0:
            if not np.isnan(arr[i + n]).any():
                break
            n -= 1
    else:
        while i + n >= 0 and n < 0:
            if not np.isnan(arr[i + n]).any():
                break
            n += 1
    return n == 0, n


def compute_results(
    trials,
    fish_poses,
    robot_poses,
    period,
    wall_corners,
    output_folder,
    output_filename="results",
):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    environment = Environment(wall_corners)

    n_steps = [1, 5]

    results = {
        "trial_id": [],
        "frame_number": [],
        "intra_trial_frame_number": [],
        "robot_distance": [],
        "wall_distances": [],
        "n_obstructed_walls": [],
        "n_interferred_walls": [],
        "is_robot_chasing": [],
    }
    for n_step in n_steps:
        results[f"fish_velocity_n={n_step}"] = []
        results[f"fish_turn_n={n_step}"] = []
        results[f"robot_velocity_n={n_step}"] = []
        results[f"robot_acceleration_n={n_step}"] = []
        results[f"robot_angle_diff_n={n_step}"] = []
        results[f"robot_walls_alignments_n={n_step}"] = []
    results.update(
        {
            "robot_distance_batch": [],
            "fish_velocity_batch": [],
            "fish_acceleration_batch": [],
            "fish_turn_batch": [],
            "fish_direction_batch": [],
            "robot_velocity_batch": [],
            "robot_acceleration_batch": [],
            "robot_angle_diff_batch": [],
            "robot_walls_alignments_batch": [],
            "alpha_batch": [],
            "beta_batch": [],
            "alpha_clockwise_batch": [],
            "beta_clockwise_batch": [],
        }
    )
    results.update(
        {
            "robot_distance_slopes": [],
            "fish_velocity_slopes": [],
            "robot_velocity_slopes": [],
        }
    )
    results.update(
        {
            "fish_wall_category": [],
            "robot_wall_category": [],
        }
    )
    results.update(
        {
            "fish_distance_contributions": [],
            "robot_distance_contributions": [],
        }
    )
    results.update(
        {
            "fish_turn_1hz": [],
        }
    )
    results.update(
        {
            "robot_fish_follow_metric": [],
        }
    )

    for trial_id, trial in trials.iterrows():
        start_frame = trial["start_frame"]
        end_frame = trial["end_frame"]
        trial_fish_poses = fish_poses[start_frame:end_frame]
        trial_robot_poses = robot_poses[start_frame:end_frame]

        n_frames = end_frame - start_frame
        results["trial_id"].extend([trial_id] * n_frames)
        results["frame_number"].extend(range(start_frame, end_frame))
        results["intra_trial_frame_number"].extend(range(n_frames))

        #######################################################################
        robot_distances = np.full(n_frames, np.nan)
        for i in range(n_frames):
            if (
                np.isnan(trial_fish_poses[i]).any()
                or np.isnan(trial_robot_poses[i]).any()
            ):
                continue
            else:
                robot_distances[i] = np.linalg.norm(
                    trial_fish_poses[i, :2] - trial_robot_poses[i, :2]
                )
        results[f"robot_distance"].extend(robot_distances.tolist())

        wall_distances = []
        wall_sorted_indexes = []
        for i in range(n_frames):
            if np.isnan(trial_fish_poses[i]).any():
                wall_distances.append([np.nan] * 4)
                wall_sorted_indexes.append([np.nan] * 4)
            else:
                _wall_distances, _wall_indexes = environment.get_wall_distances_sorted(
                    trial_fish_poses[i, :2]
                )
                wall_distances.append(_wall_distances)
                wall_sorted_indexes.append(_wall_indexes)
        results["wall_distances"].extend(wall_distances)

        #######################################################################
        for n_step in n_steps:
            fish_velocities = np.full(n_frames, np.nan)
            for i in range(n_frames - n_step):
                tfp_b, tfp_n_step = n_step_nan_check(trial_fish_poses, i, n_step)
                if (
                    np.isnan(trial_fish_poses[i]).any()
                    or tfp_b  # np.isnan(trial_fish_poses[i + n_step]).any()
                ):
                    continue
                else:
                    fish_velocities[i] = np.linalg.norm(
                        trial_fish_poses[i + tfp_n_step, :2] - trial_fish_poses[i, :2]
                    ) / (tfp_n_step * period)
            results[f"fish_velocity_n={n_step}"].extend(fish_velocities.tolist())

            fish_turns = np.full(n_frames, np.nan)
            for i in range(n_step, n_frames - n_step):
                tfp_b_min, tfp_n_step_min = n_step_nan_check(
                    trial_fish_poses, i, -n_step
                )
                tfp_b_plus, tfp_n_step_plus = n_step_nan_check(
                    trial_fish_poses, i, n_step
                )
                if (
                    np.isnan(trial_fish_poses[i]).any()
                    or tfp_b_min  # np.isnan(trial_fish_poses[i - n_step]).any()
                    or tfp_b_plus  # np.isnan(trial_fish_poses[i + n_step]).any()
                ):
                    continue
                else:
                    angle = np.arctan2(
                        trial_fish_poses[i, 1]
                        - trial_fish_poses[i + tfp_n_step_min, 1],
                        trial_fish_poses[i, 0]
                        - trial_fish_poses[i + tfp_n_step_min, 0],
                    )
                    angle2 = np.arctan2(
                        trial_fish_poses[i + tfp_n_step_plus, 1]
                        - trial_fish_poses[i, 1],
                        trial_fish_poses[i + tfp_n_step_plus, 0]
                        - trial_fish_poses[i, 0],
                    )
                fish_turns[i] = abs(geometry.diff_angles(angle, angle2))
            results[f"fish_turn_n={n_step}"].extend(fish_turns.tolist())

            robot_velocities = np.full(n_frames, np.nan)
            for i in range(n_frames - n_step):
                trp_b, trp_n_step = n_step_nan_check(trial_robot_poses, i, n_step)
                if (
                    np.isnan(trial_robot_poses[i]).any()
                    or trp_b  # np.isnan(trial_robot_poses[i + n_step]).any()
                ):
                    continue
                else:
                    robot_velocities[i] = np.linalg.norm(
                        trial_robot_poses[i + trp_n_step, :2] - trial_robot_poses[i, :2]
                    ) / (trp_n_step * period)
            results[f"robot_velocity_n={n_step}"].extend(robot_velocities.tolist())

            robot_accelerations = np.full(n_frames, np.nan)
            for i in range(n_frames - n_step):
                rv_b, rv_n_step = n_step_nan_check(robot_velocities, i, n_step)
                if (
                    np.isnan(robot_velocities[i]).any()
                    or rv_b  # np.isnan(robot_velocities[i + n_step]).any()
                ):
                    continue
                else:
                    robot_accelerations[i] = (
                        robot_velocities[i + rv_n_step] - robot_velocities[i]
                    ) / (rv_n_step * period)
            results[f"robot_acceleration_n={n_step}"].extend(
                robot_accelerations.tolist()
            )

            robot_angle_diff = np.full(n_frames, np.nan)
            for i in range(n_frames - n_step):
                trp_b, trp_n_step = n_step_nan_check(trial_robot_poses, i, n_step)
                if (
                    np.isnan(trial_fish_poses[i]).any()
                    or np.isnan(trial_robot_poses[i]).any()
                    or trp_b  # np.isnan(trial_robot_poses[i + n_step]).any()
                ):
                    continue
                else:
                    angle = np.arctan2(
                        trial_fish_poses[i, 1] - trial_robot_poses[i, 1],
                        trial_fish_poses[i, 0] - trial_robot_poses[i, 0],
                    )
                    robot_angle = np.arctan2(
                        trial_robot_poses[i + trp_n_step, 1] - trial_robot_poses[i, 1],
                        trial_robot_poses[i + trp_n_step, 0] - trial_robot_poses[i, 0],
                    )
                    robot_angle_diff[i] = abs(geometry.diff_angles(angle, robot_angle))
            results[f"robot_angle_diff_n={n_step}"].extend(robot_angle_diff.tolist())

            robot_walls_alignments = np.full((n_frames, 4), np.nan)
            for i in range(n_frames - n_step):
                trp_b, trp_n_step = n_step_nan_check(trial_robot_poses, i, n_step)
                if (
                    np.isnan(trial_robot_poses[i]).any()
                    or trp_b  # np.isnan(trial_robot_poses[i + n_step]).any()
                    or np.isnan(wall_sorted_indexes[i]).any()
                ):
                    continue
                else:
                    for j, wall_index in enumerate(wall_sorted_indexes[i]):
                        robot_vector = (
                            trial_robot_poses[i + trp_n_step, :2]
                            - trial_robot_poses[i, :2]
                        )
                        wall_vector = environment.get_wall_vector_by_index(wall_index)
                        rwa = geometry.angle_between_vectors(robot_vector, wall_vector)
                        robot_walls_alignments[i, j] = rwa
            results[f"robot_walls_alignments_n={n_step}"].extend(robot_walls_alignments.tolist())

        #######################################################################
        # Batch
        batch = 5

        batch_robot_poses = np.full_like(trial_robot_poses, np.nan)
        for i in range(batch, trial_robot_poses.shape[0]):
            batch_robot_poses[i] = np.nanmean(trial_robot_poses[i - batch : i], axis=0)

        batch_fish_poses = np.full_like(trial_fish_poses, np.nan)
        for i in range(batch, trial_fish_poses.shape[0]):
            batch_fish_poses[i] = np.nanmean(trial_fish_poses[i - batch : i], axis=0)

        batch_robot_directions = np.full((trial_robot_poses.shape[0], 2), np.nan)
        for i in range(batch, trial_robot_poses.shape[0] - batch):
            p = batch_robot_poses[i]
            q = batch_robot_poses[i + batch]
            batch_robot_directions[i] = q[:2] - p[:2]

        batch_fish_directions = np.full((trial_fish_poses.shape[0], 2), np.nan)
        for i in range(batch, trial_fish_poses.shape[0] - batch):
            p = batch_fish_poses[i]
            q = batch_fish_poses[i + batch]
            batch_fish_directions[i] = q[:2] - p[:2]

        # 
        robot_distance_batch = np.full(n_frames, np.nan)
        for i in range(batch, n_frames - batch):
            if (
                np.isnan(batch_fish_poses[i]).any()
                or np.isnan(batch_robot_poses[i]).any()
            ):
                continue
            else:
                robot_distance_batch[i] = np.linalg.norm(
                    batch_fish_poses[i, :2] - batch_robot_poses[i, :2]
                )
        results["robot_distance_batch"].extend(robot_distance_batch.tolist())

        fish_velocity_batch = np.full(n_frames, np.nan)
        for i in range(batch, n_frames - batch):
            if (
                np.isnan(batch_fish_poses[i]).any()
                or np.isnan(batch_fish_poses[i + batch]).any()
            ):
                continue
            else:
                fish_velocity_batch[i] = np.linalg.norm(
                    batch_fish_poses[i + batch, :2] - batch_fish_poses[i, :2]
                ) / (batch * period)
        results["fish_velocity_batch"].extend(fish_velocity_batch.tolist())

        fish_acceleration_batch = np.full(n_frames, np.nan)
        for i in range(batch, n_frames - batch):
            if (
                np.isnan(fish_velocity_batch[i]).any()
                or np.isnan(fish_velocity_batch[i + batch]).any()
            ):
                continue
            else:
                fish_acceleration_batch[i] = (
                    fish_velocity_batch[i + batch] - fish_velocity_batch[i]
                ) / (batch * period)
        results["fish_acceleration_batch"].extend(fish_acceleration_batch.tolist())

        fish_turn_batch = np.full(n_frames, np.nan)
        for i in range(batch, n_frames - batch):
            if (
                np.isnan(batch_fish_poses[i]).any()
                or np.isnan(batch_fish_poses[i + batch]).any()
                or np.isnan(batch_fish_poses[i - batch]).any()
            ):
                continue
            else:
                angle = np.arctan2(
                    batch_fish_poses[i, 1] - batch_fish_poses[i - batch, 1],
                    batch_fish_poses[i, 0] - batch_fish_poses[i - batch, 0],
                )
                angle2 = np.arctan2(
                    batch_fish_poses[i + batch, 1] - batch_fish_poses[i, 1],
                    batch_fish_poses[i + batch, 0] - batch_fish_poses[i, 0],
                )
                fish_turn_batch[i] = abs(geometry.diff_angles(angle, angle2))
        results["fish_turn_batch"].extend(fish_turn_batch.tolist())

        results["fish_direction_batch"].extend(batch_fish_directions.tolist())

        robot_velocity_batch = np.full(n_frames, np.nan)
        for i in range(batch, n_frames - batch):
            if (
                np.isnan(batch_robot_poses[i]).any()
                or np.isnan(batch_robot_poses[i + batch]).any()
            ):
                continue
            else:
                robot_velocity_batch[i] = np.linalg.norm(
                    batch_robot_poses[i + batch, :2] - batch_robot_poses[i, :2]
                ) / (batch * period)
        results["robot_velocity_batch"].extend(robot_velocity_batch.tolist())

        robot_acceleration_batch = np.full(n_frames, np.nan)
        for i in range(batch, n_frames - batch):
            if (
                np.isnan(robot_velocity_batch[i]).any()
                or np.isnan(robot_velocity_batch[i + batch]).any()
            ):
                continue
            else:
                robot_acceleration_batch[i] = (
                    robot_velocity_batch[i + batch] - robot_velocity_batch[i]
                ) / (batch * period)
        results["robot_acceleration_batch"].extend(robot_acceleration_batch.tolist())

        robot_angle_diff_batch = np.full(n_frames, np.nan)
        for i in range(batch, n_frames - batch):
            if (
                np.isnan(batch_robot_poses[i]).any()
                or np.isnan(batch_fish_poses[i]).any()
                or np.isnan(batch_robot_poses[i + batch]).any()
            ):
                continue
            else:
                angle = np.arctan2(
                    batch_fish_poses[i, 1] - batch_robot_poses[i, 1],
                    batch_fish_poses[i, 0] - batch_robot_poses[i, 0],
                )
                robot_angle = np.arctan2(
                    batch_robot_poses[i + batch, 1] - batch_robot_poses[i, 1],
                    batch_robot_poses[i + batch, 0] - batch_robot_poses[i, 0],
                )
                robot_angle_diff_batch[i] = abs(geometry.diff_angles(angle, robot_angle))
        results["robot_angle_diff_batch"].extend(robot_angle_diff_batch.tolist())

        robot_walls_alignments_batch = np.full((n_frames, 4), np.nan)
        for i in range(batch, n_frames - batch):
            if (
                np.isnan(batch_robot_directions[i]).any()
                or np.isnan(wall_sorted_indexes).any()
            ):
                continue
            else:
                for j, wall_index in enumerate(wall_sorted_indexes[i]):
                    wall_vector = environment.get_wall_vector_by_index(wall_index)
                    rwa = geometry.angle_between_vectors(
                            batch_robot_directions[i], wall_vector
                        )
                    robot_walls_alignments_batch[i, j] = rwa
        results["robot_walls_alignments_batch"].extend(robot_walls_alignments_batch.tolist())

        alpha_batch = np.full(n_frames, np.nan)
        alpha_clockwise_batch = np.full(n_frames, np.nan)
        for i in range(batch, n_frames - batch):
            if (
                np.isnan(batch_fish_poses[i]).any()
                or np.isnan(batch_robot_poses[i]).any()
                or np.isnan(batch_fish_directions[i]).any()
            ):
                continue
            else:
                q_f =  batch_fish_directions[i]
                q_fr = batch_fish_poses[i, :2] - batch_robot_poses[i, :2]
                alpha_batch[i] = geometry.angle_between_vectors(
                    q_f, q_fr
                )
                alpha_clockwise_batch[i] = geometry.vectors_angle(
                    q_f, q_fr
                )
        results["alpha_batch"].extend(alpha_batch.tolist())
        results["alpha_clockwise_batch"].extend(alpha_clockwise_batch.tolist())

        beta_batch = np.full(n_frames, np.nan)
        beta_clockwise_batch = np.full(n_frames, np.nan)
        for i in range(batch, n_frames - batch):
            if (
                np.isnan(batch_robot_directions[i]).any()
                or np.isnan(batch_robot_poses[i]).any()
                or np.isnan(batch_fish_poses[i]).any()
            ):
                continue
            else:
                q_r =  batch_robot_directions[i]
                q_fr = batch_fish_poses[i, :2] - batch_robot_poses[i, :2]
                beta_batch[i] = geometry.angle_between_vectors(
                    q_r, q_fr
                )
                beta_clockwise_batch[i] = geometry.vectors_angle(
                    q_r, q_fr
                )
        results["beta_batch"].extend(beta_batch.tolist())
        results["beta_clockwise_batch"].extend(beta_clockwise_batch.tolist())

        #######################################################################
        # Slopes

        x_period = 1 # period
        robot_distance_slopes = np.full(n_frames, np.nan)
        for i in range(batch, n_frames - batch):
            if (
                np.isnan(robot_distance_batch[i]).any()
                or np.isnan(robot_distance_batch[i + batch]).any()
            ):
                continue
            else:
                robot_distance_slopes[i] = (
                    robot_distance_batch[i + batch] - robot_distance_batch[i]
                ) / (batch * x_period)
        results["robot_distance_slopes"].extend(robot_distance_slopes.tolist())

        fish_velocity_slopes = np.full(n_frames, np.nan)
        for i in range(batch, n_frames - batch):
            if (
                np.isnan(fish_velocity_batch[i]).any()
                or np.isnan(fish_velocity_batch[i + batch]).any()
            ):
                continue
            else:
                fish_velocity_slopes[i] = (
                    fish_velocity_batch[i + batch] - fish_velocity_batch[i]
                ) / (batch * x_period)
        results["fish_velocity_slopes"].extend(fish_velocity_slopes.tolist())

        robot_velocity_slopes = np.full(n_frames, np.nan)
        for i in range(batch, n_frames - batch):
            if (
                np.isnan(robot_velocity_batch[i]).any()
                or np.isnan(robot_velocity_batch[i + batch]).any()
            ):
                continue
            else:
                robot_velocity_slopes[i] = (
                    robot_velocity_batch[i + batch] - robot_velocity_batch[i]
                ) / (batch * x_period)
        results["robot_velocity_slopes"].extend(robot_velocity_slopes.tolist())

        #######################################################################
        # Scenarios
        n_obstructed_walls = np.full(n_frames, np.nan)
        n_interferred_walls = np.full(n_frames, np.nan)

        for i in range(batch, n_frames - batch):
            if (
                np.isnan(batch_robot_directions[i]).any()
                or np.isnan(batch_fish_poses[i]).any()
            ):
                continue
            else:
                _n_obstructed_walls, _n_interferred_walls = (
                    scenarios.get_wall_information(
                        environment, batch_robot_directions[i], batch_fish_poses[i, :2]
                    )
                )
                n_obstructed_walls[i] = _n_obstructed_walls
                n_interferred_walls[i] = _n_interferred_walls
        results["n_obstructed_walls"].extend(n_obstructed_walls.tolist())
        results["n_interferred_walls"].extend(n_interferred_walls.tolist())

        is_robot_chasing = np.full(n_frames, np.nan)
        for i in range(batch, n_frames - batch):
            robot_velocity = results[f"robot_velocity_batch"][i]
            robot_angle_diff = results[f"robot_angle_diff_batch"][i]
            if np.isnan(robot_velocity) or np.isnan(robot_angle_diff):
                continue
            else:
                is_robot_chasing[i] = scenarios.is_robot_chasing(
                    robot_velocity, robot_angle_diff
                )
        results["is_robot_chasing"].extend(is_robot_chasing.tolist())
    
        #######################################################################
        # Wall categories
        fish_wall_category = np.full(n_frames, np.nan)

        for i in range(batch, n_frames - batch):
            if (
                np.isnan(batch_fish_poses[i]).any()
            ):
                continue
            else:
                fish_wall_distances, _ = environment.get_wall_distances_sorted(
                    batch_fish_poses[i, :2]
                )
                fish_wall_category[i] = np.sum(fish_wall_distances < 8.5)
        results["fish_wall_category"].extend(fish_wall_category.tolist())
        
        robot_wall_category = np.full(n_frames, np.nan)

        for i in range(batch, n_frames - batch):
            if (
                np.isnan(batch_robot_directions[i]).any()
            ):
                continue
            else:
                robot_wall_distances, _ = environment.get_wall_distances_sorted(
                    batch_robot_poses[i, :2]
                )
                robot_wall_category[i] = np.sum(robot_wall_distances < 8.5)
        results["robot_wall_category"].extend(robot_wall_category.tolist())

        #######################################################################
        # Distance contributions
        fish_distance_contributions = np.full(n_frames, np.nan)
        robot_distance_contributions = np.full(n_frames, np.nan)

        _step = 1
        for i in range(batch, n_frames - batch - _step):
            if (
                np.isnan(batch_fish_poses[i]).any()
                or np.isnan(batch_robot_poses[i]).any()
                or np.isnan(batch_fish_poses[i + _step]).any()
                or np.isnan(batch_robot_poses[i + _step]).any()
            ):
                continue
            else:
                _d = np.linalg.norm(
                    batch_fish_poses[i, :2] - batch_robot_poses[i, :2]
                )
                _d_f = np.linalg.norm(
                    batch_fish_poses[i + _step, :2] - batch_robot_poses[i, :2]
                )
                _d_r = np.linalg.norm(
                    batch_fish_poses[i, :2] - batch_robot_poses[i + _step, :2]
                )
                _c_f = _d_f / _d
                _c_r = _d_r / _d
                c_f = _c_f / (_c_f + _c_r)
                c_r = _c_r / (_c_f + _c_r)
                fish_distance_contributions[i] = _c_f
                robot_distance_contributions[i] = _c_r
        results["fish_distance_contributions"].extend(fish_distance_contributions.tolist())
        results["robot_distance_contributions"].extend(robot_distance_contributions.tolist())

        #######################################################################
        # 1 Hz
        fish_turn_1hz = np.full(n_frames, np.nan)
        _step = 25
        for i in range(_step, n_frames - _step - batch):
            if (
                np.isnan(batch_fish_poses[i]).any()
                or np.isnan(batch_fish_poses[i - batch]).any()
                or np.isnan(batch_fish_poses[i + _step]).any()
                or np.isnan(batch_fish_poses[i + _step + batch]).any()
            ):
                continue
            else:
                angle = np.arctan2(
                    batch_fish_poses[i, 1] - batch_fish_poses[i - batch, 1],
                    batch_fish_poses[i, 0] - batch_fish_poses[i - batch, 0],
                )
                angle2 = np.arctan2(
                    batch_fish_poses[i + _step + batch, 1] - batch_fish_poses[i + _step, 1],
                    batch_fish_poses[i + _step + batch, 0] - batch_fish_poses[i + _step, 0],
                )
                fish_turn_1hz[i] = geometry.diff_angles(angle, angle2)
        results["fish_turn_1hz"].extend(fish_turn_1hz.tolist())

        #######################################################################
        # Follow metric
        robot_fish_follow_metric = np.full(n_frames, np.nan)
        _step = 10
        for i in range(batch, n_frames - batch):
            if (
                np.isnan(batch_fish_poses[i:i + _step]).any()
                or np.isnan(batch_robot_poses[i:i + _step]).any()
            ):
                continue
            else:
                rffm = metrics.calculate_follow(
                    batch_robot_poses[i:i + _step], batch_fish_poses[i:i + _step]
                )
                robot_fish_follow_metric[i] = np.mean(rffm)
        results["robot_fish_follow_metric"].extend(robot_fish_follow_metric.tolist())
    
    results = pd.DataFrame(results)
    results.to_pickle(f"{output_folder}/{output_filename}.pkl")

    return results
