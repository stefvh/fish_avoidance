import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from core.experiment import Experiment
from core.study import study


size_input = (width_input, height_input) = (2048, 2048)
size_output = (width_output, height_output) = (640, 480)


def rescale_position(position):
    x, y = position
    x = (x / width_input) * width_output
    y = (y / height_input) * height_output
    return x, y


def rescale_corners(corners):
    corners[:, 0] *= (width_output / 100)
    corners[:, 1] *= (height_output / 100)
    corners = corners.astype(int)
    return corners


def label_trajectories_in_video(
    camera_filename,
    output_folder,
    output_filename,
    rectification_corners,
    trajectories,
    robot_positions,
    fish_positions,
):
    vid_input = cv2.VideoCapture(camera_filename)

    vid_output = cv2.VideoWriter(
        f"{output_folder}/{output_filename}.mp4", cv2.VideoWriter_fourcc(*"MPEG"), 25, size_output
    )

    corners = rescale_corners(rectification_corners)

    all_trajectory_frames = []
    for trajectory in trajectories:
        all_trajectory_frames.extend(trajectory['frame'].to_list())

    color_default, color_robot, color_fish = (0, 255, 0), (0, 255, 255), (255, 0, 255)
    color_trajectory = (255, 0, 0)
    frame_number = 0
    while True:
        ret, frame = vid_input.read()
        if not ret:
            break

        frame = cv2.resize(frame, size_output)

        draw_y = 30
        draw_i = 1

        # Draw frame number
        cv2.putText(frame,
            "F: " + str(frame_number),
            (10, draw_i * draw_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color_default,
            2,
            cv2.LINE_AA,
        )
        draw_i += 1

        # Draw legend of color_robot and color_fish
        cv2.putText(frame,
            "Robot",
            (10, draw_i * draw_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color_robot,
            2,
            cv2.LINE_AA,
        )
        draw_i += 1
        cv2.putText(frame,
            "Fish",
            (10, draw_i * draw_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color_fish,
            2,
            cv2.LINE_AA,
        )
        draw_i += 1

        # Draw rectification corners
        if frame_number in all_trajectory_frames:
            rectangle_color = color_trajectory
        else:
            rectangle_color = color_default
        cv2.rectangle(
            frame,
            (corners[0][0], corners[0][1]),
            (corners[2][0], corners[2][1]),
            rectangle_color,
            2,
        )

        # Draw trajectory id
        text = "T: #"
        if frame_number in all_trajectory_frames:
            for t_i, trajectory in enumerate(trajectories):
                if frame_number in trajectory['frame'].to_list():
                    text = "T: " + str(t_i)
        cv2.putText(frame,
            text,
            (10, draw_i * draw_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color_trajectory,
            2,
            cv2.LINE_AA,
        )
        draw_i += 1

        # Draw robot and fish positions, if positions are not np.nan
        if frame_number < robot_positions.shape[0] and not np.isnan(robot_positions[frame_number]).any():
            x, y = rescale_position(robot_positions[frame_number])
            cv2.circle(frame, (int(x), int(y)), 6, color_robot, 1)
        if frame_number < fish_positions.shape[0] and not np.isnan(fish_positions[frame_number]).any():
            x, y = rescale_position(fish_positions[frame_number])
            cv2.circle(frame, (int(x), int(y)), 6, color_fish, 1)

        vid_output.write(frame)
        frame_number += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        cv2.imshow("frame", frame)


def plot_trajectory_descriptives(axes, trajectory: pd.DataFrame):
    label_fontsize = 26
    linewidth = 2

    for ax in axes:
        ax.yaxis.label.set_size(label_fontsize)

    ax = axes[0]
    ax.plot(trajectory['frame'], 
            trajectory['distance'], 
            label='Distance',
            color='tab:blue',
            linewidth=linewidth)
    ax.set_ylabel("Distance (cm)")

    ax = axes[1]
    ax.plot(trajectory['frame'], 
            trajectory['speed_fish'], 
            label='Speed Fish',
            color='tab:orange',
            linewidth=linewidth)
    ax.set_ylabel("Fish Speed (cm/s)")

    ax = axes[2]
    ax.plot(trajectory['frame'],
             trajectory['turn_fish'], 
             label='Turn Fish',
             color='tab:green',
            linewidth=linewidth)
    ax.set_ylabel("Fish Turn (rad)")
    ax.hlines(0, trajectory['frame'].min(), trajectory['frame'].max(), color='tab:gray', linestyle='--')

    ax = axes[3]
    ax.plot(trajectory['frame'], 
            trajectory['acceleration_fish'], 
            label='Acceleration Fish',
            color='tab:purple',
            linewidth=linewidth)
    ax.set_ylabel("Acceleration (cm/s²)")

    # ax = axes[3]
    # ax.plot(trajectory['frame'], 
    #         trajectory['speed_robot'], 
    #         label='Robot Speed',
    #         color='tab:purple',
    #         linewidth=linewidth)
    # ax.set_ylabel("Robot Speed (cm/s)")

    ax = axes[4]
    ax.plot(trajectory['frame'], 
            trajectory['cos_rev'], 
            label='Cosine Fish Retreat',
            color='tab:red',
            linewidth=linewidth)
    ax.set_ylabel("Cos Fish Retreat")
    ax.hlines(0, trajectory['frame'].min(), trajectory['frame'].max(), color='tab:gray', linestyle='--')
    
    return ax


def plot_trajectory_visual(ax, trajectory: pd.DataFrame):
    label_fontsize = 26

    cmap = plt.cm.get_cmap('Blues')
    n_frames = trajectory.shape[0]
    alphas = np.linspace(0, 1, n_frames)
    for i in range(n_frames - 1):
        ax.plot(trajectory['x_robot'].iloc[i:i+2], trajectory['y_robot'].iloc[i:i+2], color=cmap(alphas[i]), lw=4)

    cmap = plt.cm.get_cmap('Greens')
    for i in range(n_frames - 1):
        ax.plot(trajectory['x_fish'].iloc[i:i+2], trajectory['y_fish'].iloc[i:i+2], color=cmap(alphas[i]), lw=4)
    
    # See if the fish is near a wall
    wall_threshold = 10.5
    wall_distances = np.array(trajectory['wall_distances_fish'].to_list())
    wall_near = np.any(wall_distances < wall_threshold)
    # Text to display
    text = "Fish Near Wall" if wall_near else "Fish Not Near Wall"
    ax.text(0.05, 0.05, text, fontsize=label_fontsize, ha='left', va='center', transform=ax.transAxes)

    ax.set_xlabel("X (cm)", fontsize=label_fontsize)
    ax.set_ylabel("Y (cm)", fontsize=label_fontsize)

    # add blue to legend
    ax.plot([], [], color='blue', label='Robot Trajectory')
    ax.plot([], [], color='green', label='Fish Trajectory')
    ax.legend()


def plot_trajectory(t_i: int, trajectory: pd.DataFrame):
    fig, axes = plt.subplots(5, 2)
    fig.set_figheight(20)
    fig.set_figwidth(40)

    gs = axes[0, 0].get_gridspec()
    for ax in axes[0:, 0]:
        ax.remove()
    axbig = fig.add_subplot(gs[0:, 0])

    plot_trajectory_visual(axbig, trajectory)
    plot_trajectory_descriptives(axes[:, 1], trajectory)

    min_frame = trajectory['frame'].min()
    max_frame = trajectory['frame'].max()
    filename = f"T={t_i}_F={min_frame}-{max_frame}.png"
    filefolder = f"{study.plot_output_folder}/2"
    if not os.path.exists(filefolder):
        os.makedirs(filefolder)
    filepath = f"{filefolder}/{filename}"
    fig.savefig(filepath,)


def get_trajectories(data: pd.DataFrame, experiment_id: int):
    df = data[data['experiment_id'] == experiment_id]

    trajectories_frames = []
    current_trajectory_frames = []

    arr_speed_robot = df['speed_robot'].to_numpy()
    arr_distances = df['distance'].to_numpy()
    arr_cos_apv = df['cos_apv'].to_numpy()
    arr_wall_distances_robot = np.array(df['wall_distances_robot'].to_list())
    arr_frame = df['frame'].to_numpy()

    n_rows = df.shape[0]

    delta = 15
    wall_threshold = 10.5
    min_trajectory_length = 25
    distance_threshold = 8.0
    speed_robot_threshold = 2.5 # 1.0

    i = 0
    while i < n_rows:
        if len(current_trajectory_frames) == 0:
            # Verify to start a new trajectory
            if i + delta >= n_rows:
                break
            # The delta frames should be consecutive
            if not np.all(np.diff(arr_frame[i:i+delta]) == 1):
                i += 1
                continue
            # Condition 1: Robot is moving
            condition_1 = np.all(arr_speed_robot[i:i+delta] > speed_robot_threshold)
            # Condition 2: Distance between robot and fish is decreasing
            condition_2 = np.all(np.diff(arr_distances[i:i+delta]) < 0)
            # Condition 3: Robot is moving towards the fish
            condition_3 = np.all(arr_cos_apv[i:i+delta] > 0.5)
            # Condition 4: Robot is not near a wall
            condition_4 = np.all(np.all(arr_wall_distances_robot[i:i+delta] > wall_threshold))
            # Condition 5: Fish is initially far from the robot
            condition_5 = arr_distances[i] > 1.25 * distance_threshold

            if condition_1 and condition_2 and condition_3 and condition_4 and condition_5:
                current_trajectory_frames = arr_frame[i:i+delta].tolist()
                i += delta
            else:
                i += 1
        else:
            if arr_frame[i] == arr_frame[i-1] + 1:
                # Verify to end the current trajectory
                # Condition 1: Robot has stopped
                condition_1 = arr_speed_robot[i] < speed_robot_threshold
                # Condition 2: Robot is near a wall
                condition_2 = np.any(arr_wall_distances_robot[i] < wall_threshold)

                if condition_1 or condition_2:
                    # Verify if it was a valid trajectory
                    # Condition A: Trajectory is long enough
                    condition_a = len(current_trajectory_frames) > min_trajectory_length
                    # Condition B: The robot reached the fish close enough in the trajectory
                    distances = arr_distances[i-len(current_trajectory_frames):i]
                    condition_b = np.any(distances < distance_threshold)

                    if condition_a and condition_b:
                        trajectories_frames.append(current_trajectory_frames)
                    current_trajectory_frames = []
                else:
                    current_trajectory_frames.append(arr_frame[i])
            else:
                # Reset the current trajectory
                current_trajectory_frames = []
            i += 1

    # For each trajectory, get the corresponding data
    trajectories = []
    for trajectory_frames in trajectories_frames:
        start_frame = trajectory_frames[0]
        end_frame = trajectory_frames[-1]
        trajectory = df[(df['frame'] >= start_frame) & (df['frame'] <= end_frame)]
        trajectories.append(trajectory)
    
    return trajectories


if __name__ == "__main__":
    data = study.data

    trajectories = get_trajectories(data, 0)
    for t_i, trajectory in enumerate(trajectories):
        plot_trajectory(t_i, trajectory)

    # trajectory_16 = trajectories[16]
    # # print(trajectory_16['wall_distances_fish'])

    # wall_distances = np.array(trajectory_16['wall_distances_fish'].to_list())
    # print(wall_distances)

    # experiment_0 = study.list_experiments[0]
    # print(experiment_0.corners)
    # filefolder = f"{study.plot_output_folder}/2"
    # if not os.path.exists(filefolder):
    #     os.makedirs(filefolder)
    # label_trajectories_in_video(
    #     experiment_0.camera_filename,
    #     filefolder,
    #     "video_with_trajectories",
    #     experiment_0.rectification_corners,
    #     trajectories,
    #     experiment_0.robot_pixels,
    #     experiment_0.fish_pixels,
    # )

