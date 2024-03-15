import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

# plt.rcParams.update({
#     'text.usetex': True,
#     'text.latex.preamble':r'\usepackage{xcolor}'  r'\usepackage{amsfonts}' r'\usepackage{amsmath}'
# })

from pre_process.experiment import Experiment


def _extend_wall(a, b, d_x):
    slope = (b[1] - a[1]) / (b[0] - a[0])
    extended_point = np.array(
        [
            a[0] + d_x,
            a[1] + d_x * slope,
        ]
    )
    if d_x > 0:
        return np.array(
            [
                a,
                extended_point,
            ]
        )
    else:
        return np.array(
            [
                extended_point,
                b,
            ]
        )


def _get_intersection(v, w):
    xdiff = (v[0][0] - v[1][0], w[0][0] - w[1][0])
    ydiff = (v[0][1] - v[1][1], w[0][1] - w[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception("lines do not intersect")
    d = (det(v[0], v[1]), det(w[0], w[1]))
    intersection = (det(d, xdiff) / div, det(d, ydiff) / div)
    return intersection


def _extend_tracking_area_to_corners(tracking_area):
    corners = np.zeros((4, 2))
    n_points = tracking_area.shape[0]

    d_x = 30
    for i in range(4):
        j = (2 * i + 1) % n_points
        v = _extend_wall(tracking_area[j], tracking_area[(j + 1) % n_points], d_x)
        w = _extend_wall(
            tracking_area[(j + 2) % n_points], tracking_area[(j + 3) % n_points], -d_x
        )
        corners[(i + 1) % 4] = _get_intersection(v, w)

    return corners


def render_img(
    counter,
    index,
    i,
    result,
    position_arena,
    robot_positions,
    robot_directions,
    fish_positions,
    fish_directions,
    p_xlim,
    p_ylim,
    video_size,
):
    fig, ax = plt.subplots()

    text = f"Trial: {index}\n"
    text += f"Frame: {i}\n"
    l_texts = [
        (r"$d$", result["robot_distance"], None),
        (r"$N_O$", result["n_obstructed_walls"], 0),
        (r"$N_I$", result["n_interferred_walls"], 0),
        ("Ch?", result["is_robot_chasing"], 0),
        (r"$v_F$", result["fish_velocity_n=5"], None),
        (r"$\theta_F$", result["fish_turn_n=5"], None),
        (r"$v_R$", result["robot_velocity_n=5"], None),
        (r"$\alpha$", result["robot_angle_diff_n=5"], None),
    ]
    for l_text in l_texts:
        a = f"{l_text[0]} = "
        b = f"{l_text[1]:.2f}"
        # if l_text[2] is not None:
        #     if l_text[1] > l_text[2]:
        #         b = r"$\textcolor{green}{" + b + r"}$"

        text += f"{a}{b}\n"

    ax.text(
        45,
        -55,
        text,
        bbox=dict(facecolor="grey", edgecolor="blue", pad=10.0),
    )

    ax.plot(
        position_arena[:, 0],
        position_arena[:, 1],
        "-",
        color="orange",
    )

    if not np.isnan(robot_positions[i]).any():
        ax.quiver(
            robot_positions[i][0],
            robot_positions[i][1],
            np.cos(robot_directions[i]),
            np.sin(robot_directions[i]),
            color="green",
            scale=20,
        )
        ax.scatter(robot_positions[i][0], robot_positions[i][1], color="green")
        ax.annotate(
            f"Robot",
            (robot_positions[i][0], robot_positions[i][1]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )
    if not np.isnan(fish_positions[i]).any():
        ax.quiver(
            fish_positions[i][0],
            fish_positions[i][1],
            np.cos(fish_directions[i]),
            np.sin(fish_directions[i]),
            color="blue",
            scale=20,
        )
        ax.scatter(fish_positions[i][0], fish_positions[i][1], color="blue")
        ax.annotate(
            f"Fish",
            (fish_positions[i][0], fish_positions[i][1]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )

    ax.set_xlim(p_xlim)
    ax.set_ylim(p_ylim)

    # fig to image
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = cv2.resize(img, video_size)

    # close
    plt.close(fig)

    return img


def render_trial(
    output_folder,
    output_filename,
    video_size,
    index,
    trial,
    results,
    position_arena,
    robot_positions,
    robot_directions,
    fish_positions,
    fish_directions,
    p_xlim,
    p_ylim,
):
    start_frame, end_frame = trial["start_frame"], trial["end_frame"]

    vw = cv2.VideoWriter(
        f"{output_folder}/{output_filename}_{index}.mp4",
        cv2.VideoWriter_fourcc(*"MPEG"),
        25,
        video_size,
    )

    trial_results = results.loc[results["trial_id"] == index]

    counter = 0
    for i in range(start_frame, end_frame):
        img = render_img(
            counter,
            index,
            i,
            trial_results.iloc[counter],
            position_arena,
            robot_positions,
            robot_directions,
            fish_positions,
            fish_directions,
            p_xlim,
            p_ylim,
            video_size,
        )

        # write to video
        vw.write(img)

        counter += 1

    vw.release()
    cv2.destroyAllWindows()

    print(f"...Finished rendering trial {index}.")

def render_trial_from_experiment(
    output_folder: str,
    output_filename: str,
    experiment: Experiment,
    trial_id: int,
):
    sub_folder = "tracking"
    output_folder = f"{output_folder}/{sub_folder}"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    robot_positions = experiment.robot_poses[:, :2]
    fish_positions = experiment.fish_poses[:, :2]

    rp_lim = (min(robot_positions[:, 0]), max(robot_positions[:, 0]))
    rp_lim_without_nan = (
        min(robot_positions[~np.isnan(robot_positions[:, 0]), 0]),
        max(robot_positions[~np.isnan(robot_positions[:, 0]), 0]),
    )
    fp_lim = (min(fish_positions[:, 0]), max(fish_positions[:, 0]))
    fp_lim_without_nan = (
        min(fish_positions[~np.isnan(fish_positions[:, 0]), 0]),
        max(fish_positions[~np.isnan(fish_positions[:, 0]), 0]),
    )
    rp_lim, fp_lim = rp_lim_without_nan, fp_lim_without_nan
    position_lim = (min(rp_lim[0], fp_lim[0]), max(rp_lim[1], fp_lim[1]))

    offset = 15
    p_xlim = (position_lim[0] - offset, position_lim[1] + offset)
    p_ylim = (position_lim[0] - offset, position_lim[1] + offset)

    position_arena = np.array(
        [
            (position_lim[0], position_lim[0]),
            (position_lim[0], position_lim[1]),
            (position_lim[1], position_lim[1]),
            (position_lim[1], position_lim[0]),
            (position_lim[0], position_lim[0]),
        ]
    )

    batch = 5
    robot_directions = np.zeros((robot_positions.shape[0], 1))
    for i in range(batch, robot_positions.shape[0] - batch):
        p = np.mean(robot_positions[i - batch : i], axis=0)
        q = np.mean(robot_positions[i + 1 : i + 1 + batch], axis=0)
        robot_directions[i] = np.arctan2(
            q[1] - p[1],
            q[0] - p[0],
        )
    fish_directions = np.zeros((fish_positions.shape[0], 1))
    for i in range(batch, fish_positions.shape[0] - batch):
        p = np.mean(fish_positions[i - batch : i], axis=0)
        q = np.mean(fish_positions[i + 1 : i + 1 + batch], axis=0)
        fish_directions[i] = np.arctan2(
            q[1] - p[1],
            q[0] - p[0],
        )

    render_trial(
        output_folder,
        output_filename,
        (640, 480),
        trial_id,
        experiment.trials.iloc[trial_id],
        experiment.results,
        position_arena,
        robot_positions,
        robot_directions,
        fish_positions,
        fish_directions,
        p_xlim,
        p_ylim,
    )


def render(
    output_folder,
    output_filename,
    tracking_area_corners,
    robot_positions,
    fish_positions,
    trials,
    results,
):
    corners = _extend_tracking_area_to_corners(tracking_area_corners)

    corners = corners - 50
    tracking_area_corners = tracking_area_corners - 50

    offset = 10
    p_xlim = (min(corners[:, 0]) - offset, max(corners[:, 0]) + offset)
    p_ylim = (min(corners[:, 1]) - offset, max(corners[:, 1]) + offset)

    video_size = (640, 480)
    plot_tracking_arena = np.append(
        tracking_area_corners, [tracking_area_corners[0]], axis=0
    )
    plot_corner_area = np.append(corners, [corners[0]], axis=0)

    rp_lim = (min(robot_positions[:, 0]), max(robot_positions[:, 0]))
    rp_lim_without_nan = (
        min(robot_positions[~np.isnan(robot_positions[:, 0]), 0]),
        max(robot_positions[~np.isnan(robot_positions[:, 0]), 0]),
    )
    fp_lim = (min(fish_positions[:, 0]), max(fish_positions[:, 0]))
    fp_lim_without_nan = (
        min(fish_positions[~np.isnan(fish_positions[:, 0]), 0]),
        max(fish_positions[~np.isnan(fish_positions[:, 0]), 0]),
    )
    rp_lim, fp_lim = rp_lim_without_nan, fp_lim_without_nan
    position_lim = (min(rp_lim[0], fp_lim[0]), max(rp_lim[1], fp_lim[1]))

    offset = 15
    p_xlim = (position_lim[0] - offset, position_lim[1] + offset)
    p_ylim = (position_lim[0] - offset, position_lim[1] + offset)

    position_arena = np.array(
        [
            (position_lim[0], position_lim[0]),
            (position_lim[0], position_lim[1]),
            (position_lim[1], position_lim[1]),
            (position_lim[1], position_lim[0]),
            (position_lim[0], position_lim[0]),
        ]
    )

    batch = 5
    robot_directions = np.zeros((robot_positions.shape[0], 1))
    for i in range(batch, robot_positions.shape[0] - batch):
        p = np.mean(robot_positions[i - batch : i], axis=0)
        q = np.mean(robot_positions[i + 1 : i + 1 + batch], axis=0)
        robot_directions[i] = np.arctan2(
            q[1] - p[1],
            q[0] - p[0],
        )
    fish_directions = np.zeros((fish_positions.shape[0], 1))
    for i in range(batch, fish_positions.shape[0] - batch):
        p = np.mean(fish_positions[i - batch : i], axis=0)
        q = np.mean(fish_positions[i + 1 : i + 1 + batch], axis=0)
        fish_directions[i] = np.arctan2(
            q[1] - p[1],
            q[0] - p[0],
        )

    for index, row in trials.iterrows():
        # if index > 0:
        #     break
        render_trial(
            output_folder,
            output_filename,
            video_size,
            index,
            row,
            results,
            position_arena,
            robot_positions,
            robot_directions,
            fish_positions,
            fish_directions,
            p_xlim,
            p_ylim,
        )
