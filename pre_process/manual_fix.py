import os
import pickle
import cv2

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np

from core.experiment import Experiment, get_experiment

from pre_process.manual_poses.generate_suggestion import from_frame_number

from utils.parser import get_default_parser

import warnings

warnings.filterwarnings("ignore")

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "text.latex.preamble": r"\usepackage{amsfonts}"
        r"\usepackage{amsmath}"
        r"\usepackage{color}",
    }
)


def parse_args():
    parser = get_default_parser()

    parser.add_argument("-sf", "--start_frame", type=int, default=13250)
    parser.add_argument("-ef", "--end_frame", type=int, default=13290)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--suggest", action="store_true")

    args = parser.parse_args()
    return args


def plot_world(ax, experiment: Experiment, frame_number: int, suggest: bool):
    # World
    ax.set_xlim(experiment.arena_xlim)
    ax.set_ylim(experiment.arena_ylim)

    # Generate suggestions
    robot_pose = experiment.robot_poses[frame_number]
    fish_pose = experiment.fish_poses[frame_number]

    suggested_robot_pose, suggested_fish_pose = None, None

    if suggest:
        start_frame, end_frame = None, None
        if np.isnan(robot_pose).any():
            print(f"Finding suggested robot pose for frame {frame_number}...")
            for i in range(1, 200):
                if not np.isnan(experiment.robot_poses[frame_number - i]).any():
                    start_frame = frame_number - i
                    break
            for i in range(1, 200):
                if not np.isnan(experiment.robot_poses[frame_number + i]).any():
                    end_frame = frame_number + i
                    break
        if start_frame is not None and end_frame is not None:
            suggest_robot_poses, suggest_fish_poses = from_frame_number(
                experiment, start_frame, end_frame, False
            )
            suggested_robot_pose = suggest_robot_poses[frame_number - start_frame]

        start_frame, end_frame = None, None
        if np.isnan(fish_pose).any():
            print(f"Finding suggested fish pose for frame {frame_number}...")
            for i in range(1, 200):
                if not np.isnan(experiment.fish_poses[frame_number - i]).any():
                    start_frame = frame_number - i
                    break
            for i in range(1, 200):
                if not np.isnan(experiment.fish_poses[frame_number + i]).any():
                    end_frame = frame_number + i
                    break
        if start_frame is not None and end_frame is not None:
            suggest_robot_poses, suggest_fish_poses = from_frame_number(
                experiment, start_frame, end_frame, False
            )
            suggested_fish_pose = suggest_fish_poses[frame_number - start_frame]

    if (
        np.isnan(robot_pose).any()
        and not experiment.robot_manual_corrections[frame_number]
        and suggested_robot_pose is not None
    ):
        robot_pose = suggested_robot_pose
    if (
        np.isnan(fish_pose).any()
        and not experiment.fish_manual_corrections[frame_number]
        and suggested_fish_pose is not None
    ):
        fish_pose = suggested_fish_pose

    # Scatter robot and fish positions
    color_robot, color_fish = (0, 1.0, 1.0), (1.0, 0, 1.0)
    ax.scatter(
        robot_pose[0],
        robot_pose[1],
        color=color_robot,
        label="Robot",
    )
    ax.scatter(
        fish_pose[0],
        fish_pose[1],
        color=color_fish,
        label="Fish",
    )
    # Quiver the orientation
    ax.quiver(
        robot_pose[0],
        robot_pose[1],
        0.1 * np.cos(robot_pose[2]),
        0.1 * np.sin(robot_pose[2]),
        color=color_robot,
    )
    ax.quiver(
        fish_pose[0],
        fish_pose[1],
        0.1 * np.cos(fish_pose[2]),
        0.1 * np.sin(fish_pose[2]),
        color=color_fish,
    )

    # Draw info
    draw_x = -47.5
    draw_y = 42.5
    draw_offset = 10
    draw_fontsize = 9
    draw_ha = "left"
    trial_id = None
    for index, row in experiment.trials.iterrows():
        if row["start_frame"] <= frame_number <= row["end_frame"]:
            trial_id = index
            break
    if trial_id is None:
        trial_id = "-"
    text = str(frame_number)
    if experiment.robot_manual_corrections[frame_number] or \
        experiment.fish_manual_corrections[frame_number]:
        text += " (manual)"
        # color text in orange
        color = (1.0, 0.5, 0)
    else:
        color = (0, 0, 0)
    ax.text(
        draw_x,
        draw_y - 0 * draw_offset,
        f"Frame : {text}",
        fontsize=draw_fontsize,
        ha=draw_ha,
        va="center",
        color=color,
    )
    ax.text(
        draw_x,
        draw_y - 1 * draw_offset,
        f"Trial : {trial_id}",
        fontsize=draw_fontsize,
        ha=draw_ha,
        va="center",
    )
    text = ""
    if suggested_robot_pose is not None:
        robot_pose = suggested_robot_pose
    else:
        robot_pose = experiment.robot_poses[frame_number]
    for x in robot_pose:
        text += f"{x:.2f},"
    if suggested_robot_pose is not None:
        text += f" (suggested)"
    if experiment.robot_manual_corrections[frame_number]:
        text_color = (1.0, 0.5, 0)
    else:
        text_color = (0, 0, 0)
    ax.text(
        draw_x - 10,
        draw_y - 2 * draw_offset,
        rf"R",
        fontsize=draw_fontsize,
        ha=draw_ha,
        va="center",
        color=color_robot,
    )
    ax.text(
        draw_x,
        draw_y - 2 * draw_offset,
        f"Robot pose: {text}",
        fontsize=draw_fontsize,
        ha=draw_ha,
        va="center",
        color=text_color,
    )
    text = ""
    if suggested_fish_pose is not None:
        fish_pose = suggested_fish_pose
    else:
        fish_pose = experiment.fish_poses[frame_number]
    for x in fish_pose:
        text += f"{x:.2f},"
    if suggested_fish_pose is not None:
        text += f" (suggested)"
    if experiment.fish_manual_corrections[frame_number]:
        text_color = (1.0, 0.5, 0)
    else:
        text_color = (0, 0, 0)
    ax.text(
        draw_x - 10,
        draw_y - 3 * draw_offset,
        rf"F",
        fontsize=draw_fontsize,
        ha=draw_ha,
        va="center",
        color=color_fish,
    )
    ax.text(
        draw_x,
        draw_y - 3 * draw_offset,
        f"Fish pose: {text}",
        fontsize=draw_fontsize,
        ha=draw_ha,
        va="center",
        color=text_color,
    )

    ax.set_xticks([])
    ax.set_yticks([])


def manual_fix(
    experiment: Experiment,
    start_frame: int,
    end_frame: int,
    include_video: bool,
    suggest: bool,
    output_folder: str,
    fix_output_folder: str,
):
    n_frames = end_frame - start_frame

    # # Detect possible issues
    # # get indexes of experiment.robot_poses where there are NaN values
    # robot_nan_indexes = np.where(np.isnan(experiment.robot_poses).any(axis=1))[0]
    # fish_nan_indexes = np.where(np.isnan(experiment.fish_poses).any(axis=1))[0]
    # print(robot_nan_indexes.shape)

    if include_video:
        # Load video
        output_filename = output_folder + "/output.mp4"
        if not os.path.exists(output_filename):
            print(f"Output video not found: {output_filename}")
            print(f"Using output_raw instead...")
            output_filename = output_folder + "/output_raw.mp4"
        cap = cv2.VideoCapture(output_filename)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Create pdf
    n_fig_rows = 5
    if include_video:
        n_fig_cols = 2
    else:
        n_fig_cols = 1
    title_fontsize = 16

    if include_video:
        pdf_v = "video"
    else:
        pdf_v = "flat"
    with PdfPages(f"{fix_output_folder}/{start_frame}-{end_frame}_{pdf_v}.pdf") as pdf:
        for i in range(start_frame, end_frame, 1):

            if i == int(0.1 * n_frames):
                print("Processing... 10%")
            elif i == int(0.25 * n_frames):
                print("Processing... 25%")
            elif i == int(0.5 * n_frames):
                print("Processing... 50%")
            elif i == int(0.75 * n_frames):
                print("Processing... 75%")
            elif i == int(0.95 * n_frames):
                print("Processing... 95%")
            else:
                pass

            if (i - start_frame) % n_fig_rows != 0:
                continue
            else:
                if (i - start_frame) > 0:
                    pdf.savefig()
                    plt.close(fig)
            fig, axes = plt.subplots(
                n_fig_rows, n_fig_cols, figsize=(2 * n_fig_cols + 2, 2 * n_fig_rows)
            )
            # fig.subplots_adjust(wspace=2/6)

            if include_video:
                axes[0, 0].set_title(r"Video", fontsize=title_fontsize)
                axes[0, 1].set_title(r"Info", fontsize=title_fontsize)
            else:
                axes[0].set_title(r"Info", fontsize=title_fontsize)

            for j in range(n_fig_rows):
                if (i - start_frame) + j >= n_frames:
                    break
                if include_video:
                    ret, frame = cap.read()
                    # Frame
                    ax = axes[j, 0]
                    ax.imshow(frame)

                    ax.set_xticks([])
                    ax.set_yticks([])

                    # Experiment info
                    ax = axes[j, 1]
                else:
                    ax = axes[j]
                    ax.set_aspect("equal")
                plot_world(ax, experiment, i + j, suggest)

        pdf.savefig()
        plt.close(fig)

    if include_video:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parse_args()
    variables = vars(args)

    experiment = get_experiment(**variables)

    input_folder, output_folder = experiment.input_folder, experiment.output_folder
    with open(output_folder + "/experiment.pkl", "rb") as f:
        experiment = pickle.load(f)

    fix_output_folder = output_folder + "/manual_fix"
    if not os.path.exists(fix_output_folder):
        os.makedirs(fix_output_folder)

    manual_fix(
        experiment,
        variables["start_frame"],
        variables["end_frame"],
        variables["video"],
        variables["suggest"],
        output_folder,
        fix_output_folder,
    )
