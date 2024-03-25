import os
import pickle
import cv2

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np

from priors.foldering import get_data_folders
from pre_process.experiment import Experiment, get_experiment

from utils.parser import get_default_parser

import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "text.latex.preamble": r"\usepackage{amsfonts}" r"\usepackage{amsmath}" r"\usepackage{color}",
    }
)

def parse_args():
    parser = get_default_parser()

    parser.add_argument("-sf", "--start_frame", type=int, default=13250)
    parser.add_argument("-ef", "--end_frame", type=int, default=13290)

    args = parser.parse_args()
    return args

def plot_world(ax, experiment: Experiment, frame_number):
    # World
    ax.set_xlim(experiment.arena_xlim)
    ax.set_ylim(experiment.arena_ylim)

    # Scatter robot and fish positions
    color_robot, color_fish = (0, 1., 1.), (1., 0, 1.)
    ax.scatter(
        experiment.robot_poses[frame_number][0],
        experiment.robot_poses[frame_number][1],
        color=color_robot,
        label="Robot",
    )
    ax.scatter(
        experiment.fish_poses[frame_number][0],
        experiment.fish_poses[frame_number][1],
        color=color_fish,
        label="Fish",
    )
    # Quiver the orientation
    ax.quiver(
        experiment.robot_poses[frame_number][0],
        experiment.robot_poses[frame_number][1],
        0.1 * np.cos(-experiment.robot_poses[frame_number][2]),
        0.1 * np.sin(-experiment.robot_poses[frame_number][2]),
        color=color_robot,
    )
    ax.quiver(
        experiment.fish_poses[frame_number][0],
        experiment.fish_poses[frame_number][1],
        0.1 * np.cos(-experiment.fish_poses[frame_number][2]),
        0.1 * np.sin(-experiment.fish_poses[frame_number][2]),
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
    if experiment.manual_corrections[frame_number]:
        text += " (manual)"
    ax.text(
        draw_x,
        draw_y - 0 * draw_offset,
        f"Frame : {text}",
        fontsize=draw_fontsize,
        ha=draw_ha,
        va="center",
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
    for x in experiment.robot_poses[frame_number]:
        text += f"{x:.2f},"
    ax.text(
        draw_x - 10,
        draw_y - 2 * draw_offset,
        fr"R",
        fontsize=draw_fontsize,
        ha=draw_ha,
        va="center",
        color=color_robot
    )
    ax.text(
        draw_x,
        draw_y - 2 * draw_offset,
        f"Robot pose: {text}",
        fontsize=draw_fontsize,
        ha=draw_ha,
        va="center",
    )
    text = ""
    for x in experiment.fish_poses[frame_number]:
        text += f"{x:.2f},"
    ax.text(
        draw_x - 10,
        draw_y - 3 * draw_offset,
        fr"F",
        fontsize=draw_fontsize,
        ha=draw_ha,
        va="center",
        color=color_fish
    )
    ax.text(
        draw_x,
        draw_y - 3 * draw_offset,
        f"Fish pose: {text}",
        fontsize=draw_fontsize,
        ha=draw_ha,
        va="center",
    )

    ax.set_xticks([])
    ax.set_yticks([])


def manual_fix(experiment: Experiment, 
                   start_frame: int,
                   end_frame: int,
                   output_folder: str, 
                   fix_output_folder: str):
    n_frames = end_frame - start_frame

    cap = cv2.VideoCapture(output_folder + "/output.mp4")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    n_fig_rows = 5
    n_fig_cols = 2

    title_fontsize = 16
    with PdfPages(f"{fix_output_folder}/{start_frame}-{end_frame}.pdf") as pdf:
        for i in range(start_frame, end_frame, 1):
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

            axes[0, 0].set_title(r"Video", fontsize=title_fontsize)
            axes[0, 1].set_title(r"Info", fontsize=title_fontsize)

            for j in range(n_fig_rows):
                if (i - start_frame) + j >= n_frames:
                    break
                ret, frame = cap.read()
                # Frame
                ax = axes[j, 0]
                ax.imshow(frame)

                ax.set_xticks([])
                ax.set_yticks([])

                # Experiment info
                ax = axes[j, 1]
                plot_world(ax, experiment, i + j)

        pdf.savefig()
        plt.close(fig)

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

    manual_fix(experiment, 
                   variables["start_frame"],
                   variables["end_frame"],
                   output_folder, 
                   fix_output_folder)
