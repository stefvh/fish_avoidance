import os

import numpy as np
import pandas as pd
from analysis.frame_selection import get_peak_frames
from pre_process.experiment import Experiment, get_experiment
from utils.parser import get_default_parser

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from utils.maths.probabilities import (
    compute_joint_conditional_probability,
    compute_conditional_occurences,
)


def parse_args():
    parser = get_default_parser()

    parser.add_argument(
        "-tf",
        "--trials_filter",
        type=str,
        default="all",
    )

    return parser.parse_args()


def plot_jc(
    filename,
    jc_probabilities,
    x_bins,
    y_bins,
    z_names,
):
    n_states, n_x_states, n_y_states = jc_probabilities.shape

    fig, axes = plt.subplots(1, n_states, figsize=(16, 12))

    for i in range(n_states):
        ax = axes[i]
        # plot jc as main figure

        im = ax.imshow(
            jc_probabilities[i].T,
            cmap="GnBu_r",
            origin="lower",
            interpolation="none",
            # extent=[2.5, 102.5, 2.5, 102.5],
            vmin=0,
            vmax=0.12,
        )
        ax.set_xticks(np.arange(n_x_states))
        xticklabels = [f"{x:.1f}" for x in x_bins]
        ax.set_xticklabels(xticklabels)

        ax.set_yticks(np.arange(n_y_states))
        yticklabels = [f"{y:.1f}" for y in y_bins]
        ax.set_yticklabels(yticklabels)

        ax.set_xlabel(r"$v_f$")
        ax.set_ylabel(r"$w_f$")
        # cbar = fig.colorbar(im, ax=ax)

        # plot marginal probabilities histograms
        divider = make_axes_locatable(ax)
        axbottom = divider.append_axes("bottom", size=1, pad=0.3, sharex=ax)
        axleft = divider.append_axes("left", size=1, pad=0.4, sharey=ax)

        axbottom.set_xlabel(r"$v_f$", fontsize=16)
        axleft.set_ylabel(r"$w_f$", fontsize=16)

        # marginal probabilities
        p_x = np.sum(jc_probabilities[i], axis=1)
        p_y = np.sum(jc_probabilities[i], axis=0)

        # histogram covering space between each other
        axbottom.hist(x_bins, bins=x_bins, weights=p_x, orientation="vertical")
        axbottom.set_xlim((x_bins[0], x_bins[-1] - 0.1))
        axbottom.set_ylim((0, 0.2))
        axleft.hist(y_bins, bins=y_bins, weights=p_y, orientation="horizontal")
        axleft.set_ylim((y_bins[0], y_bins[-1] - 6.2))
        axleft.set_xlim((0, 1.0))

        ax.set_title(f"State {z_names[i]}")

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_co(
    filename,
    co_probabilities,
    x_bins,
    z_names,
):
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    N_Z = sum([co_probabilities[i]["N_Z"] for i in range(len(co_probabilities))])

    for i in range(len(co_probabilities)):
        q_x = co_probabilities[i]["Q_X"] / N_Z
        n_x_states = len(q_x)
        ax.hist(
            x_bins,
            bins=x_bins,
            weights=q_x,
            orientation="vertical",
            label=z_names[i],
            alpha=0.5,
        )
        ax.set_xticks(np.arange(n_x_states))
        xticklabels = [f"{x:.1f}" for x in x_bins]
        ax.set_xticklabels(xticklabels)

    ax.set_xlabel(r"$v_f$")
    ax.set_ylabel(r"$Q_X$")

    ax.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def run_states(experiment: Experiment, output_folder: str):
    results = experiment.results
    robot_states = experiment.robot_states

    results = results.merge(robot_states, on="frame_number", how="left")

    results = results[
        ["frame_number", "fish_velocity_batch", "fish_turn_batch", "state"]
    ]
    results = results.dropna(subset=["fish_velocity_batch", "fish_turn_batch", "state"])

    results = results[
        results["state"].isin(["APPROACHING_SLOW", "APPROACHING_FAST", "INITIALIZING"])
    ]
    results["b_state"] = results["state"].apply(
        lambda x: 1 if "APPROACHING" in x else 0
    )

    batch = 5
    v_f = results["fish_velocity_batch"]
    w_f = results["fish_turn_batch"] / (batch * experiment.period)

    r_s = results["b_state"]

    v_f = v_f.to_numpy()
    w_f = w_f.to_numpy()
    r_s = r_s.to_numpy()

    v_f_min, v_f_max = np.min(v_f), np.max(v_f)
    v_f_bins = np.linspace(v_f_min, v_f_max, 20)
    w_f_min, w_f_max = np.min(w_f), np.max(w_f)
    w_f_bins = np.linspace(w_f_min, w_f_max, 10)

    p = compute_joint_conditional_probability(
        x=v_f, y=w_f, z=r_s, x_bins=v_f_bins, y_bins=w_f_bins
    )

    plot_jc(
        filename=f"{output_folder}/joint_conditional_probability.pdf",
        jc_probabilities=p,
        x_bins=v_f_bins,
        y_bins=w_f_bins,
        z_names=["INITIALIZING", "APPROACHING"],
    )

    q = compute_conditional_occurences(x=v_f, z=r_s, x_bins=v_f_bins)

    plot_co(
        filename=f"{output_folder}/conditional_occurences.pdf",
        co_probabilities=q,
        x_bins=v_f_bins,
        z_names=["INITIALIZING", "APPROACHING"],
    )


def run_timewindow(experiment: Experiment, filter_frames: list, output_folder: str):
    output_folder = f"{output_folder}/timewindow"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    results = experiment.results

    #
    results = results[results["frame_number"].isin(filter_frames)]

    results = results[
        [
            "trial_id",
            "frame_number",
            "fish_velocity_batch",
            "fish_turn_batch",
            "robot_distance",
        ]
    ]
    results = results.dropna(subset=["fish_velocity_batch", "fish_turn_batch"])

    peak_frames = get_peak_frames(results)

    results["b_state"] = results["frame_number"].apply(
        lambda x: 1 if x in peak_frames else 0
    )

    batch = 5
    v_f = results["fish_velocity_batch"]
    w_f = results["fish_turn_batch"] / (batch * experiment.period)

    r_s = results["b_state"]

    v_f = v_f.to_numpy()
    w_f = w_f.to_numpy()
    r_s = r_s.to_numpy()

    v_f_min, v_f_max = np.min(v_f), np.max(v_f)
    v_f_bins = np.linspace(v_f_min, v_f_max, 20)
    w_f_min, w_f_max = np.min(w_f), np.max(w_f)
    w_f_bins = np.linspace(w_f_min, w_f_max, 10)

    p = compute_joint_conditional_probability(
        x=v_f, y=w_f, z=r_s, x_bins=v_f_bins, y_bins=w_f_bins
    )

    plot_jc(
        filename=f"{output_folder}/joint_conditional_probability.pdf",
        jc_probabilities=p,
        x_bins=v_f_bins,
        y_bins=w_f_bins,
        z_names=["REST", "PEAK"],
    )

    q = compute_conditional_occurences(x=v_f, z=r_s, x_bins=v_f_bins)

    plot_co(
        filename=f"{output_folder}/conditional_occurences.pdf",
        co_probabilities=q,
        x_bins=v_f_bins,
        z_names=["REST", "PEAK"],
    )

    return


def run(experiment: Experiment, trials_filter: str):
    root_folder = experiment.input_folder
    output_folder = f"{root_folder}/plots"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if trials_filter == "all":
        frame_numbers = range(
            experiment.trials.iloc[0]["start_frame"],
            experiment.trials.iloc[-1]["end_frame"],
        )
    elif trials_filter == "peak_repulsion":
        filepath = f"plotting/hypotheses/peak_repulsion_ids/{experiment.experiment_datetime}.csv"
        cv_input = pd.read_csv(filepath)
        trial_ids = list(cv_input["trial_id"])

        frame_numbers = []
        for trial_id in trial_ids:
            trial = experiment.trials[experiment.trials["trial_id"] == trial_id].iloc[0]
            frame_numbers += list(range(trial["start_frame"], trial["end_frame"]))

    # run_states(experiment, output_folder)

    run_timewindow(experiment, frame_numbers, output_folder)


if __name__ == "__main__":
    args = parse_args()
    variables = vars(args)

    experiment = get_experiment(**variables)
    run(experiment, args.trials_filter)
