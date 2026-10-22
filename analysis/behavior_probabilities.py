import os
import numpy as np

from core.experiment import Experiment, get_experiment

from utils.parser import get_default_parser
from utils.maths.probabilities import (
    compute_joint_conditional_probability,
    compute_conditional_occurences,
)

from plotting.general.prob_jc import plot as plot_jc
from plotting.general.histograms import plot_co

from analysis.frame_selection import get_filtered_frames
import analysis.conditioning as conditioning


def parse_args():
    parser = get_default_parser()

    parser.add_argument(
        "-tf",
        "--trials_filter",
        type=str,
        default="all",
    )
    parser.add_argument(
        "-c",
        "--condition",
        type=str,
        default="robot_state",
    )

    return parser.parse_args()

def analyse_vf_wf_z(
    results,
    z_names,
    output_folder,
    batch=5,
    period=0.04,
):
    v_f = results["fish_velocity_batch"]
    w_f = results["fish_turn_batch"] / (batch * period)
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
        x_label=r"$v_f$",
        y_label=r"$w_f$",
        z_names=z_names,
    )

    q = compute_conditional_occurences(x=v_f, z=r_s, x_bins=v_f_bins)

    plot_co(
        filename=f"{output_folder}/conditional_occurences.pdf",
        co_probabilities=q,
        x_bins=v_f_bins,
        x_label=r"$v_f$",
        y_label=r"$Q(X \mid Z)$",
        z_names=z_names,
    )


def run_by_robot_state(experiment: Experiment, filter_frames: list, output_folder: str):
    output_folder = f"{output_folder}/by_robot_state"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    results = experiment.results
    results = results[results["frame_number"].isin(filter_frames)]

    results, z_names = conditioning.by_robot_state(results, experiment.robot_states)

    analyse_vf_wf_z(results, z_names, output_folder,
                    batch=5, period=experiment.period)


def run_by_time_near_distance_dip(experiment: Experiment, filter_frames: list, output_folder: str):
    output_folder = f"{output_folder}/by_time_near_distance_dip"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    results = experiment.results
    results = results[results["frame_number"].isin(filter_frames)]

    results, z_names = conditioning.by_time_near_distance_dip(results)

    analyse_vf_wf_z(results, z_names, output_folder,
                    batch=5, period=experiment.period)


def run(experiment: Experiment, condition:str, trials_filter: str):
    root_folder = experiment.input_folder
    output_folder = f"{root_folder}/plots/behavior_probabilities/{trials_filter}"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    frame_numbers = get_filtered_frames(experiment, trials_filter)

    if condition == "robot_state":
        run_by_robot_state(experiment, frame_numbers, output_folder)
    elif condition == "time_near_distance_dip":
        run_by_time_near_distance_dip(experiment, frame_numbers, output_folder)
    else:
        raise ValueError(f"Invalid condition: {condition}")


if __name__ == "__main__":
    args = parse_args()
    variables = vars(args)

    experiment = get_experiment(**variables)
    run(experiment, args.condition, args.trials_filter)
