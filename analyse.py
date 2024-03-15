import os
import pickle
import argparse

from pre_process.environment import Environment
from post_process.analysis import compute_results
from priors.foldering import get_data_folders


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent_folder", type=str, default="June13_2")
    parser.add_argument("--experiment_datetime", type=str, default="20230613-165310")

    args = parser.parse_args()
    return args

def run(
    parent_folder,
    experiment_datetime,
    output_folder = "results",
    output_filename = "results",
):
    folder, exp_folder = get_data_folders(parent_folder, experiment_datetime)
    experiment = pickle.load(open(exp_folder + "/experiment.pkl", "rb"))

    result_folder = f"{folder}/{output_folder}"
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    
    compute_results(
        experiment.trials,
        experiment.fish_poses,
        experiment.robot_poses,
        experiment.period,
        experiment.tracking_area_corners,
        result_folder,
        output_filename,
    )


if __name__ == "__main__":
    args = parse_args()
    variables = vars(args)

    # if file raw_measurements.pkl exists, load it
    # folder, exp_folder = get_data_folders(
    #     variables["parent_folder"], variables["experiment_datetime"]
    # )
    # result_folder = folder + "/results"
    # if os.path.exists(f"{result_folder}/raw_measurements.pkl"):
    #     raw_measurements = pd.read_pickle(f"{result_folder}/raw_measurements.pkl")
    # else:
    run(**variables)
