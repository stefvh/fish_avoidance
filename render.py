import argparse
import pickle

from priors.foldering import get_data_folders
import post_process.rendering as rendering
from core.experiment import Experiment


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent_folder", type=str, default="June13_2")
    parser.add_argument("--experiment_datetime", type=str, default="20230613-165310")
    parser.add_argument("--trial_id", type=int, default=0)

    args = parser.parse_args()
    return args


def run(
    parent_folder,
    experiment_datetime,
    trial_id,
):
    folder, exp_folder = get_data_folders(parent_folder, experiment_datetime)
    experiment = pickle.load(open(exp_folder + "/experiment.pkl", "rb"))

    output_folder = f"{folder}/output"

    rendering.render_trial_from_experiment(
        output_folder,
        "tracking",
        experiment,
        trial_id,
    )

if __name__ == "__main__":
    args = parse_args()
    variables = vars(args)
    run(**variables)
