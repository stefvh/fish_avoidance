import operator
import os

from matplotlib import pyplot as plt
import numpy as np
from analysis.frame_selection import get_distance_dip_frames, get_filtered_frames
from pre_process.experiment import Experiment, get_experiment
from utils.parser import get_default_parser

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def parse_args():
    parser = get_default_parser()

    parser.add_argument(
        "-tf",
        "--trials_filter",
        type=str,
        default="all",
    )

    return parser.parse_args()


def run(experiment: Experiment, trials_filter: str):
    root_folder = experiment.input_folder
    output_folder = f"{root_folder}/plots/distance_regression/{trials_filter}"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    filter_frames = get_filtered_frames(experiment, trials_filter)

    results = experiment.results
    results = results[results["frame_number"].isin(filter_frames)]

    dip_frames_per_trial, dip_values = get_distance_dip_frames(results, frame_window=15)

    max_af_values = []
    for dip_frames in dip_frames_per_trial:
        # find max fish acceleration in the dip frames
        dip_results = results[results["frame_number"].isin(dip_frames)]
        max_fish_acceleration = dip_results["fish_acceleration_batch"].max()
        max_af_values.append(max_fish_acceleration)
    
    fig, ax = plt.subplots()
    labels = [f"{trial_id}" for trial_id in results["trial_id"].unique()]
    ax.scatter(dip_values, max_af_values)
    for i, txt in enumerate(labels):
        ax.annotate(txt, (dip_values[i], max_af_values[i]))
    ax.set_xlabel(r"min distance")
    ax.set_ylabel(r"max fish acceleration")

    #########

    X = np.copy(dip_values).reshape(-1, 1)
    y = np.copy(max_af_values).reshape(-1, 1)

    # reorder the data
    sort_axis = operator.itemgetter(0)
    sorted_zip = sorted(zip(X, y), key=sort_axis)
    X, y = zip(*sorted_zip)

    # Train linear regression model on whole dataset
    lr = LinearRegression()
    lr.fit(X, y)

    # Train polynomial regression model on the whole dataset
    pr = PolynomialFeatures(degree = 4)
    X_poly = pr.fit_transform(X)
    lr_2 = LinearRegression()
    lr_2.fit(X_poly, y)

    #  Predict results
    y_pred_lr = lr.predict(X)           # Linear Regression
    y_pred_poly = lr_2.predict(X_poly)  # Polynomial Regression

    plt.plot(X, lr.predict(X), color = 'firebrick')
    plt.plot(X, lr_2.predict(X_poly), color = 'green')

    plt.savefig(f"{output_folder}/distance_regression.pdf")
    

if __name__ == "__main__":
    args = parse_args()
    variables = vars(args)
    experiment = get_experiment(**variables)
    run(experiment, args.trials_filter)