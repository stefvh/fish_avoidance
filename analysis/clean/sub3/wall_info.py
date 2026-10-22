from matplotlib import pyplot as plt
import numpy as np

from core.experiment import Experiment, get_experiment_by_id
from core.study import study
from pre_process.environment import Environment

from analysis.clean.reactions import extract_trials
from analysis.clean.sub3.help import save_plot

from analysis.clean.avoidance_mechanisms import get_data

import seaborn as sns


def measure_wall_side():
    df = get_data()
    df = df[(df["v_avoid"] > 0) & (df["speed_fish"] > 2)]

    experiments = [get_experiment_by_id(exp_id) for exp_id in [0, 1, 2, 3, 4]]

    measurements = []
    for exp in experiments:
        df_exp = df[df["experiment_id"] == exp.id]
        environment = Environment(exp.corners)
        for i, row in df_exp.iterrows():
            dict = environment.get_wall_side((row["x_fish"], row["y_fish"]), row["heading_fish"])
            wall_sides = dict["wall_sides"]
            # wall_sides is a dict, i need the values
            d = wall_sides.values()
            measurements.extend(list(d))
    
    # Count the amount of 'left' and 'right'
    left = measurements.count("left")
    right = measurements.count("right")
    # Total amount
    total = left + right
    print(f"Left: {left/total*100:.2f}%")
    print(f"Right: {right/total*100:.2f}%")


if __name__ == "__main__":
    measure_wall_side()