import os
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde

from core.experiment import Experiment
from core.study import study

from analysis.clean.reactions import extract_trials, get_baseline_data_subjects

from analysis.clean.avoidance_mechanisms import get_data
from plotting.publishing.subset_trials import get_trials_by_data


def pre_encounter_analysis():
    exp_trials = extract_trials()

    df_baseline = get_baseline_data_subjects()
    print(df_baseline.shape)
    distance_bins = np.arange(40, -5, -5)  # Generates [40, 35, 30, ..., 5, 0]



if __name__ == "__main__":
    pre_encounter_analysis()
