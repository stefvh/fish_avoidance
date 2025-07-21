from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import statsmodels

from core.experiment import Experiment
from core.study import study

import utils.plotter as plotter

from scipy import stats
import statsmodels.formula.api as smf


from plotting.publishing.subset_trials import get_trials

def foo(exp_trials):
    values = {}

    for exp_id, trials in enumerate(exp_trials):
        values[exp_id] = []
        for trial in trials:
            distances = trial["data"]["distance"].to_numpy()
            value = np.min(distances)
            values[exp_id].append(value)

    return values

def plot_distributions(exp_values):
    for exp_id, values in exp_values.items():
        print(f"Subject {exp_id}: {np.mean(values)}, {np.std(values)}")

    fig, axes = plt.subplots(1, 5, figsize=(22, 4))

    for exp_id, values in exp_values.items():
        ax = axes[exp_id]
        ax.hist(values, bins=10)
        ax.set_title(f"Subject {exp_id}")
        ax.set_xlim(0, 12)
    
    fig.tight_layout()
    fig.savefig(f"{study.plot_output_folder}/TEST-PUBLISH/encounter_distances.pdf")

def create_dataframe(exp_trials, map_dict):
    data = {
        "experiment_id": [],
        "trial_id": [],
        "time": [],
        "encounter_distance": [],
    }

    for exp_id, trials in enumerate(exp_trials):
        for i, trial in enumerate(trials):
            m_trial_id = trial["m_trial_id"]
            start_frame = trial["start_frame"]
            encounter_distance = map_dict[exp_id][i]
            data["experiment_id"].append(exp_id)
            data["trial_id"].append(m_trial_id)
            data["time"].append(start_frame)
            data["encounter_distance"].append(encounter_distance)

    df = pd.DataFrame(data)

    return df

def statistics(data):
    # Fit the mixed effects linear model (random intercept for experiment_id)
    model = smf.mixedlm("encounter_distance ~ trial_id", data, groups=data['experiment_id'], re_formula="~1").fit()

    # Print the summary of the model
    print(model.summary())

    # Fit the null model (intercept-only model)
    null_model = smf.mixedlm("encounter_distance ~ 1", data, groups=data['experiment_id'], re_formula="~1").fit()

    # Likelihood Ratio Test
    LR_stat = 2 * (model.llf - null_model.llf)  # Likelihood ratio test statistic
    df = model.df_modelwc - null_model.df_modelwc  # Degrees of freedom
    p_value = stats.chi2.sf(LR_stat, df)  # Calculate the p-value using chi-square distribution

    # Extract random effects as an array
    random_effects = np.array([v[0] for v in model.random_effects.values()])

    # Calculate the variance of the random effects
    var_random_effects = np.var(random_effects)

    # Residual variance (from the model fit)
    var_residuals = model.scale  # model.scale provides the residual variance (sigma^2)
    
    # Total variance
    total_variance = var_random_effects + var_residuals

    # Calculate repeatability
    repeatability = var_random_effects / total_variance

    # Print results
    print(f"Repeatability: {repeatability}")
    print(f"Likelihood Ratio Test statistic: {LR_stat}")
    print(f"p-value for the model: {p_value}")

if __name__ == "__main__":
    exp_trials = get_trials()
    exp_values = foo(exp_trials)
    # plot_distributions(exp_values)
    df = create_dataframe(exp_trials, exp_values)
    statistics(df)