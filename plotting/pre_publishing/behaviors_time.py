from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels
import statsmodels.genmod
import statsmodels.genmod.families
import statsmodels.stats
import statsmodels.stats.anova

from core.experiment import Experiment
from core.study import study

from analysis.clean.sub5.speed_distance import get_speed_distance_by_arb
from analysis.one.wrapper_fixed import run_plm_model
from analysis.one.wrapper_fixed_gen import run_pglm_model
from core.experiment import Experiment
from core.study import study

import networkx as nx

import utils.plotter as plotter

import statsmodels.formula.api as smf

def get_trials():
    remove_exp_trials = [
        [10, 21],
        [2, 11, 12, 13, 15],
        [8],
        [],
        []
    ]
    data = study.data

    exp_trials = []

    for exp in study.list_experiments:
        exp_id = exp.id
        experiment_datetime = exp.experiment_datetime

        trials = []
        remove_trials = remove_exp_trials[exp_id]
        df = pd.read_csv(f"pre_process/manual_trials/{experiment_datetime}.csv")
        for i, row in df.iterrows():
            if row["trial_id"] in remove_trials:
                continue
            trial_data = data[
                (data["experiment_id"] == exp_id)
                & (data["frame"] >= row["start_frame"])
                & (data["frame"] <= row["end_frame"])
            ]
            if trial_data.shape[0] != row["end_frame"] - row["start_frame"] + 1:
                print("Missing frames in trial data")
                print(f"Experiment ID: {exp_id}, Trial ID: {i}")
                print(
                    f"Expected: {row['end_frame'] - row['start_frame'] + 1}, Actual: {trial_data.shape[0]}"
                )
            else:
                trial = {
                    "experiment_id": exp_id,
                    "trial_id": i,
                    "m_trial_id": row["trial_id"],
                    "start_frame": row["start_frame"],
                    "end_frame": row["end_frame"],
                    "data": trial_data,
                }
                trials.append(trial)
        exp_trials.append(trials)
    
    print("-------------")

    return exp_trials


def mapping(rev, speed_fish):
    if speed_fish < 2:
        return 1
    if rev >= 0:
        return 2
    else:
        return 0

def get_map_dict(exp_trials):
    map_dict = {}

    for exp_id, trials in enumerate(exp_trials):
        map_dict[exp_id] = []
        for trial in trials:
            z_values = trial["data"]["distance"].to_numpy()
            rev_values = trial["data"]["rev"].to_numpy()
            speed_fish_values = trial["data"]["speed_fish"].to_numpy()

            # Measurements
            ef = np.argmin(z_values)
            max_t = z_values.shape[0]
            td = 10
            ph1 = (0, max(0, ef - td))
            ph2 = (max(0, ef - td), ef + td + 1)
            ph3 = (ef + td + 1, max_t)
            phases = [ph1, ph2, ph3]
            revs_by_avg = []
            speeds_by_avg = []
            for l, phase in enumerate(phases):
                avg_rev = np.mean(rev_values[phase[0]:phase[1]])
                revs_by_avg.append(avg_rev)
                avg_speed = np.mean(speed_fish_values[phase[0]:phase[1]])
                speeds_by_avg.append(avg_speed)
            
            k = mapping(revs_by_avg[0], speeds_by_avg[0])
            l = mapping(revs_by_avg[1], speeds_by_avg[1])
            m = mapping(revs_by_avg[2], speeds_by_avg[2])

            map_dict[exp_id].append((k, l, m))

    return map_dict

def plot_for_each_expid_the_k_value_over_trials(exp_trials, map_dict):
    for exp_id, trials in enumerate(exp_trials):
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        for i, trial in enumerate(trials):
            m_trial_id = trial["m_trial_id"]
            k_value = map_dict[exp_id][i][0]
            ax.scatter(m_trial_id, k_value, label=f"Trial {m_trial_id}")
        ax.set_title(f"Experiment {exp_id}")
        ax.set_xlabel("Trial ID")
        ax.set_ylabel("K value")
        # ax.legend()
        fig.savefig(f"{study.plot_output_folder}/TEST-PUBLISH/exp{exp_id}_k_value.pdf")

def create_dataframe(exp_trials, map_dict):
    data = {
        "experiment_id": [],
        "trial_id": [],
        "time": [],
        "avoidance": [],
        "robot_speed": [],
    }

    for exp_id, trials in enumerate(exp_trials):
        for i, trial in enumerate(trials):
            m_trial_id = trial["m_trial_id"]
            start_frame = trial["start_frame"]
            k, l, m = map_dict[exp_id][i]
            data["experiment_id"].append(exp_id)
            data["trial_id"].append(m_trial_id)
            data["time"].append(start_frame)
            avoidance = 1 if k > 1 else 0
            data["avoidance"].append(avoidance)
            robot_speed, _ = get_speed_distance_by_arb(trial["data"])
            data["robot_speed"].append(robot_speed)

    df = pd.DataFrame(data)

    return df

def statistics(data):
    # Fit a mixed effects logistic regression model (with random intercept for subjects)
    model = smf.glm("avoidance ~ trial_id + robot_speed + (1 | experiment_id)", data, groups=data['experiment_id'], family=statsmodels.genmod.families.Binomial()).fit()

    # Print the summary of the model
    print(model.summary())

    # Fit the null model (intercept-only model)
    null_model = smf.glm(formula="avoidance ~ 1", data=data, family=statsmodels.genmod.families.Binomial()).fit()

    # Likelihood Ratio Test
    LR_stat = 2 * (model.llf - null_model.llf)  # Likelihood ratio test statistic
    df = model.df_model - null_model.df_model  # Degrees of freedom
    p_value = stats.chi2.sf(LR_stat, df)  # Calculate the p-value using chi-square distribution

    # Calculate repeatability
    individual_effects = model.params["Intercept"]
    var_individual_effects = model.cov_params().loc["Intercept", "Intercept"]
    residuals = model.resid_response
    var_residuals = np.var(residuals)
    total_variance = var_individual_effects + var_residuals
    repeatability = var_individual_effects / total_variance

    print(f"Repeatability: {repeatability}")

    print(f"Likelihood Ratio Test statistic: {LR_stat}")
    print(f"p-value for the model: {p_value}")


def plot_avoidance(df):
    # Group by experiment_id and calculate the cumulative avoidance percentage over trials
    plt.figure(figsize=(10, 6))

    for exp_id in df['experiment_id'].unique():
        # Filter the dataframe for each subject (experiment_id)
        subject_data = df[df['experiment_id'] == exp_id]

        # Calculate the cumulative sum of avoidance and normalize by the number of trials
        subject_data['cumulative_avoidance'] = subject_data['avoidance'].cumsum() / (subject_data['trial_id'].rank(method='first'))

        # Plot the cumulative avoidance percentage for each subject
        plt.plot(subject_data['trial_id'], 
                 subject_data['cumulative_avoidance'], 
                 label=f'{exp_id+1}',
                 color=plotter.colors[exp_id * 2])
    
    fontsize = 16
    tick_fontsize = 13
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)

    plt.xlabel('Trial number', fontsize=fontsize)
    plt.ylabel('Cumulative average of pre-encounter avoidance', fontsize=fontsize)
    legend = plt.legend(title="Subjects",
                # loc='lower right',
                bbox_to_anchor=(0.875, 0.1),
                edgecolor='white',
                fontsize=13)
    legend.get_title().set_fontsize(14)
    plt.tight_layout()
    # plt.grid(True)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=0.5)
    plt.axhline(y=1, color='black', linestyle='-', alpha=0.5, linewidth=0.5)
    plt.savefig(f"{study.plot_output_folder}/TEST-PUBLISH/avoidance_by_trials.pdf")


if __name__ == "__main__":
    exp_trials = get_trials()
    map_dict = get_map_dict(exp_trials)
    # plot_for_each_expid_the_k_value_over_trials(exp_trials, map_dict)
    df = create_dataframe(exp_trials, map_dict)
    statistics(df)
    # plot_avoidance(df)
