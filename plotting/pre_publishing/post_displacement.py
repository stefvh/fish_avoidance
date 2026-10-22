from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import statsmodels

from analysis.one.wrapper_fixed import run_plm_model
from core.experiment import Experiment
from core.study import study

from utils.maths.statistical import spearman_confidence_interval, spearman_monte_carlo
import utils.plotter as plotter

from scipy import stats
import statsmodels.formula.api as smf

import seaborn as sns

from scipy import stats

from plotting.publishing.subset_trials import get_trials

def get_min_post_duration(exp_trials):
    delta_t = np.inf

    for exp_id, trials in enumerate(exp_trials):
        for trial in trials:
            distances = trial["data"]["distance"].to_numpy()

            # Measurements
            ef = np.argmin(distances)
            max_t = distances.shape[0]
            td = 10
            _dt = max_t - (ef + td + 1)
            if _dt < delta_t:
                delta_t = _dt
    
    return delta_t
            

def get_values(exp_trials, delta_t):
    displacements = {}
    apvs = {}
    encounter_distances = {}

    delta_t = 50

    for exp_id, trials in enumerate(exp_trials):
        displacements[exp_id] = []
        apvs[exp_id] = []
        encounter_distances[exp_id] = []
        for trial in trials:
            distances = trial["data"]["distance"].to_numpy()
            np_apv = trial["data"]["apv"].to_numpy()

            ef = np.argmin(distances)
            max_t = distances.shape[0]
            td = 10

            ph1 = (0, max(0, ef - td))
            ph2 = (max(0, ef - td), ef + td + 1)
            start_post = ef + td + 1

            if start_post + delta_t >= max_t:
                displacements[exp_id].append(np.nan)
                apvs[exp_id].append(np.nan)
                encounter_distances[exp_id].append(np.nan)
                continue

            encounter_distance = np.min(distances[ph2[0]:ph2[1]])
            encounter_distances[exp_id].append(encounter_distance)

            apv = np.max(np_apv[ph2[0]:ph2[1]])
            apvs[exp_id].append(apv)

            start_distance = distances[start_post]

            end_post = start_post + delta_t - 1
            end_distance = distances[end_post]

            displacement = end_distance - start_distance
            displacements[exp_id].append(displacement)

    return displacements, apvs, encounter_distances

def plot_distributions(exp_values):
    for exp_id, values in exp_values.items():
        print(f"Subject {exp_id}: {np.mean(values)}, {np.std(values)}")

    fig, axes = plt.subplots(1, 5, figsize=(22, 4))

    for exp_id, values in exp_values.items():
        ax = axes[exp_id]
        ax.hist(values, bins=10)
        ax.set_title(f"Subject {exp_id}")
        # ax.set_xlim(0, 12)
    
    fig.tight_layout()
    fig.savefig(f"{study.plot_output_folder}/TEST-PUBLISH/displacements.pdf")

def create_dataframe(exp_trials, disp, apv, enc_dist):
    data = {
        "experiment_id": [],
        "trial_id": [],
        "time": [],
        "displacement": [],
        "approach_speed": [],
        "encounter_distance": []
    }

    for exp_id, trials in enumerate(exp_trials):
        for i, trial in enumerate(trials):
            m_trial_id = trial["m_trial_id"]
            start_frame = trial["start_frame"]
            displacement = disp[exp_id][i]
            approach_speed = apv[exp_id][i]
            encounter_distance = enc_dist[exp_id][i]
            data["experiment_id"].append(exp_id)
            data["trial_id"].append(m_trial_id)
            data["time"].append(start_frame)
            data["displacement"].append(displacement)
            data["approach_speed"].append(approach_speed)
            data["encounter_distance"].append(encounter_distance)

    df = pd.DataFrame(data)

    df = df.dropna()

    return df

def statistics(data):
    # Fit the mixed effects linear model (random intercept for experiment_id)
    model = smf.mixedlm("displacement ~ trial_id + approach_speed + encounter_distance", data, groups=data['experiment_id'], re_formula="~1").fit()

    # Print the summary of the model
    print(model.summary())

    # Fit the null model (intercept-only model)
    null_model = smf.mixedlm("displacement ~ 1", data, groups=data['experiment_id'], re_formula="~1").fit()

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


def fe_model(data):
    result = run_plm_model(
        data,
        response_var="displacement",
        predictor_vars=[
            "approach_speed",
            "encounter_distance"
        ],
        indexes=["experiment_id", "trial_id"],
        model_type="within",
        effect="twoways"
    )
    print(result)

def plot_scatter_displacement_approach_speed_per_exp(df):
    fig, axes = plt.subplots(1, 5, figsize=(22, 4))

    for exp_id in range(5):
        ax = axes[exp_id]
        exp_df = df[df["experiment_id"] == exp_id]
        ax.scatter(exp_df["approach_speed"], exp_df["displacement"])
        ax.set_title(f"Subject {exp_id}")
        ax.set_xlabel("Approach Speed")
        ax.set_ylabel("Displacement")
    
    fig.tight_layout()
    fig.savefig(f"{study.plot_output_folder}/TEST-PUBLISH/scatter_displacement_approach_speed.pdf")

def plot_distributions_of_speed_fish_and_displacement_per_exp(df):
    fig, axes = plt.subplots(2, 5, figsize=(22, 10))

    d_sf = []
    d_sf_std = []
    d_disp = []
    d_disp_std = []

    for exp_id in range(5):
        ax = axes[0, exp_id]
        exp_df = df[df["experiment_id"] == exp_id]

        # Get speed_fish data and normalize it
        speed_fish_data = study.data[study.data["experiment_id"] == exp_id]["speed_fish"]
        # speed_fish_min, speed_fish_max = speed_fish_data.min(), speed_fish_data.max()
        # speed_fish_normalized = (speed_fish_data - speed_fish_min) / (speed_fish_max - speed_fish_min)
        # Plot normalized speed_fish on the primary y-axis
        sns.histplot(speed_fish_data, ax=ax, bins=10, color="red", alpha=0.5)
        ax.set_ylabel("Speed Fish Count", color="red")
        ax.tick_params(axis="y", labelcolor="red")

        d_sf.append(speed_fish_data.mean())
        d_sf_std.append(speed_fish_data.std())

        ax.set_xlim(0, 20)

        ax = axes[1, exp_id]
        sns.histplot(exp_df["displacement"], ax=ax, bins=10, color="blue", alpha=0.5)
        ax.set_ylabel("Displacement Count", color="blue")
        ax.tick_params(axis="y", labelcolor="blue")

        d_disp.append(exp_df["displacement"].mean())
        d_disp_std.append(exp_df["displacement"].std())

    fig.tight_layout()
    fig.savefig(f"{study.plot_output_folder}/TEST-PUBLISH/distributions_speed_fish_displacement.pdf")

    # # Assume experiment_ids, c_ft, t_ft are already defined
    # plt.figure(figsize=(10, 6))

    # # Perform a paired t-test
    # t_stat, p_val = stats.ttest_rel(d_sf, d_disp)

    # # Output the results
    # print(f"T-statistic: {t_stat}")
    # print(f"P-value: {p_val}")

    # # Paired Line Plot
    # for sf, disp in zip(d_sf, d_disp):
    #     plt.plot([0, 1], [sf, disp], marker='o', color='gray', linestyle='-', linewidth=1)
        
    # # Markers for control and test values
    # plt.scatter(np.zeros_like(d_sf), d_sf, color='blue', label='Fish speed', zorder=5)
    # plt.scatter(np.ones_like(d_disp), d_disp, color='red', label='Displacement', zorder=5)

    # # Labels and formatting
    # plt.xticks([0, 1], ['Fish speed', 'Displacement'])
    # plt.title('Paired Comparison of Fish speed and Displacement')
    # plt.legend()
    # plt.grid(True)
    # plt.savefig(f"{study.plot_output_folder}/TEST-PUBLISH/sf_disp_paired_comparison.pdf")

    # Perform Spearman's rank correlation
    corr_coef, p_val = stats.spearmanr(d_sf, d_disp)

    # Output Spearman Correlation and P-value
    print(f"Spearman Correlation Coefficient: {corr_coef} (Effect size)")
    print(f"P-value: {p_val}")

    lower_ci, upper_ci = spearman_confidence_interval(d_sf, d_disp)
    print(f"95% Confidence Interval for Spearman Correlation: [{lower_ci}, {upper_ci}]")

    mean_corr, std_corr, ci_lower, ci_upper = spearman_monte_carlo(d_sf, d_sf_std, d_disp, d_disp_std)

    print(f"Monte Carlo Simulation Results:")
    print(f"Mean Correlation: {mean_corr}")
    print(f"Standard Deviation of Correlation: {std_corr}")
    print(f"95% Confidence Interval: [{ci_lower}, {ci_upper}]")

    # plot speed_fish on x and displacement on y
    plt.figure(figsize=(10, 6))
    plt.scatter(d_sf, d_disp)

    plt.errorbar(d_sf, d_disp, 
             xerr=d_sf_std, yerr=d_disp_std, 
             fmt='o', capsize=5, color='blue', ecolor='gray', 
             elinewidth=1, markeredgewidth=1)

    plt.xlabel("Speed Fish")
    plt.ylabel("Displacement")
    plt.title("Speed Fish vs Displacement")
    plt.grid(True)
    plt.savefig(f"{study.plot_output_folder}/TEST-PUBLISH/sf_disp_scatter.pdf")


def plot_correlation_fish_speed_displacement(df):
    d_sf = []
    d_disp = []

    for exp_id in range(5):
        exp_df = df[df["experiment_id"] == exp_id]
        speed_fish_data = study.data[study.data["experiment_id"] == exp_id]["speed_fish"]
        d_sf.append(speed_fish_data.mean())
        d_disp.append(exp_df["displacement"].mean())

    # Perform Spearman's rank correlation
    corr_coef, p_val = stats.spearmanr(d_sf, d_disp)

    # Output Spearman Correlation and P-value
    print(f"Spearman Correlation Coefficient: {corr_coef} (Effect size)")
    print(f"P-value: {p_val}")

    lower_ci, upper_ci = spearman_confidence_interval(np.array(d_sf), np.array(d_disp))
    print(f"95% Confidence Interval for Spearman Correlation: [{lower_ci}, {upper_ci}]")

    # plot speed_fish on x and displacement on y
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(d_sf, d_disp, s=40, marker="x", color="tab:red")
    ax.set_xlabel(r"mean fish speed $\bar{v}_F$", fontsize=16)
    ax.set_ylabel(r"mean displacement $\bar{D}_F$", fontsize=16)

    # label each point with the experiment id
    for i, txt in enumerate(range(5)):
        text = f"{txt+1}"
        offset = 0.1
        ax.annotate(text, (d_sf[i] + offset, d_disp[i] + offset), fontsize=14)

    # Text box with correlation coefficient and p-value
    textstr = "Spearman correlation:\n" + rf"$r_s$ = {corr_coef:.2f}" + \
                "\n" + rf"$p$ = {p_val:.2f}"
    #  "\n"+ rf"$95\%$ CI = [" + f"{pearson_conf[0]:.2f}, {pearson_conf[1]:.2f}]" + \
    props = dict(boxstyle='square', facecolor='wheat', alpha=0., edgecolor='black')
    ax.text(0.6, 0.15, textstr, transform=ax.transAxes, fontsize=14,
                  verticalalignment='center', bbox=props)

    ax.set_xlim(2, 9)
    ax.set_ylim(4, 20)

    inset = fig.add_axes([0.15, 0.6, 0.4, 0.3])
    avoidances = [61.82467275, 83.55957768, 89.98382401, 51.03106376, 85.58078887]

    # plot d_disp against avoidances
    inset.scatter(avoidances, d_disp, s=40, marker="x", color="tab:blue")
    inset.set_xlabel(r"Time of avoidance (\%)", fontsize=14)
    inset.set_yticks([])
    # inset.set_ylabel(r"mean displacement $\bar{D}_F$", fontsize=12)

    # label each point with the experiment id
    for i, txt in enumerate(range(5)):
        text = f"{txt+1}"
        offset = 0.1
        inset.annotate(text, (avoidances[i] + 1.0, d_disp[i] - 1.0), fontsize=14)

    inset.set_xlim(45, 95)
    inset.set_ylim(4, 20)

    fig.tight_layout()

    # plt.grid(True)
    fig.savefig(f"{study.plot_output_folder}/PUBLISH/displacement_correlation.pdf")

    # avoidances = [61.82467275, 83.55957768, 89.98382401, 51.03106376, 85.58078887]

    # # plot d_disp against avoidances
    # fig, ax = plt.subplots(figsize=(6, 4))
    # ax.scatter(avoidances, d_disp, s=40, marker="x", color="tab:red")
    # ax.set_xlabel(r"mean avoidance $\bar{A}_F$", fontsize=16)
    # ax.set_ylabel(r"mean displacement $\bar{D}_F$", fontsize=16)

    # # label each point with the experiment id
    # for i, txt in enumerate(range(5)):
    #     text = f"{txt+1}"
    #     offset = 0.1
    #     ax.annotate(text, (avoidances[i] + offset, d_disp[i] + offset), fontsize=14)

    # # Perform Spearman's rank correlation
    # corr_coef, p_val = stats.spearmanr(avoidances, d_disp)

    # # Output Spearman Correlation and P-value
    # print(f"Spearman Correlation Coefficient: {corr_coef} (Effect size)")
    # print(f"P-value: {p_val}")

    # lower_ci, upper_ci = spearman_confidence_interval(np.array(avoidances), np.array(d_disp))
    # print(f"95% Confidence Interval for Spearman Correlation")
          
    # Text box with correlation coefficient and p-value

    # fig.tight_layout()

    # # plt.grid(True)
    # fig.savefig(f"{study.plot_output_folder}/TEST-PUBLISH/displacement_avo_correlation.pdf")

if __name__ == "__main__":
    exp_trials = get_trials()
    delta_t = get_min_post_duration(exp_trials)
    print(delta_t)
    exp_displacements, exp_apvs, exp_encounter_distances = get_values(exp_trials, delta_t)
    # plot_distributions(exp_displacements)
    df = create_dataframe(exp_trials, exp_displacements, exp_apvs, exp_encounter_distances)
    # statistics(df)
    # fe_model(df)
    # plot_scatter_displacement_approach_speed_per_exp(df)
    # plot_distributions_of_speed_fish_and_displacement_per_exp(df)
    plot_correlation_fish_speed_displacement(df)