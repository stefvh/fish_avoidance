
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats

from core.experiment import Experiment
from core.study import study

from analysis.full.get_peak_data import get_peaks_data
from analysis.full.run_reg_fe import get_sampled_dataframe

from scipy.stats import mannwhitneyu

import seaborn as sns

import utils.plotter as plotter


def plot_tests(control, rv_df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fontsize_label = 18

    # ax[0]: Distributions of fish retreat speed
    ax = axes[0]
    group_1 = control['rev']
    group_2 = rv_df['rv_rev']

    sns.kdeplot(group_1, ax=ax, label=r"$S \setminus S_{acc}$", color="tab:blue")
    sns.kdeplot(group_2, ax=ax, label=r"$S_{acc}$", color="tab:orange")

    statistic, p_value = mannwhitneyu(group_1, group_2)
    print("Mann-Whitney U statistic:", statistic)
    print("p-value:", p_value)
    if p_value < 0.001:
        add_text = r"$p < 0.001$"
    else:
        add_text = f"p = {p_value:.3f}"
    # ax.text(0.65, 0.875, f"Mann-Whitney U test:\n {add_text}", transform=ax.transAxes)
    print(f"p-value: {p_value:.3f}")

    ax.set_ylabel(r"Density", fontsize=fontsize_label)
    ax.set_xlabel(r"$v_{avoid}^F$ (cm/s)", fontsize=20)
    ax.tick_params(axis='both', labelsize=16)

    ax.legend(loc='upper right', fancybox=True, edgecolor='white', fontsize=14)

    ax.text(0.02, 1.05, r"\textbf{A.}", fontsize=16, weight='bold',
        transform=ax.transAxes, verticalalignment='center', horizontalalignment='center')

    group_1_mean = np.mean(group_1)
    group_2_mean = np.mean(group_2)
    std_group1 = np.std(group_1, ddof=1)  # Use ddof=1 for sample standard deviation
    std_group2 = np.std(group_2, ddof=1)
    cohens_d = (group_2_mean - group_1_mean) / np.sqrt((std_group1 ** 2 + std_group2 ** 2) / 2)
    effect_size_units_group1 = cohens_d * std_group1
    print("Effect size in units of group 1:", effect_size_units_group1)
    # ax.text(0.65, 0.80, r"Effect size ($S_{acc} - S_{-acc}$) in units of $S_{-acc}$:" + f"{effect_size_units_group1:.2f}", transform=ax.transAxes)

    # ax[1]: Paired line plot
    ax = axes[1]
    ax.text(0.02, 1.05, r"\textbf{B.}", fontsize=16, weight='bold',
        transform=ax.transAxes, verticalalignment='center', horizontalalignment='center')

    delimiter = 2
    experiment_ids = control['experiment_id'].unique()
    # c_ft is an array for each experiment_id it gives the percentage of rows where rev > 2 during this experiment
    c_ft = np.array([control_rv[control_rv['experiment_id'] == experiment_id]['rev'].gt(delimiter).mean() for experiment_id in experiment_ids])
    t_ft = np.array([rv_df[rv_df['experiment_id'] == experiment_id]['rv_rev'].gt(delimiter).mean() for experiment_id in experiment_ids])
    # Perform a paired t-test
    t_stat, p_val = stats.ttest_rel(t_ft, c_ft)

    # Output the results
    print(f"T-statistic: {t_stat}")
    print(f"P-value: {p_val}")

    # # ax.text(0.7, 0.2, f"Paired t-test:", transform=ax.transAxes)
    # # ax.text(0.7, 0.15, r"$T = $ " + f"{t_stat:.3f}", transform=ax.transAxes)
    # if p_val < 0.001:
    #     add_text = r"$p < 0.001$"
    # elif p_val < 0.01:
    #     add_text = r"$p < 0.01$"
    # elif p_val < 0.05:
    #     add_text = r"$p < 0.05$"
    # else:
    #     add_text = f"p = {p_val:.3f}"
    # full_text = "Paired t-test:\n" + r"$T = $ " + f" {t_stat:.3f}\n" + add_text
    # ax.text(0.7, 0.1, f"{full_text}", transform=ax.transAxes)

    x_points = [0.2, 0.8]

    ax.axvline(x_points[0], color='gray', linestyle='-', linewidth=0.8)
    ax.axvline(x_points[1], color='gray', linestyle='-', linewidth=0.8)

    for i, experiment_id in enumerate(experiment_ids):
        ax.plot(x_points, [c_ft[i], t_ft[i]], marker='o', color='black', linestyle='-', linewidth=1)
        
    # # Markers for control and test values
    # ax.scatter(np.zeros_like(c_ft), c_ft, color='blue', label='Control')
    # ax.scatter(np.ones_like(t_ft), t_ft, color='red', label='Test')
    # Labels and formatting
    ax.set_xticks(x_points, [r'$S \setminus S_{acc}$', r'$S_{acc}$'])
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=16)

    ax.set_ylabel(r'Time percentage of avoidance', fontsize=fontsize_label)
    ax.set_xlim(0, 1)
    ax.set_ylim(0.3, 1)

    fig.tight_layout()

    fig.savefig(f"{study.plot_output_folder}/PUBLISH/avoid_by_acc.pdf")

if __name__ == "__main__":
    data = study.data

    sample_time_points, _ = get_peaks_data(
        func_bound=None,
        func_multiplier=1,
        acc_max_height=0,
        vel_max_height=0,
        acc_prominence=15,
        do_group=False,
    )
    rv_df, ra_df = get_sampled_dataframe(sample_time_points)
    control = data.copy()
    control_rv = control[~control['frame'].isin(rv_df['frame'])]

    plot_tests(control_rv, rv_df)

    x = rv_df['rv_acceleration_fish']
    y = rv_df['rv_rev'] - rv_df['rev']
    # Create a condition for points where the difference is positive
    positive_difference = y > 0
    # Now count the number of points where the condition is True
    positive_count = positive_difference.sum()
    positive_percentage = positive_count / len(y) * 100
    # Print the result
    print(f"Percentage of points where RV_REV > REV: {positive_percentage:.2f}%")

