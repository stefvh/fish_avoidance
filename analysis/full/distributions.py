
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats

from core.experiment import Experiment
from core.study import study

from analysis.full.get_peak_data import get_peaks_data
from analysis.full.run_reg_fe import get_sampled_dataframe

import seaborn as sns

from scipy.stats import mannwhitneyu

def plot(control, rv_df, ra_df):
    # Creating the 3x3 subplot grid
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))

    # Row 1: rev
    ax = axes[0, 0]
    group_1 = control['rev']
    group_2 = rv_df['rv_rev']
    statistic, p_value = mannwhitneyu(group_1, group_2)

    # Output the result
    print("Mann-Whitney U statistic:", statistic)
    print("p-value:", p_value)
    ax.text(0.5, 0.80, f"Mann-Whitney U statistic: {statistic:.0f}", transform=ax.transAxes)
    if p_value < 0.05:
        add_text = "Significant Difference"
    else:
        add_text = "NO Significant Difference"
    ax.text(0.5, 0.75, f"p-value: {p_value:.3f} (" + add_text + ")", transform=ax.transAxes)

    group_1_mean = np.mean(group_1)
    group_2_mean = np.mean(group_2)

    std_group1 = np.std(group_1, ddof=1)  # Use ddof=1 for sample standard deviation
    std_group2 = np.std(group_2, ddof=1)
    # Calculate Cohen's d
    cohens_d = (group_1_mean - group_2_mean) / np.sqrt((std_group1 ** 2 + std_group2 ** 2) / 2)
    print("Cohen's d:", cohens_d)
    ax.text(0.5, 0.10, f"Cohen's d (G1-G2): {cohens_d:.3f}", transform=ax.transAxes)

    effect_size_units_group1 = cohens_d * std_group1
    print("Effect size in units of group 1:", effect_size_units_group1)
    ax.text(0.5, 0.05, f"Effect size in units of G1: {effect_size_units_group1:.3f}", transform=ax.transAxes)

    # print means
    print("Control mean:", group_1_mean)
    print("RA_DF mean:", group_2_mean)

    # vline at 0 on axes[0, 0]
    axes[0, 0].axvline(0, color='red', linestyle='dashed', linewidth=1)
    axes[0, 0].axvline(2, color='black', linestyle='dashed', linewidth=1)

    sns.kdeplot(control['rev'], ax=axes[0, 0])
    axes[0, 0].set_title('Control - Rev')

    sns.kdeplot(rv_df['rv_rev'], ax=axes[0, 0])
    axes[0, 1].set_title('RV_DF - Rev')

    sns.kdeplot(ra_df['ra_rev'], ax=axes[0, 0])
    axes[0, 2].set_title('RA_DF - Rev')

    # Row 2: distance
    sns.kdeplot(control['distance'], ax=axes[1, 0])
    axes[1, 0].set_title('Control - Distance')

    sns.kdeplot(rv_df['distance'], ax=axes[1, 1])
    axes[1, 1].set_title('RV_DF - Distance')

    sns.kdeplot(ra_df['distance'], ax=axes[1, 2])
    axes[1, 2].set_title('RA_DF - Distance')

    # Row 3: apv
    sns.kdeplot(control['apv'], ax=axes[2, 0])
    axes[2, 0].set_title('Control - APV')

    sns.kdeplot(rv_df['apv'], ax=axes[2, 0])
    axes[2, 1].set_title('RV_DF - APV')

    sns.kdeplot(ra_df['apv'], ax=axes[2, 0])
    axes[2, 2].set_title('RA_DF - APV')

    # Adjusting layout
    plt.tight_layout()

    # Display the plot
    plt.show()

    plt.savefig(f"{study.plot_output_folder}/distributions.png")


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

    print(rv_df.columns)

    # plot x-axis: acceleration_fish, y-axis: the difference rv_rev - rev
    x = ra_df['ra_acceleration_fish']
    y = rv_df['rv_rev'] - rv_df['rev']
    plt.figure(figsize=(6, 4))
    plt.scatter(x, y, alpha=0.5)
    plt.axhline(0, color='red', linestyle='dashed', linewidth=1)
    plt.axvline(0, color='green', linestyle='dashed', linewidth=1)
    plt.xlabel('Acceleration Fish')
    plt.ylabel('Difference RV_REV - REV')
    plt.title('Scatter plot of acceleration_fish and difference RV_REV - REV')
    plt.grid(True)
    plt.savefig(f"{study.plot_output_folder}/scatter_plot.png")

    # Create a condition for points where the difference is positive
    positive_difference = y > 0

    # Now count the number of points where the condition is True
    positive_count = positive_difference.sum()

    positive_percentage = positive_count / len(y) * 100

    # Print the result
    print(f"Percentage of points where RV_REV > REV: {positive_percentage:.2f}%")


    plot(control, rv_df, ra_df)
    control_rv = control[~control['frame'].isin(rv_df['frame'])]
    control_ra = control[~control['frame'].isin(ra_df['frame'])]

    delimiter = 2
    # percentage of rev > 2
    print("Control")
    print(control_rv[control_rv['rev'] > delimiter].shape[0] / control_rv.shape[0])
    print("RV_DF")
    print(rv_df[rv_df['rv_rev'] > delimiter].shape[0] / rv_df.shape[0])
    print("RA_DF")
    print(ra_df[ra_df['ra_rev'] > delimiter].shape[0] / ra_df.shape[0])

    experiment_ids = control['experiment_id'].unique()
    # c_ft is an array for each experiment_id it gives the percentage of rows where rev > 2 during this experiment
    c_ft = np.array([control_rv[control_rv['experiment_id'] == experiment_id]['rev'].gt(delimiter).mean() for experiment_id in experiment_ids])
    t_ft = np.array([rv_df[rv_df['experiment_id'] == experiment_id]['rv_rev'].gt(delimiter).mean() for experiment_id in experiment_ids])
    # Perform a paired t-test
    t_stat, p_val = stats.ttest_rel(c_ft, t_ft)

    # Output the results
    print(f"T-statistic: {t_stat}")
    print(f"P-value: {p_val}")

    print("Control")
    print(control_rv[control_rv['speed_fish'] < delimiter].shape[0] / control_rv.shape[0])
    print("RV_DF")
    print(rv_df[rv_df['rv_speed_fish'] < delimiter].shape[0] / rv_df.shape[0])
    print("RA_DF")
    print(ra_df[ra_df['ra_speed_fish'] < delimiter].shape[0] / ra_df.shape[0])

    # Assume experiment_ids, c_ft, t_ft are already defined
    plt.figure(figsize=(10, 6))

    # Paired Line Plot
    for i, experiment_id in enumerate(experiment_ids):
        plt.plot([0, 1], [c_ft[i], t_ft[i]], marker='o', color='gray', linestyle='-', linewidth=1)
        
    # Markers for control and test values
    plt.scatter(np.zeros_like(c_ft), c_ft, color='blue', label='Control')
    plt.scatter(np.ones_like(t_ft), t_ft, color='red', label='Test')

    # Labels and formatting
    plt.xticks([0, 1], ['Control', 'Test'])
    plt.ylabel('Percentage of rows with rev > 2')
    plt.title('Paired Comparison of Control and Test Groups')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{study.plot_output_folder}/paired_comparison.png")