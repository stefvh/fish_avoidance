from matplotlib import pyplot as plt
import numpy as np
from core.experiment import Experiment
from core.study import study

from analysis.full.get_peak_data import get_peaks_data
from analysis.full.run_reg_fe import get_sampled_dataframe

import pandas as pd

from scipy.stats import mannwhitneyu


def compare_distr(d1: pd.Series, d2: pd.Series):
    fig, ax = plt.subplots()

    ax.set_title("Distance Distribution for Both Groups")
    ax.hist(d1, bins=50, density=True, alpha=0.5, label="Control")
    ax.hist(d2, bins=50, density=True, alpha=0.5, label="Reactive")
    ax.set_xlabel("Distance (cm)")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 32)
    ax.set_ylim(0, 0.2)
    ax.legend()

    # Perform Mann-Whitney U test
    statistic, p_value = mannwhitneyu(d1, d2)

    # Output the result
    print("Mann-Whitney U statistic:", statistic)
    print("p-value:", p_value)
    ax.text(0.5, 0.80, f"Mann-Whitney U statistic: {statistic:.0f}", transform=ax.transAxes)
    if p_value < 0.05:
        add_text = "Significant Difference"
    else:
        add_text = "NO Significant Difference"
    ax.text(0.5, 0.75, f"p-value: {p_value:.3f} (" + add_text + ")", transform=ax.transAxes)

    axin = ax.inset_axes([0.6, 0.325, 0.35, 0.35])
    # axin.boxplot([group_1_fish_velocities, group_2_fish_velocities], showmeans=True)

    x1 = d1.astype(float)
    x2 = d2.astype(float)
    violin = axin.violinplot([x1, x2], 
                             showmedians=True,
                             showmeans=True,
                             positions=[1, 2],
                             quantiles=[[0.25, 0.75],
                                        [0.25, 0.75]],)
    for partname in ("cquantiles",):
        vp = violin[partname]
        # vp.set_edgecolor("red")
        vp.set_linewidth(0.75)
        vp.set_linestyle("--")
    for partname in ("cmeans",):
        vp = violin[partname]
        vp.set_edgecolor("green")
        vp.set_linewidth(1)
    # for partname in ('cmins', 'cmaxes'):
    #     vp = violin[partname]
    #     vp.set_linestyle("--")

    axin.set_xticks([1, 2])
    group_1_mean = np.mean(d1)
    group_1_median = np.median(d1)
    group_2_mean = np.mean(d2)
    group_2_median = np.median(d2)
    axin.set_xticklabels([r"Control" "\n" rf"Mean: {group_1_mean:.3f}" "\n" rf"Median: {group_1_median:.3f}", 
                          r"Reactive" "\n" rf"Mean: {group_2_mean:.3f}" "\n" rf"Median: {group_2_median:.3f}"])
    axin.set_ylabel("Distance (cm)")

    std_group1 = np.std(d1, ddof=1)  # Use ddof=1 for sample standard deviation
    std_group2 = np.std(d2, ddof=1)
    # Calculate Cohen's d
    cohens_d = (group_1_mean - group_2_mean) / np.sqrt((std_group1 ** 2 + std_group2 ** 2) / 2)
    print("Cohen's d:", cohens_d)
    ax.text(0.5, 0.10, f"Cohen's d (G1-G2): {cohens_d:.3f}", transform=ax.transAxes)

    effect_size_units_group1 = cohens_d * std_group1
    print("Effect size in units of group 1:", effect_size_units_group1)
    ax.text(0.5, 0.05, f"Effect size in units of G1: {effect_size_units_group1:.3f}", transform=ax.transAxes)

    plt.savefig(f"{study.plot_output_folder}/distance_distribution.pdf")


if __name__ == "__main__":
    data = study.data
    sample_time_points, _ = get_peaks_data(
        func_bound=None,
        func_multiplier=1,
        acc_max_height=0,
        vel_max_height=0,
        acc_prominence=12.5,
        do_group=False,
    )
    rv_df, ra_df = get_sampled_dataframe(sample_time_points)

    dist_reaction = ra_df['distance']

    control = data.copy()
    ra_frames = ra_df['frame']
    control = control[~control['frame'].isin(ra_frames)]
    dist_control = control['distance']

    compare_distr(dist_control, dist_reaction)
