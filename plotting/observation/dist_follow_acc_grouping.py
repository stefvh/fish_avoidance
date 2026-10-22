import os
import matplotlib.pyplot as plt
import numpy as np

from core.experiment import Experiment, get_experiment

from utils.parser import parse_args

from scipy.stats import shapiro, mannwhitneyu


def extend_group(results, group, extension=10):
    group_frame_numbers = group["frame_number"].unique()
    group_frame_numbers = np.concatenate([np.arange(frame_number, frame_number + extension) for frame_number in group_frame_numbers])
    group_frame_numbers = np.unique(group_frame_numbers)
    group = results[results["frame_number"].isin(group_frame_numbers)]
    return group


def foldering(experiment: Experiment):
    output_folder = experiment.input_folder + "/plots"
    sub_folder = "observation"
    output_filename = "dist_follow_acc_grouping.pdf"
    if not os.path.exists(f"{output_folder}/{sub_folder}"):
        os.makedirs(f"{output_folder}/{sub_folder}")
    return f"{output_folder}/{sub_folder}/{output_filename}"


def run(experiment: Experiment):
    fig, axes = plt.subplots(2, 2, figsize=(20, 10))

    results = experiment.results

    # Group 1: Faraway Non-Following Robot
    distance_threshold_far = 30
    group_1 = results[(results["robot_distance"] > distance_threshold_far) & 
                      (results["robot_fish_follow_metric"] < 0)]
    group_1 = extend_group(results, group_1)

    ax = axes[0, 0]
    ax.set_title("Fish Acceleration Distribution for Faraway Non-Following Robot")
    group_1_fish_velocities = group_1["fish_acceleration_batch"]

    # Remove NaN values
    group_1_fish_velocities = group_1_fish_velocities.dropna()
    group_1_fish_velocities = group_1_fish_velocities.values

    ax.hist(group_1_fish_velocities, bins=50, density=True, color="tab:blue")
    ax.set_xlabel("Fish Acceleration (cm/s)")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 32)
    ax.set_ylim(0, 0.2)

    statistic, p_value = shapiro(group_1_fish_velocities)
    print("Shapiro-Wilk test statistic:", statistic)
    print("p-value:", p_value)
    ax.text(0.5, 0.95, f"Shapiro-Wilk test statistic: {statistic:.3f}", transform=ax.transAxes)
    if p_value <  0.05:
        add_text = "NOT Normally Distributed"
    else:
        add_text = "Normally Distributed"
    ax.text(0.5, 0.90, f"p-value: {p_value:.3f} (" + add_text + ")", transform=ax.transAxes)

    # Group 2: Closeby Following Robot
    distance_threshold_close = 6
    group_2 = results[(results["robot_distance"] < distance_threshold_close) & 
                      (results["robot_fish_follow_metric"] > 0)]
    group_2 = extend_group(results, group_2)

    ax = axes[0, 1]
    ax.set_title("Fish Acceleration Distribution for Closeby Following Robot")
    group_2_fish_velocities = group_2["fish_acceleration_batch"]

    group_2_fish_velocities = group_2_fish_velocities.dropna()
    group_2_fish_velocities = group_2_fish_velocities.values

    ax.hist(group_2_fish_velocities, bins=50, density=True, color="tab:orange")
    ax.set_xlabel("Fish Acceleration (cm/s)")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 32)
    ax.set_ylim(0, 0.2)

    statistic, p_value = shapiro(group_2_fish_velocities)
    print("Shapiro-Wilk test statistic:", statistic)
    print("p-value:", p_value)
    ax.text(0.5, 0.95, f"Shapiro-Wilk test statistic: {statistic:.3f}", transform=ax.transAxes)
    if p_value <  0.05:
        add_text = "NOT Normally Distributed"
    else:
        add_text = "Normally Distributed"
    ax.text(0.5, 0.90, f"p-value: {p_value:.3f} (" + add_text + ")", transform=ax.transAxes)

    # Group 1 and 2 together
    ax = axes[1, 0]
    ax.set_title("Fish Acceleration Distribution for Both Groups")
    ax.hist(group_1_fish_velocities, bins=50, density=True, alpha=0.5, label="Faraway Non-Following Robot")
    ax.hist(group_2_fish_velocities, bins=50, density=True, alpha=0.5, label="Closeby Following Robot")
    ax.set_xlabel("Fish Acceleration (cm/s)")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 32)
    ax.set_ylim(0, 0.2)
    ax.legend()

    # Perform Mann-Whitney U test
    statistic, p_value = mannwhitneyu(group_1_fish_velocities, group_2_fish_velocities)

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

    x1 = group_1_fish_velocities.astype(float)
    x2 = group_2_fish_velocities.astype(float)
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
    group_1_mean = np.mean(group_1_fish_velocities)
    group_1_median = np.median(group_1_fish_velocities)
    group_2_mean = np.mean(group_2_fish_velocities)
    group_2_median = np.median(group_2_fish_velocities)
    axin.set_xticklabels([r"Faraway" "\n" rf"Mean: {group_1_mean:.3f}" "\n" rf"Median: {group_1_median:.3f}", 
                          r"Closeby" "\n" rf"Mean: {group_2_mean:.3f}" "\n" rf"Median: {group_2_median:.3f}"])
    axin.set_ylabel("Fish Acceleration (cm/s)")

    std_group1 = np.std(group_1_fish_velocities, ddof=1)  # Use ddof=1 for sample standard deviation
    std_group2 = np.std(group_2_fish_velocities, ddof=1)
    # Calculate Cohen's d
    cohens_d = (group_1_mean - group_2_mean) / np.sqrt((std_group1 ** 2 + std_group2 ** 2) / 2)
    print("Cohen's d:", cohens_d)
    ax.text(0.5, 0.10, f"Cohen's d (G1-G2): {cohens_d:.3f}", transform=ax.transAxes)

    effect_size_units_group1 = cohens_d * std_group1
    print("Effect size in units of group 1:", effect_size_units_group1)
    ax.text(0.5, 0.05, f"Effect size in units of G1: {effect_size_units_group1:.3f}", transform=ax.transAxes)

    # Residuals
    ax = axes[1, 1]
    ax.set_title("Residuals (Group2 - Group1)")
    print("Group1 Size:", len(group_1_fish_velocities))
    print("Group2 Size:", len(group_2_fish_velocities))
    n_samples = int(0.95 * len(group_1_fish_velocities))
    group_1_fish_velocities = np.random.choice(group_1_fish_velocities, n_samples)
    group_2_fish_velocities = np.random.choice(group_2_fish_velocities, n_samples)

    residuals = group_2_fish_velocities - group_1_fish_velocities
    ax.hist(residuals, bins=50, density=True, color="tab:red")
    ax.set_xlabel("Residuals")
    ax.set_ylabel("Density")
    ax.set_xlim(-20, 20)
    ax.set_ylim(0, 0.2)

    statistic, p_value = shapiro(residuals)
    print("Shapiro-Wilk test statistic:", statistic)
    print("p-value:", p_value)
    ax.text(0.5, 0.95, f"Shapiro-Wilk test statistic: {statistic:.3f}", transform=ax.transAxes)
    if p_value <  0.05:
        add_text = "NOT Normally Distributed"
    else:
        add_text = "Normally Distributed"
    ax.text(0.5, 0.90, f"p-value: {p_value:.3f} (" + add_text + ")", transform=ax.transAxes)

    #inset Q-Q plot of residuals with normal distribution
    axin = ax.inset_axes([0.10, 0.45, 0.35, 0.35])
    axin.set_title("Q-Q Plot of Residuals")
    axin.set_xlabel("Theoretical Quantiles")
    axin.set_ylabel("Sample Quantiles")
    axin.set_xlim(-20, 20)
    axin.set_ylim(-20, 20)
    axin.plot([-20, 20], [-20, 20], color="black", linestyle="--")
    axin.scatter(np.sort(np.random.normal(0, np.std(residuals), n_samples)), np.sort(residuals), color="tab:red")

    output_filename = foldering(experiment)
    plt.savefig(output_filename)


if __name__ == "__main__":
    args = parse_args()
    variables = vars(args)

    experiment = get_experiment(**variables)
    run(experiment)
