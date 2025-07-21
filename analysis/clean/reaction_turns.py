
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from analysis.one.wrapper_fixed import run_plm_model
from core.experiment import Experiment
from core.study import study

from analysis.clean.reactions import extract_trials, get_baseline_data_subjects, save_plot

from scipy.stats import wilcoxon
from scipy import stats
import statsmodels
import statsmodels.genmod
import statsmodels.genmod.families
import statsmodels.stats
import statsmodels.stats.anova

import statsmodels.formula.api as smf


def measure_significant_turns_metrics(exp_trials, df_baseline, distance_bins, post_approach_frames=10):
    # Define the distance bins with an additional post-approach bin
    extended_bins = distance_bins + [-5]
    
    # Initialize lists to store significant turns, max speed, and max acceleration data
    cos_avoid_data = []
    max_speed_data = []
    max_accel_data = []

    for exp_idx, trials in enumerate(exp_trials):
        exp_cos_avoid = [[] for _ in range(len(extended_bins) - 1)]
        exp_max_speed = [[] for _ in range(len(extended_bins) - 1)]
        exp_max_accel = [[] for _ in range(len(extended_bins) - 1)]

        # Calculate turn threshold based on the 95th percentile of absolute turns in the baseline data
        df_exp_bas = df_baseline[df_baseline["experiment_id"] == exp_idx]
        df_exp_bas['abs_turn_fish'] = df_exp_bas['turn_fish'].abs()
        turn_threshold = df_exp_bas['abs_turn_fish'].quantile(0.95)

        for trial in trials:
            # Separate the data before the end of approach index
            end_idx = trial["end_approach_index"]
            trial_data = trial["data"].iloc[:end_idx]

            # Process each bin within the defined distance range
            for bin_idx in range(len(distance_bins) - 1):
                lower_bound = distance_bins[bin_idx + 1]
                upper_bound = distance_bins[bin_idx]

                bin_data = trial_data[(trial_data["distance"] >= lower_bound) & (trial_data["distance"] < upper_bound)]

                # Identify significant turns and metrics within the bin
                for _, row in bin_data.iterrows():
                    if abs(row['turn_fish']) > turn_threshold:
                        current_frame = row['frame']

                        # Extract frames around the turn
                        pre_turn_data = trial_data[(trial_data['frame'] >= current_frame - 8) & (trial_data['frame'] < current_frame)]
                        post_turn_data = trial_data[(trial_data['frame'] > current_frame) & (trial_data['frame'] <= current_frame + 8)]

                        if len(pre_turn_data) >= 4 and len(post_turn_data) >= 4:
                            before_cos_avoid = pre_turn_data['cos_avoid'].iloc[-1]
                            after_cos_avoid = post_turn_data['cos_avoid'].iloc[0]

                            max_speed_before = pre_turn_data['speed_fish'].max()
                            max_speed_after = post_turn_data['speed_fish'].max()

                            max_accel_before = pre_turn_data['acceleration_fish'].max()
                            max_accel_after = post_turn_data['acceleration_fish'].max()

                            exp_cos_avoid[bin_idx].append((before_cos_avoid, after_cos_avoid))
                            exp_max_speed[bin_idx].append((max_speed_before, max_speed_after))
                            exp_max_accel[bin_idx].append((max_accel_before, max_accel_after))

            # Process the post-approach bin
            post_approach_data = trial["data"].iloc[end_idx:end_idx + post_approach_frames]

            for _, row in post_approach_data.iterrows():
                if abs(row['turn_fish']) > turn_threshold:
                    current_frame = row['frame']

                    # Extract frames around the turn in the post-approach phase
                    pre_turn_data = trial_data[trial_data['frame'] == current_frame - 4]
                    post_turn_data = trial_data[trial_data['frame'] == current_frame + 4]
                    
                    if not pre_turn_data.empty and not post_turn_data.empty:
                        before_cos_avoid = pre_turn_data['cos_avoid'].values[0]
                        after_cos_avoid = post_turn_data['cos_avoid'].values[0]
                        exp_cos_avoid[-1].append((before_cos_avoid, after_cos_avoid))

                    # Extract frames around the turn in the post-approach phase
                    pre_turn_data = post_approach_data[(post_approach_data['frame'] >= current_frame - 8) & (post_approach_data['frame'] < current_frame)]
                    post_turn_data = post_approach_data[(post_approach_data['frame'] > current_frame) & (post_approach_data['frame'] <= current_frame + 8)]

                    if len(pre_turn_data) >= 4 and len(post_turn_data) >= 4:
                        max_speed_before = pre_turn_data['speed_fish'].max()
                        max_speed_after = post_turn_data['speed_fish'].max()
                        exp_max_speed[-1].append((max_speed_before, max_speed_after))
                        max_accel_before = pre_turn_data['acceleration_fish'].max()
                        max_accel_after = post_turn_data['acceleration_fish'].max()
                        exp_max_accel[-1].append((max_accel_before, max_accel_after))

        # Collect data for the current experiment
        cos_avoid_data.append(exp_cos_avoid)
        max_speed_data.append(exp_max_speed)
        max_accel_data.append(exp_max_accel)

    return extended_bins, cos_avoid_data, max_speed_data, max_accel_data


def plot_test(measurements, distance_bins):
    n_bins = len(distance_bins) - 1
    bin_labels = [f"{distance_bins[i]}-{distance_bins[i+1]}" for i in range(len(distance_bins)-1)]
    if distance_bins[-1] == -5:
        bin_labels[-1] = "Post"  # Label for the new bin

    for measurement, data in measurements.items():
        n_experiments = len(data)

        fig, axes = plt.subplots(nrows=n_bins, ncols=n_experiments, figsize=(n_experiments * 5, n_bins * 4), sharex=True, sharey=True)
        
        if n_experiments == 1:
            axes = np.expand_dims(axes, axis=1)

        for exp_idx, exp_data in enumerate(data):
            for bin_idx, bin_data in enumerate(exp_data):
                ax = axes[bin_idx, exp_idx]

                if bin_data:
                    before, after = zip(*bin_data)
                    # Plot bars at positions 0 and 1
                    means = [np.nanmean(before), np.nanmean(after)]
                    ax.bar([0, 1], means, color=['blue', 'orange'], alpha=0.6)
                    
                    # Overlay scatter points on top of bars
                    ax.scatter(np.zeros(len(before)), before, color='blue', edgecolor='k', alpha=0.6)
                    ax.scatter(np.ones(len(after)), after, color='orange', edgecolor='k', alpha=0.6)

                    # Perform Wilcoxon signed-rank test
                    try:
                        w_stat, p_value = wilcoxon(before, after)
                        sample_size = len(before)
                    except ValueError:  # Handle cases where sample size is too small or data is identical
                        w_stat, p_value, sample_size = np.nan, np.nan, len(before)
                    
                    # Annotate with Wilcoxon test results and sample size
                    ax.text(0.5, 0.95, f'p={p_value:.3e}, n={sample_size}',
                            transform=ax.transAxes, ha='center', va='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.6))
                
                
                ax.set_title(f"Exp {exp_idx+1}, {bin_labels[bin_idx]}")
                ax.set_xticks([0, 1])
                ax.set_xticklabels(['Before', 'After'])
                ax.set_ylabel(measurement)
                if measurement == 'cos_avoid':
                    ax.axhline(0, color='black', linewidth=0.5)

        plt.tight_layout()
        save_plot(fig, f"reaction_turns_{measurement}.pdf")


def measure_distance_based(exp_trials, distance_bins):
    # Initialize data structures to store lists of lists of lists for each experiment
    n_experiments = len(exp_trials)
    cos_avoid_data = [[] for _ in range(n_experiments)]

    # Iterate over each experiment and each trial within the experiment
    for exp_idx, trials in enumerate(exp_trials):
        # Initialize lists of lists for each distance bin within this experiment
        exp_avoid_bins = [[] for _ in range(len(distance_bins))]
        
        for trial in trials:
            # Get data up to end of approach index
            end_idx = trial["end_approach_index"]
            trial_data = trial["data"]
            
            # Bin the data based on the distance bins
            for bin_idx in range(len(distance_bins) - 1):
                lower_bound = distance_bins[bin_idx + 1]
                upper_bound = distance_bins[bin_idx]
                
                # Filter rows within the current distance bin
                bin_data = trial_data[(trial_data["distance"] >= lower_bound) & (trial_data["distance"] < upper_bound)]               
                
                # Append cos_avoid values to respective bin lists
                exp_avoid_bins[bin_idx].append(bin_data["cos_avoid"].tolist())
            
            # Additional bin for 10 frames after the end of approach
            post_approach_data = trial_data.iloc[end_idx:end_idx + 10]
            if not post_approach_data.empty:
                exp_avoid_bins[-1].append(post_approach_data["cos_avoid"].tolist())
        
        # After processing all trials in the experiment, add lists to main data structures
        cos_avoid_data[exp_idx] = exp_avoid_bins
    
    return cos_avoid_data


def plot_data_over_bins(data, bins):
    # Define bin labels for x-axis (e.g., 40-35, 35-30, ..., 5-0, Post)
    bin_labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins)-1)]
    bin_labels.append("Post")  # Label for the last "Post" bin

    n_experiments = len(data)
    n_bins = len(bins)

    fig, axes = plt.subplots(nrows=n_experiments, ncols=1, figsize=(10, n_experiments * 5), sharex=True, sharey=True)
    
    if n_experiments == 1:
        axes = [axes]

    for subj_idx, subj_data in enumerate(data):
        ax = axes[subj_idx]
        
        means = []  # Store means to connect with lines
        bin_indices = []  # Track which bins have data
        for bin_idx, bin_data in enumerate(subj_data):
            # Flatten bin data across all trials for this bin
            flattened_bin_data = [value for trial in bin_data for value in trial]
            if flattened_bin_data:
                means.append(np.nanmean(flattened_bin_data))
                bin_indices.append(bin_idx)

                flattened_bin_data = np.array(flattened_bin_data)
                flattened_bin_data = flattened_bin_data[~np.isnan(flattened_bin_data)]

                # Plot boxplot and scatter points
                ax.boxplot(flattened_bin_data, positions=[bin_idx], widths=0.6)
                ax.scatter([bin_idx] * len(flattened_bin_data), flattened_bin_data, color='blue', edgecolor='k', alpha=0.6)
        
        # Connect means with lines
        ax.plot(bin_indices, means, color='red', linestyle='--', marker='o', label='Mean Cos Avoid')

        # Set labels and title
        ax.set_title(f'Experiment {subj_idx+1}')
        ax.set_xticks(range(n_bins))
        ax.set_xticklabels(bin_labels)
        ax.set_ylabel('Cos Avoid')

    axes[-1].set_xlabel('Bins')
    plt.tight_layout()
    save_plot(fig, "reaction_turns_cos_avoid.pdf")


def create_turn_dataframe(exp_trials, distance_bins, turn_threshold):
    # Initialize an empty list to collect rows for the DataFrame
    data_rows = []

    # Iterate over each experiment and each trial within the experiment
    for exp_idx, trials in enumerate(exp_trials):
        for trial in trials:
            # Get data up to the end of approach index
            end_idx = trial["end_approach_index"]
            trial_data = trial["data"].iloc[:end_idx]
            
            # Bin the data and process each distance bin separately
            for bin_idx in range(len(distance_bins)):
                if bin_idx == len(distance_bins) - 1:
                    lower_bound = 0
                else:
                    lower_bound = distance_bins[bin_idx + 1]
                upper_bound = distance_bins[bin_idx]
                
                # Filter rows within the current distance bin
                bin_data = trial_data[(trial_data["distance"] >= lower_bound) & (trial_data["distance"] < upper_bound)]
                
                if not bin_data.empty:
                    # Identify the frame with maximum turn within this bin
                    max_turn_idx = bin_data['turn_fish'].abs().idxmax()
                    max_turn_value = bin_data.loc[max_turn_idx, 'turn_fish']
                    
                    # Determine if the max turn is significant based on the threshold
                    turn_binary = 1 if abs(max_turn_value) > turn_threshold else 0
                    
                    # Retrieve corresponding angle, fish_speed, robot_speed, and distance
                    angle_fr = bin_data.loc[max_turn_idx, 'angle_rf']
                    fish_speed = bin_data.loc[max_turn_idx, 'speed_fish']
                    robot_speed = bin_data.loc[max_turn_idx, 'speed_robot']
                    distance = bin_data.loc[max_turn_idx, 'distance']
                    
                    # Recompute angle for 0 to π
                    re_angle_rf = 2 * np.pi - angle_fr if angle_fr > np.pi else angle_fr
                    # Calculate speed ratio
                    speed_ratio = fish_speed / robot_speed if robot_speed != 0 else np.nan
                    distance_bin = f"{upper_bound}-{lower_bound}"

                    # Get past and future fish speed from max turn frame (-4 and +4)
                    # ensure we don't go out of bounds --> take max from -4 to 0 and 0 to 4
                    past_speed = bin_data.loc[max_turn_idx - 4:max_turn_idx, 'speed_fish'].max()
                    future_speed = bin_data.loc[max_turn_idx + 1:max_turn_idx + 4, 'speed_fish'].max()

                    #
                    past_cos_avoid = bin_data.loc[max_turn_idx - 4:max_turn_idx, 'cos_avoid'].mean()
                    future_cos_avoid = bin_data.loc[max_turn_idx + 1:max_turn_idx + 4, 'cos_avoid'].mean()

                    # Append the data as a new row
                    data_rows.append({
                        'binary_turn': turn_binary,
                        'continuous_turn': max_turn_value,
                        'angle_rf': re_angle_rf,
                        'o_angle_rf': angle_fr,
                        'speed_ratio': speed_ratio,
                        'distance_bin': distance_bin,
                        'distance': distance,
                        'experiment_id': exp_idx,
                        'past_speed': past_speed,
                        'future_speed': future_speed,
                        'past_cos_avoid': past_cos_avoid,
                        'future_cos_avoid': future_cos_avoid
                    })

    # Create DataFrame
    turn_df = pd.DataFrame(data_rows)
    return turn_df


def statistics(data):
    # Fit a mixed effects logistic regression model (with random intercept for subjects)
    model = smf.glm(
        "binary_turn ~ angle_rf + speed_ratio + distance + (1 | experiment_id)",
        data, groups=data['experiment_id'],
        family=statsmodels.genmod.families.Binomial()
    ).fit()

    # Print the summary of the model
    print(model.summary())

    # Fit the null model (intercept-only model)
    null_model = smf.glm(formula="binary_turn ~ 1", data=data, family=statsmodels.genmod.families.Binomial()).fit()

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


def plot_turns_by_distance_and_experiment(turn_df, distance_bins, experiments):
    # Get unique distance bins and experiment IDs for plotting grid
    unique_bins = sorted(turn_df['distance_bin'].unique())
    unique_experiments = sorted(turn_df['experiment_id'].unique())
    
    # Create a figure and axes array with appropriate dimensions
    fig, axes = plt.subplots(len(unique_bins), len(unique_experiments), figsize=(5 * len(unique_experiments), 4 * len(unique_bins)),
                             sharex=True, sharey=False)
    fig.suptitle('Continuous and Binary Turn vs Speed Ratio across Distance Bins and Experiments', fontsize=16)
    
    # Iterate through each subplot position
    for row_idx, distance_bin in enumerate(unique_bins):
        for col_idx, experiment_id in enumerate(unique_experiments):
            ax = axes[row_idx, col_idx]
            
            # Filter data for current distance bin and experiment
            plot_data = turn_df[(turn_df['distance_bin'] == distance_bin) & (turn_df['experiment_id'] == experiment_id)]

            # Normalize continuous turn values from [0, pi] to [0, 1]
            plot_data['continuous_turn'] = plot_data['continuous_turn'].abs() / np.pi
            
            # Plot continuous turn vs speed ratio
            ax.scatter(plot_data['speed_ratio'], plot_data['continuous_turn'], label='Continuous Turn', alpha=0.7, color="red")
            
            # Plot binary turn vs speed ratio as a second y-axis
            ax.scatter(plot_data['speed_ratio'], plot_data['binary_turn'], label='Binary Turn', alpha=0.5, marker='x', color="black")
            
            # Set labels and titles
            if col_idx == 0:
                ax.set_ylabel(f"Distance Bin: {distance_bin}")
            if row_idx == len(unique_bins) - 1:
                ax.set_xlabel("Speed Ratio")
            if row_idx == 0:
                ax.set_title(f"Experiment {experiment_id + 1}")

            ax.set_xlim(-0.1, 1.1)  # Limit speed ratio to 0-2
            ax.set_ylim(-0.1, 1.1)
            
            # Add a legend to each subplot
            ax.legend()
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit the title
    save_plot(fig, "turns_by_distance_and_experiment.pdf")


def plot_post_angle_speed(exp_trials):
    # Initialize lists to hold combined angle_fr and fish_speed values
    combined_angles = []
    combined_speeds = []
    
    # Calculate the number of experiments and set up the figure
    n_experiments = len(exp_trials)
    fig, axs = plt.subplots(n_experiments + 1, 2, figsize=(12, 6 * (n_experiments + 1)))

    # Iterate over each experiment
    for exp_idx, trials in enumerate(exp_trials):
        exp_angles = []
        exp_speeds = []
        
        for trial in trials:
            # Get data up to end of approach index
            end_idx = trial["end_approach_index"]
            trial_data = trial["data"]
            
            # Extract post-approach data (10 frames after end approach index)
            post_approach_data = trial_data.iloc[end_idx+6:end_idx + 10]
            
            # If post-approach data exists, get angle_rf and fish_speed
            if not post_approach_data.empty:
                max_speed = post_approach_data["speed_fish"].max()
                max_speed_idx = post_approach_data["speed_fish"].idxmax()
                max_angle = post_approach_data.loc[max_speed_idx, "angle_rf"]
                
                # Append to experiment-specific lists
                exp_angles.append(max_angle)
                exp_speeds.append(max_speed)
                
                # Append to combined lists
                combined_angles.append(max_angle)
                combined_speeds.append(max_speed)
        
        # Plot individual experiment data
        axs[exp_idx, 0].scatter(exp_speeds, exp_angles, color='blue', alpha=0.7)
        axs[exp_idx, 0].set_xlabel('Fish Speed')
        axs[exp_idx, 0].set_ylabel('Angle RF')
        axs[exp_idx, 0].set_title(f'Experiment {exp_idx + 1}: 2D Scatter Plot')

        ax_polar_exp = fig.add_subplot(n_experiments + 1, 2, 2 * exp_idx + 2, projection='polar')
        ax_polar_exp.scatter(exp_angles, exp_speeds, color='green', alpha=0.7)
        ax_polar_exp.set_title(f'Experiment {exp_idx + 1}: Circular Plot')
        ax_polar_exp.set_xlabel('Angle RF (radians)')
        ax_polar_exp.set_ylabel('Fish Speed')

    # Plot combined data for all experiments in the last row
    axs[-1, 0].scatter(combined_speeds, combined_angles, color='purple', alpha=0.7)
    axs[-1, 0].set_xlabel('Fish Speed')
    axs[-1, 0].set_ylabel('Angle RF')
    axs[-1, 0].set_title('Combined: 2D Scatter Plot')

    ax_polar_combined = fig.add_subplot(n_experiments + 1, 2, 2 * n_experiments + 2, projection='polar')
    ax_polar_combined.scatter(combined_angles, combined_speeds, color='orange', alpha=0.7)
    ax_polar_combined.set_title('Combined: Circular Plot')
    ax_polar_combined.set_xlabel('Angle RF (radians)')
    ax_polar_combined.set_ylabel('Fish Speed')

    # Define bins for speed (radial) and angle (angular)
    speed_bins = 5
    angle_bins = 8
    speed_edges = np.linspace(0, max(combined_speeds), speed_bins + 1)  # 6 edges for 5 bins
    angle_edges = np.linspace(0, 2 * np.pi, angle_bins + 1)  # 13 edges for 12 bins

    # Create a 2D histogram with specified bins
    hist, _, _ = np.histogram2d(combined_angles, combined_speeds, bins=[angle_edges, speed_edges])

    # Use the edges directly for r and theta so that they match the required shape for pcolormesh
    theta, r = np.meshgrid(angle_edges, speed_edges)  # r has shape (6,), theta has shape (13,)

    # # Plot the histogram data using pcolormesh for the density overlay
    pcm = ax_polar_combined.pcolormesh(theta, r, hist.T, shading='auto', cmap='plasma', alpha=0.7)
    # # pcm.set_array(np.ma.masked_where(hist.T == 0, hist.T))  # Mask empty bins


    # Color bar for density
    fig.colorbar(pcm, ax=ax_polar_combined, label="Density")



    plt.tight_layout()
    save_plot(fig, "post_angle_speed.pdf")


def plot_polar_distributions(turn_df):
    import utils.plotter

    # Separate data by binary_turn
    no_turn_data = turn_df[turn_df['binary_turn'] == 0]['angle_rf']
    yes_turn_data = turn_df[turn_df['binary_turn'] == 1]['angle_rf']
    
    # Create polar plot
    fig = plt.figure(figsize=(6, 4))
    ax = plt.subplot(111, polar=True)

    # Plot histograms for each binary_turn category
    bins = np.linspace(0, 2 * np.pi, 25)  # Define bins for the angles

    ax.hist(no_turn_data, bins=bins, alpha=0.5, label="No Turn", density=True, color="tab:blue")
    ax.hist(yes_turn_data, bins=bins, alpha=0.5, label="Turn", density=True, color="purple")

    # Restrict to half the circle
    ax.set_thetamin(0)  # 0 degrees
    ax.set_thetamax(180)  # 180 degrees

    radian_ticks = np.linspace(0, np.pi, 5)  # Define tick positions in radians
    tick_labels = [
        (
            r"$0$"
            if tick == 0  # Hide 2π duplicate label at 0
            else (
                r"$\frac{1}{4}\pi$"
                if tick == 0.25 * np.pi
                else (
                    r"$\frac{1}{2}\pi$"
                    if tick == 0.5 * np.pi
                    else (
                        r"$\frac{3}{4}\pi$"
                        if tick == 0.75 * np.pi
                        else (
                            r"$\pi$"
                            if tick == np.pi
                            else r"" if tick == 2 * np.pi else ""
                        )       
                    )
                )
            )
        )
        for tick in radian_ticks
    ]
    fontsize_ticks = 10
    fontsize_label = 14
    ax.tick_params(axis="both", which="major", labelsize=fontsize_ticks)
    ax.set_xticks(radian_ticks)
    ax.set_xticklabels(tick_labels, fontsize=12)  # Apply custom tick labels
    ax.tick_params(axis="x", which="major", pad=5)

    ax.set_xlabel(r"relative angle $\theta_{RF}$ (rad)", fontsize=fontsize_label, labelpad=-25)
    ax.set_ylabel(r"density", fontsize=fontsize_label, labelpad=25)

    ax.legend(
        bbox_to_anchor=(0.9, 0.8), #(0.01, -0.075),
        loc="center",
        fancybox=True,
        edgecolor="white",
        fontsize=12,
    )

    plt.tight_layout()
    save_plot(fig, "bin_turn_angle_rf.pdf")


def plot_past_future_speeds_distr(turn_df):
    # Create a figure and axes array with appropriate dimensions
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    data = turn_df[
        (turn_df['binary_turn'] == 1) &
        (turn_df['distance'] < 10) 
    ]

    # data = turn_df[turn_df['binary_turn'] == 1]

    # Plot histograms for past and future speeds
    ax.hist(data['past_speed'], bins=20, alpha=0.5, label="Past Speed", density=True, color="tab:blue")
    ax.hist(data['future_speed'], bins=20, alpha=0.5, label="Future Speed", density=True, color="purple")

    ax.set_xlabel("Fish Speed")
    ax.set_ylabel("Density")
    ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit the title
    save_plot(fig, "past_future_speeds_distr.pdf")

    # subjects, conditions, speeds = [], [], []
    # for value in [0, 1]:
    #     sub_data = turn_df[
    #         (turn_df['binary_turn'] == value)
    #     ]

    #     for row in sub_data.itertuples():
    #         subjects.append(row.experiment_id)
    #         conditions.append(value)
    #         speeds.append(row.past_speed)
    
    # df = pd.DataFrame({"subject": subjects, "condition": conditions, "speed": speeds})
    # result = run_plm_model(
    #     df, 
    #     "speed", 
    #     ["condition"], 
    #     ["subject"], 
    #     "within", 
    #     "individual",
    # )

    # print(result)

    # data = turn_df[
    #     (turn_df['binary_turn'] == 1)
    # ]

    subjects, conditions, speeds = [], [], []
    for row in data.itertuples():
        subjects.append(row.experiment_id)
        conditions.append("pre")
        speeds.append(row.past_speed)

        subjects.append(row.experiment_id)
        conditions.append("post")
        speeds.append(row.future_speed)
    
    df = pd.DataFrame({"subject": subjects, "condition": conditions, "speed": speeds})
    result = run_plm_model(
        df, 
        "speed", 
        ["condition"], 
        ["subject"], 
        "within", 
        "individual",
    )

    print(result)

    # data = turn_df[
    #     (turn_df['binary_turn'] == 1)
    # ]

    # subjects, conditions, cos_avoids = [], [], []
    # for row in data.itertuples():
    #     subjects.append(row.experiment_id)
    #     conditions.append("pre")
    #     cos_avoids.append(row.past_cos_avoid)

    #     subjects.append(row.experiment_id)
    #     conditions.append("post")
    #     cos_avoids.append(row.future_cos_avoid)
    
    # df = pd.DataFrame({"subject": subjects, "condition": conditions, "cos_avoid": cos_avoids})
    # result = run_plm_model(
    #     df, 
    #     "cos_avoid", 
    #     ["condition"], 
    #     ["subject"], 
    #     "within", 
    #     "individual",
    # )

    print(result)


def compare_pre_turn_and_no_turn_speeds(turn_df):
    # Filter data for instances within the distance bin of interest (e.g., 10-5 cm)
    # distance_bin_data = turn_df[(turn_df['distance'] < 10) & (turn_df['distance'] >= 5)]
    distance_bin_data = turn_df[(turn_df['distance'] < 40) & (turn_df['distance'] >= 0)]
    
    # Separate data for pre-turn speeds (binary_turn == 1) and non-turning speeds (binary_turn == 0)
    pre_turn_data = distance_bin_data[distance_bin_data['binary_turn'] == 1]['past_speed']
    no_turn_data = distance_bin_data[distance_bin_data['binary_turn'] == 0]['past_speed']
    
    # Plot histograms for visual comparison
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.hist(pre_turn_data, bins=20, alpha=0.5, label="Pre-Turn Speed", density=True, color="tab:blue")
    ax.hist(no_turn_data, bins=20, alpha=0.5, label="No-Turn Speed", density=True, color="orange")
    ax.set_xlabel("Fish Speed")
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_plot(fig, "pre_turn_vs_no_turn_speeds.pdf")

    # Check the size and content of each dataset
    print(f"Pre-turn data size: {len(pre_turn_data)}")
    print(pre_turn_data.describe())  # Summary statistics for pre-turn data
    print(f"No-turn data size: {len(no_turn_data)}")
    print(no_turn_data.describe())  # Summary statistics for no-turn data

    # Check for NaN values
    print("NaN values in pre-turn data:", pre_turn_data.isna().sum())
    print("NaN values in no-turn data:", no_turn_data.isna().sum())

    pre_turn_data = pre_turn_data.dropna()
    no_turn_data = no_turn_data.dropna()

    # Statistical comparison: Test if pre-turn speeds are significantly different from no-turn speeds
    from scipy.stats import ttest_ind
    t_stat, p_value = ttest_ind(pre_turn_data, no_turn_data, equal_var=False)  # Welch's t-test
    
    print(f"T-statistic: {t_stat:.2f}, P-value: {p_value:.4f}")
    if p_value < 0.05:
        print("Significant difference between pre-turn speeds and no-turn speeds.")
    else:
        print("No significant difference between pre-turn speeds and no-turn speeds.")
    
    # Prepare data for the FE model
    turn_df_cleaned = turn_df.dropna(subset=['past_speed'])  # Drop NaN values
    # turn_df_cleaned = turn_df_cleaned[(turn_df_cleaned['distance'] < 10) & (turn_df_cleaned['distance'] >= 5)]

    # Add a categorical variable for individuals if not already present
    # Assuming `experiment_id` is the unique identifier for individuals
    model_data = turn_df_cleaned[['past_speed', 'binary_turn', 'experiment_id', 'distance_bin']]

    result = run_plm_model(
        model_data, 
        "past_speed", 
        ["binary_turn"], 
        ["experiment_id", "distance_bin"], 
        "within", 
        "twoways",
    )

    print(result)


def pre_encounter_analysis():
    exp_trials = extract_trials()

    df_baseline = get_baseline_data_subjects()
    distance_bins = list(range(40, -5, -5))  # Generates [40, 35, 30, ..., 5, 0]

    # extended_bins, cos_avoid_data, max_speed_data, max_accel_data = measure_significant_turns_metrics(exp_trials, df_baseline, distance_bins)
    # measurements = {
    #     'cos_avoid': cos_avoid_data,
    #     'max_speed': max_speed_data,
    #     'max_accel': max_accel_data
    # }
    # plot_test(measurements, extended_bins)

    # cos_avoid_data = measure_distance_based(exp_trials, distance_bins)
    # plot_data_over_bins(cos_avoid_data, distance_bins)

    # # Example usage with your data
    # # Assuming exp_trials and distance_bins are defined, and turn_threshold is calculated (95th percentile of baseline data)
    # Replace with your baseline threshold calculation
    max_turn_mean = df_baseline["turn_fish"].abs().mean()
    max_turn_std = df_baseline["turn_fish"].abs().std()
    turn_threshold = max_turn_mean + 2 * max_turn_std
    distance_bins = list(range(40, 0, -5))
    turn_df = create_turn_dataframe(exp_trials, distance_bins, turn_threshold)

    # Calculate statistics
    # statistics(turn_df)

    # plot_past_future_speeds_distr(turn_df)

    compare_pre_turn_and_no_turn_speeds(turn_df)

    # plot_polar_distributions(turn_df)

    # # Plot continuous and binary turn vs speed ratio across distance bins and experiments
    # plot_turns_by_distance_and_experiment(turn_df, distance_bins, exp_trials)

    # Plot post-approach angle_fr vs fish_speed for each experiment
    # plot_post_angle_speed(exp_trials)


if __name__ == "__main__":
    pre_encounter_analysis()