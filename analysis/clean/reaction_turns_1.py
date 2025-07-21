
from matplotlib import pyplot as plt
import numpy as np

from core.experiment import Experiment
from core.study import study

from analysis.clean.reactions import extract_trials, get_baseline_data_subjects, save_plot

from scipy.stats import wilcoxon


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


def measure_distance_based(exp_trials, distance_bins, extend=True):
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



def plot_data_and_differences_over_bins(data, bins):
    # Define bin labels for x-axis (e.g., 40-35, 35-30, ..., 5-0, Post)
    bin_labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins)-1)]
    bin_labels.append("Post")  # Label for the last "Post" bin

    n_experiments = len(data)
    n_bins = len(bin_labels)

    # Create a figure with two rows per subject: one for raw cos_avoid, one for differences
    fig, axes = plt.subplots(nrows=n_experiments * 2, ncols=1, figsize=(10, n_experiments * 10), sharex=True)
    
    if n_experiments == 1:
        axes = [axes]

    for subj_idx, subj_data in enumerate(data):
        ax_cos_avoid = axes[subj_idx * 2]
        ax_diff = axes[subj_idx * 2 + 1]
        
        # Variables to store the mean cos_avoid and mean differences
        means = []  # Mean cos_avoid per bin across trials
        bin_indices = []  # Indices of bins with data

        # Calculate mean cos_avoid for each bin and collect differences
        all_differences = [[] for _ in range(n_bins - 1)]  # Difference storage across bins (n_bins - 1 for differences)

        for bin_idx, bin_data in enumerate(subj_data):
            # Flatten bin data across all trials for this bin
            flattened_bin_data = [value for trial in bin_data for value in trial]
            
            if flattened_bin_data:
                mean_cos_avoid = np.nanmean(flattened_bin_data)
                means.append(mean_cos_avoid)
                bin_indices.append(bin_idx)

                # Calculate differences with previous bin, per trial (skip for first bin)
                if bin_idx > 0:
                    for trial_data, prev_trial_data in zip(subj_data[bin_idx], subj_data[bin_idx - 1]):
                        if trial_data and prev_trial_data:  # Ensure both bins have data for this trial
                            trial_diff = np.nanmean(trial_data) - np.nanmean(prev_trial_data)
                            all_differences[bin_idx - 1].append(trial_diff)

                # Plot cos_avoid boxplot and scatter points
                ax_cos_avoid.boxplot([flattened_bin_data], positions=[bin_idx], widths=0.6)
                ax_cos_avoid.scatter([bin_idx] * len(flattened_bin_data), flattened_bin_data, color='blue', edgecolor='k', alpha=0.6)

        # Connect means with lines for cos_avoid
        ax_cos_avoid.plot(bin_indices, means, color='red', linestyle='--', marker='o', label='Mean Cos Avoid')

        # Calculate and plot the mean difference per bin
        diff_means = [np.nanmean(diffs) if diffs else 0 for diffs in all_differences]  # Mean of differences
        ax_diff.plot(range(1, n_bins), diff_means, color='green', linestyle='--', marker='o', label='Mean Difference')
        ax_diff.axhline(0, color='gray', linestyle='--', linewidth=1)

        # Set labels and titles
        ax_cos_avoid.set_title(f'Experiment {subj_idx+1} - Cos Avoid')
        ax_cos_avoid.set_xticks(range(n_bins))
        ax_cos_avoid.set_xticklabels(bin_labels)
        ax_cos_avoid.set_ylabel('Cos Avoid')
        ax_cos_avoid.grid(True)

        ax_diff.set_title(f'Experiment {subj_idx+1} - Difference in Cos Avoid')
        ax_diff.set_xticks(range(1, n_bins))
        ax_diff.set_xticklabels(bin_labels[1:])  # Skip first label since no diff for first bin
        ax_diff.set_ylabel('Difference in Cos Avoid')
        ax_diff.grid(True)

    axes[-1].set_xlabel('Distance Bins')
    plt.tight_layout()
    save_plot(fig, "reaction_turns_cos_avoid_diff.pdf")



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

    cos_avoid_data = measure_distance_based(exp_trials, distance_bins)
    # plot_data_over_bins(cos_avoid_data, distance_bins)
    plot_data_and_differences_over_bins(cos_avoid_data, distance_bins)

if __name__ == "__main__":
    pre_encounter_analysis()