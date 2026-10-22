import os
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde

from core.experiment import Experiment
from core.study import study

from analysis.clean.avoidance_mechanisms import get_data
from plotting.publishing.subset_trials import get_trials_by_data

def save_plot(fig, filename):
    output_folder = "analysis/clean/figs/reactions"
    os.makedirs(output_folder, exist_ok=True)
    fig.savefig(os.path.join(output_folder, filename))
    plt.close()
    print(f"Figure saved at {output_folder}/{filename}")

def plot_density(data: np.array, name: str):
    density = gaussian_kde(data)
    x_vals = np.linspace(min(data), max(data), 1000) 
    y_vals = density(x_vals)

    # Plot with matplotlib
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.fill_between(x_vals, y_vals, color="blue", alpha=0.5)
    ax.plot(x_vals, y_vals, color="blue")

    # Labeling the axes
    ax.set_xlabel(name)
    ax.set_ylabel("Density")

    # Save the plot
    save_plot(fig, f"{name}_density.pdf")

def get_baseline_data_fuberlin(do_plot=False):
    import fish_models
    data_folder = fish_models.raw_data("live_ff_clean") 
    dset = fish_models.IoDataset(data_folder)
    speeds = dset["actions"][...,0] # 58 trials, 2 subjects, 8989 frames
    accelerations = np.diff(speeds, axis=-1)
    turns = dset["actions"][...,1]

    if do_plot:
        speeds = speeds.flatten()
        plot_density(speeds, "fuberlin_speed")
        turns = turns.flatten()
        plot_density(turns, "fuberlin_turn")
        accelerations = accelerations.flatten()
        plot_density(accelerations, "fuberlin_acceleration")

    return speeds, accelerations, turns


def plot_baseline(df):
    variables = ['speed_fish', 'turn_heading_fish', 'acceleration_fish']

    # Number of subjects
    num_subjects = 5

    # Define the overall figure layout
    fig, axes = plt.subplots(nrows=6, ncols=len(variables), figsize=(15, 18), sharex='col', gridspec_kw={'hspace': 0})

    # Loop through each variable to create density plots
    for col_idx, var in enumerate(variables):
        # Loop through each subject
        for subject_id in range(num_subjects):
            # Filter data for the specific subject and variable
            subject_data = df[df['experiment_id'] == subject_id][var].dropna()
            
            # If there's data for the subject, compute the density
            if len(subject_data) > 1:  # Avoid issues with empty or single-value data
                # Histogram for the data
                axes[subject_id, col_idx].hist(subject_data, bins=25, density=True, alpha=0.5, color='gray')
            
                # Density plot for the data
                kde = gaussian_kde(subject_data)
                x_vals = np.linspace(subject_data.min(), subject_data.max(), 200)
                y_vals = kde(x_vals)
                axes[subject_id, col_idx].plot(x_vals, y_vals)

                # Compute and show mean and std
                mean = subject_data.mean()
                std = subject_data.std()
                axes[subject_id, col_idx].axvline(mean, color='red', linestyle='-', linewidth=1)
                axes[subject_id, col_idx].axvline(mean - std, color='red', linestyle='--', linewidth=1, alpha=0.5)
                axes[subject_id, col_idx].axvline(mean + std, color='red', linestyle='--', linewidth=1, alpha=0.5)

                # For speed_fish variable, include inset of boxplot, showing IQR
                if var == 'speed_fish':
                    # Create inset
                    inset = axes[subject_id, col_idx].inset_axes([0.6, 0.6, 0.3, 0.3])
                    inset.boxplot(subject_data, vert=False)
                    inset.set_xlim(df[var].min(), df[var].max())
                    inset.set_yticks([])
                
            # Label each row for the specific subject
            if col_idx == 0:
                axes[subject_id, col_idx].set_ylabel(f'Subject {subject_id+1}', rotation=0, labelpad=40)
            
        # Combine all subjects for the combined density plot at the bottom row
        combined_data = df[var].dropna()
        kde_combined = gaussian_kde(combined_data)
        x_vals_combined = np.linspace(combined_data.min(), combined_data.max(), 200)
        y_vals_combined = kde_combined(x_vals_combined)
        axes[5, col_idx].plot(x_vals_combined, y_vals_combined, color='black')
        axes[5, 0].set_ylabel('All', rotation=0, labelpad=40)
        axes[5, col_idx].set_xlabel(var)

    # Remove extra spacing for a cleaner look
    save_plot(fig, "baseline_data.pdf")


def get_baseline_data_subjects(do_plot=False):
    df = get_data()
    df = clean_data(df)
    df = df[df["distance"] > 40]
    if do_plot:
        plot_baseline(df)

    return df


def compute_end_approach(trial):
    trial_data = trial["data"]
    v_app = trial_data["v_app"]

    begin = 5
    eai = begin
    for i, apv in enumerate(v_app[begin:]):
        if apv < 1.0:
            eai = begin + i
            break

    return eai


def measure_distance_based(exp_trials, distance_bins):
    # Initialize data structures to store lists of lists of lists for each experiment
    n_experiments = len(exp_trials)
    speed_data = [[] for _ in range(n_experiments)]
    turn_data = [[] for _ in range(n_experiments)]
    accel_data = [[] for _ in range(n_experiments)]

    # Iterate over each experiment and each trial within the experiment
    for exp_idx, trials in enumerate(exp_trials):
        # Initialize lists of lists for each distance bin within this experiment
        exp_speed_bins = [[] for _ in range(len(distance_bins))]
        exp_turn_bins = [[] for _ in range(len(distance_bins))]
        exp_accel_bins = [[] for _ in range(len(distance_bins))]
        
        for trial in trials:
            # Get data up to end of approach index
            end_idx = trial["end_approach_index"]
            trial_data = trial["data"].iloc[:end_idx]
            
            # Bin the data based on the distance bins
            for bin_idx in range(len(distance_bins) - 1):
                lower_bound = distance_bins[bin_idx + 1]
                upper_bound = distance_bins[bin_idx]
                
                # Filter rows within the current distance bin
                bin_data = trial_data[(trial_data["distance"] >= lower_bound) & (trial_data["distance"] < upper_bound)]               
                
                # Append speed, turn, and acceleration values to respective bin lists
                exp_speed_bins[bin_idx].append(bin_data["speed_fish"].tolist())
                exp_turn_bins[bin_idx].append(bin_data["turn_fish"].tolist())
                exp_accel_bins[bin_idx].append(bin_data["acceleration_fish"].tolist())
            
            # Additional bin for 10 frames after the end of approach
            post_approach_data = trial["data"].iloc[end_idx:end_idx + 10]
            if not post_approach_data.empty:
                exp_speed_bins[-1].append(post_approach_data["speed_fish"].tolist())
                exp_turn_bins[-1].append(post_approach_data["turn_fish"].tolist())
                exp_accel_bins[-1].append(post_approach_data["acceleration_fish"].tolist())
        
        # After processing all trials in the experiment, add lists to main data structures
        speed_data[exp_idx] = exp_speed_bins
        turn_data[exp_idx] = exp_turn_bins
        accel_data[exp_idx] = exp_accel_bins
    
    return speed_data, turn_data, accel_data


def clean_data(df: pd.DataFrame):
    speed_threshold = 0.5  
    df.loc[df['speed_fish'] < speed_threshold, 'turn_fish'] = 0
    df.loc[df['speed_fish'] < speed_threshold, 'cos_avoid'] = np.nan

    return df


def extract_trials(clean=True):
    df = get_data()
    if clean:
        df = clean_data(df)
    exp_trials = get_trials_by_data(df, study.list_experiments)
    
    for trials in exp_trials:
        for trial in trials:
            trial["end_approach_index"] = compute_end_approach(trial)
    
    return exp_trials


def visualize_distance_bin_data(df_baseline,
                                speed_data, 
                                acceleration_data, 
                                turn_data, 
                                distance_bins):
    """
    Visualize mean, standard deviation, coefficient of variation for speed data, and max values of acceleration and turn data
    across distance bins with separate figures for each experiment.
    
    Parameters:
    - speed_data: List of lists of lists for speed values for each experiment, bin, and trial.
    - acceleration_data: List of lists of lists for acceleration values for each experiment, bin, and trial.
    - turn_data: List of lists of lists for turn values for each experiment, bin, and trial.
    - distance_bins: Array-like structure representing the distance bin edges.
    """
    # Define bin labels for x-axis (e.g., 40-35, 35-30, ..., 5-0, Post)
    bin_labels = [f"{distance_bins[i]}-{distance_bins[i+1]}" for i in range(len(distance_bins)-1)]
    bin_labels.append("Post")  # Label for the last "Post" bin

    combined_freq_turn = []
    
    # Iterate over each experiment's data
    for exp_idx, (exp_speed_data, exp_accel_data, exp_turn_data) in enumerate(zip(speed_data, acceleration_data, turn_data)):
        # Compute thresholds from df_baseline
        df_exp_bas = df_baseline[df_baseline["experiment_id"] == exp_idx]
        df_exp_bas['abs_turn_fish'] = df_exp_bas['turn_fish'].abs()
        max_acc_mean = df_exp_bas['acceleration_fish'].mean()
        max_acc_std = df_exp_bas['acceleration_fish'].std()
        max_acc_95th = df_exp_bas['acceleration_fish'].quantile(0.95)

        max_turn_mean = df_exp_bas['abs_turn_fish'].mean()
        max_turn_std = df_exp_bas['abs_turn_fish'].std()
        max_turn_95th = df_exp_bas['abs_turn_fish'].quantile(0.975)
        max_turn_95th = max_turn_mean + 2 * max_turn_std

        speed_95th = df_exp_bas['speed_fish'].quantile(0.95)

        # Initialize lists to hold values for each statistic across bins for this experiment
        means_per_bin = []
        stds_per_bin = []
        cvs_per_bin = []
        max_accel_per_bin = []
        max_abs_turn_per_bin = []

        # Iterate over each bin in the current experiment
        for bin_idx in range(len(distance_bins)):
            # Initialize lists for speed statistics
            bin_means = []
            bin_stds = []
            bin_cvs = []
            
            # Calculate speed statistics (mean, std, CV) for each trial in this bin
            if bin_idx < len(exp_speed_data):  # Check if bin exists in experiment data
                bin_trials = exp_speed_data[bin_idx]

                for trial_values in bin_trials:
                    if len(trial_values) > 0:
                        trial_mean = np.mean(trial_values)
                        trial_std = np.std(trial_values)
                        trial_cv = trial_std / trial_mean if trial_mean != 0 else np.nan

                        bin_means.append(trial_mean)
                        bin_stds.append(trial_std)
                        bin_cvs.append(trial_cv)

            # Append speed statistics to overall lists for this experiment
            means_per_bin.append(bin_means)
            stds_per_bin.append(bin_stds)
            cvs_per_bin.append(bin_cvs)

            # Calculate max acceleration for each trial in this bin
            if bin_idx < len(exp_accel_data):
                max_accel_per_bin.append([np.max(trial) for trial in exp_accel_data[bin_idx] if len(trial) > 0])

            # Calculate max absolute turn for each trial in this bin
            if bin_idx < len(exp_turn_data):
                max_abs_turn_per_bin.append([np.max(np.abs(trial)) for trial in exp_turn_data[bin_idx] if len(trial) > 0])

        # Calculate frequency of prominent turns and accelerations for each bin
        prominent_speed_freq = []
        prominent_turn_freq = []
        prominent_accel_freq = []

        max_trials = max([trial_count for trial_count in map(len, exp_turn_data)])

        for bin_idx in range(len(distance_bins)):
            total_trials = len(exp_turn_data[bin_idx])
            prominent_speeds = sum(
                any(speed > speed_95th for speed in trial) for trial in exp_speed_data[bin_idx]
            )
            prominent_turns = sum(
                any(abs(turn) > max_turn_95th for turn in trial) for trial in exp_turn_data[bin_idx]
            )
            prominent_accels = sum(
                any(accel > max_acc_95th for accel in trial) for trial in exp_accel_data[bin_idx]
            )

            prominent_speed_freq.append(prominent_speeds / max_trials if total_trials > 0 else 0)
            prominent_turn_freq.append(prominent_turns / max_trials if total_trials > 0 else 0)
            prominent_accel_freq.append(prominent_accels / max_trials if total_trials > 0 else 0)
        
        combined_freq_turn.append(prominent_turn_freq)

        # Set up the figure and axes for the current experiment
        fig, axes = plt.subplots(nrows=6, ncols=1, figsize=(12, 15), sharex=True, gridspec_kw={'hspace': 0})
        fig.suptitle(f'Experiment {exp_idx+1}')

        # Row 1: Mean of Speed
        axes[0].boxplot(means_per_bin, positions=range(len(distance_bins)), widths=0.6)
        for idx, values in enumerate(means_per_bin):
            axes[0].scatter([idx] * len(values), values, color='blue', alpha=0.6)
        axes[0].set_ylabel('Mean')
        axes[0].axhline(speed_95th, color='purple', linestyle='--', label='95th Percentile')
        # Inset for frequency of prominent speeds
        inset_speed = axes[0].inset_axes([0.1, 0.6, 0.3, 0.3])
        inset_speed.bar(range(len(prominent_speed_freq)), prominent_speed_freq, color='gray')
        inset_speed.set_xticks(range(len(bin_labels)))
        inset_speed.set_xticklabels(bin_labels, rotation=45, fontsize=8)
        inset_speed.set_ylabel('Freq', fontsize=8)
        inset_speed.axhline(0.05, color='purple', linestyle='-', linewidth=0.8, label='Baseline 5%')

        # Row 2: Standard Deviation of Speed
        axes[1].boxplot(stds_per_bin, positions=range(len(distance_bins)), widths=0.6)
        for idx, values in enumerate(stds_per_bin):
            axes[1].scatter([idx] * len(values), values, color='orange', alpha=0.6)
        axes[1].set_ylabel('Std Dev')

        # Row 3: Coefficient of Variation (CV) of Speed
        axes[2].boxplot(cvs_per_bin, positions=range(len(distance_bins)), widths=0.6)
        for idx, values in enumerate(cvs_per_bin):
            axes[2].scatter([idx] * len(values), values, color='green', alpha=0.6)
        axes[2].set_ylabel('CV')

        # Row 4: Max Acceleration
        axes[3].boxplot(max_accel_per_bin, positions=range(len(distance_bins)), widths=0.6)
        for idx, values in enumerate(max_accel_per_bin):
            axes[3].scatter([idx] * len(values), values, color='red', alpha=0.6)
        axes[3].set_ylabel('Max Accel')
        axes[3].axhline(max_acc_mean + max_acc_std, color='blue', linestyle='--', label='Mean + 1*Std')
        axes[3].axhline(max_acc_mean + 2 * max_acc_std, color='green', linestyle='--', label='Mean + 2*Std')
        axes[3].axhline(max_acc_95th, color='purple', linestyle='--', label='95th Percentile')
        # axes[3].legend()
        # Inset for frequency of prominent accelerations
        inset_accel = axes[3].inset_axes([0.1, 0.6, 0.3, 0.3])
        inset_accel.bar(range(len(prominent_accel_freq)), prominent_accel_freq, color='gray')
        inset_accel.set_xticks(range(len(bin_labels)))
        inset_accel.set_xticklabels(bin_labels, rotation=45, fontsize=8)
        inset_accel.set_ylabel('Freq', fontsize=8)
        # inset_accel.set_xlabel('Distance', fontsize=8)
        inset_accel.axhline(0.05, color='purple', linestyle='-', linewidth=0.8, label='Baseline 5%')

        # Row 5: Max Absolute Turn
        axes[4].boxplot(max_abs_turn_per_bin, positions=range(len(distance_bins)), widths=0.6)
        for idx, values in enumerate(max_abs_turn_per_bin):
            axes[4].scatter([idx] * len(values), values, color='purple', alpha=0.6)
        axes[4].set_ylabel('Max |Turn|')
        # axes[4].set_xticks(range(len(bin_labels)))
        # axes[4].set_xticklabels(bin_labels, rotation=45)
        # axes[4].set_xlabel('Distance Bin')
        axes[4].axhline(max_turn_mean + max_turn_std, color='blue', linestyle='--', label='Mean + 1*Std')
        axes[4].axhline(max_turn_mean + 2 * max_turn_std, color='green', linestyle='--', label='Mean + 2*Std')
        axes[4].axhline(max_turn_95th, color='purple', linestyle='--', label='95th Percentile')
        # axes[4].legend()
        # Inset for frequency of prominent turns
        inset_turn = axes[4].inset_axes([0.1, 0.6, 0.3, 0.3])
        inset_turn.bar(range(len(prominent_turn_freq)), prominent_turn_freq, color='gray')
        inset_turn.set_xticks(range(len(bin_labels)))
        inset_turn.set_xticklabels(bin_labels, rotation=45, fontsize=8)
        inset_turn.set_ylabel('Freq', fontsize=8)
        # inset_turn.set_xlabel('Distance', fontsize=8)
        inset_turn.axhline(0.05, color='purple', linestyle='-', linewidth=0.8, label='Baseline 5%')

        # Row 6:  Per-Trial Differences in Mean Speed
        trial_differences = []
        for bin_idx in range(1, len(means_per_bin)):
            prev_bin_means = means_per_bin[bin_idx - 1]
            curr_bin_means = means_per_bin[bin_idx]
            bin_diffs = [curr - prev for curr, prev in zip(curr_bin_means, prev_bin_means)]
            trial_differences.append(bin_diffs)
        axes[5].boxplot(trial_differences, positions=range(1, len(distance_bins)), widths=0.6)
        for idx, values in enumerate(trial_differences):
            axes[5].scatter([idx + 1] * len(values), values, color='brown', alpha=0.6)
        axes[5].axhline(0, color='black', linestyle='-')
        axes[5].set_ylabel('Diff in Mean')
        axes[5].set_xticks(range(len(bin_labels)))
        axes[5].set_xticklabels(bin_labels, rotation=45)
        axes[5].set_xlabel('Distance Bin')
        
        # Display the plot
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
        save_plot(fig, f"e={exp_idx+1}_distance_stats.pdf")
    
    # Plot combined freq turn
    combined_freq_turn = np.array(combined_freq_turn)
    mean_freq_turn = np.mean(combined_freq_turn, axis=0)
    std_freq_turn = np.std(combined_freq_turn, axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.bar(range(len(bin_labels)), mean_freq_turn, yerr=std_freq_turn, color='gray')
    ax.set_xticks(range(len(bin_labels)))
    ax.set_xticklabels(bin_labels, rotation=45)

    ax.set_ylabel('Frequency of Prominent Turns')
    ax.set_xlabel('Distance Bin')
    save_plot(fig, "combined_freq_turn.pdf")



def measure_prominent_turns(exp_trials, df_baseline, distance_bins):
    prominent_turns_data = []

    for exp_idx, trials in enumerate(exp_trials):
        exp_prominent_turns = [[] for _ in range(len(distance_bins) - 1)]

        df_exp_bas = df_baseline[df_baseline["experiment_id"] == exp_idx]
        df_exp_bas['abs_turn_fish'] = df_exp_bas['turn_fish'].abs()
        turn_threshold = df_exp_bas['abs_turn_fish'].quantile(0.95)

        for trial in trials:
            end_idx = trial["end_approach_index"]
            trial_data = trial["data"].iloc[:end_idx]

            for bin_idx in range(len(distance_bins) - 1):
                lower_bound = distance_bins[bin_idx + 1]
                upper_bound = distance_bins[bin_idx]

                bin_data = trial_data[(trial_data["distance"] >= lower_bound) & (trial_data["distance"] < upper_bound)]

                for _, row in bin_data.iterrows():
                    if abs(row['turn_fish']) > turn_threshold:
                        current_frame = row['frame']
                        
                        # Check if there are enough frames before and after
                        pre_turn_data = trial_data[trial_data['frame'] == current_frame - 2]
                        post_turn_data = trial_data[trial_data['frame'] == current_frame + 4]
                        
                        if not pre_turn_data.empty and not post_turn_data.empty:
                            before_cos_avoid = pre_turn_data['cos_avoid'].values[0]
                            after_cos_avoid = post_turn_data['cos_avoid'].values[0]
                            exp_prominent_turns[bin_idx].append((before_cos_avoid, after_cos_avoid))
        
        
        prominent_turns_data.append(exp_prominent_turns)

    return prominent_turns_data


def plot_prominent_turns(prominent_turns_data, distance_bins):
    n_experiments = len(prominent_turns_data)
    n_bins = len(distance_bins) - 1

    fig, axes = plt.subplots(nrows=n_bins, ncols=n_experiments, figsize=(n_experiments * 5, n_bins * 4), sharex=True, sharey=True)
    
    if n_experiments == 1:
        axes = np.expand_dims(axes, axis=1)

    for exp_idx, exp_data in enumerate(prominent_turns_data):
        for bin_idx, bin_data in enumerate(exp_data):
            ax = axes[bin_idx, exp_idx]
            if bin_data:
                before, after = zip(*bin_data)
                ax.scatter(before, after, alpha=0.6)
            
            ax.set_title(f"Exp {exp_idx+1}, Bin {distance_bins[bin_idx]}-{distance_bins[bin_idx+1]}")
            if bin_idx == n_bins - 1:
                ax.set_xlabel('Before Turn')
            if exp_idx == 0:
                ax.set_ylabel('After Turn')
            
            # plot y = x line
            lims = [
                (-1, -1),
                (1, 1)
            ]
            ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
            ax.axhline(0, color='black', linestyle='-', alpha=0.5)

    plt.tight_layout()
    save_plot(fig, "prominent_turns_cos_avoid_analysis.pdf")


def measure_prominent_turns_speed_accel(exp_trials, df_baseline, distance_bins):
    max_speed_data = []
    max_accel_data = []

    for exp_idx, trials in enumerate(exp_trials):
        exp_max_speed = [[] for _ in range(len(distance_bins) - 1)]
        exp_max_accel = [[] for _ in range(len(distance_bins) - 1)]

        df_exp_bas = df_baseline[df_baseline["experiment_id"] == exp_idx]
        df_exp_bas['abs_turn_fish'] = df_exp_bas['turn_fish'].abs()
        turn_threshold = df_exp_bas['abs_turn_fish'].quantile(0.95)

        for trial in trials:
            end_idx = trial["end_approach_index"]
            trial_data = trial["data"].iloc[:end_idx]

            for bin_idx in range(len(distance_bins) - 1):
                lower_bound = distance_bins[bin_idx + 1]
                upper_bound = distance_bins[bin_idx]

                bin_data = trial_data[(trial_data["distance"] >= lower_bound) & (trial_data["distance"] < upper_bound)]

                for _, row in bin_data.iterrows():
                    if abs(row['turn_fish']) > turn_threshold:
                        current_frame = row['frame']

                        # Extract 8 frames before and after
                        pre_turn_data = trial_data[(trial_data['frame'] >= current_frame - 8) & (trial_data['frame'] < current_frame)]
                        post_turn_data = trial_data[(trial_data['frame'] > current_frame) & (trial_data['frame'] <= current_frame + 8)]

                        if len(pre_turn_data) >= 4 and len(post_turn_data) >= 4:
                            max_speed_before = pre_turn_data['speed_fish'].max()
                            max_speed_after = post_turn_data['speed_fish'].max()

                            max_accel_before = pre_turn_data['acceleration_fish'].max()
                            max_accel_after = post_turn_data['acceleration_fish'].max()

                            exp_max_speed[bin_idx].append((max_speed_before, max_speed_after))
                            exp_max_accel[bin_idx].append((max_accel_before, max_accel_after))

        max_speed_data.append(exp_max_speed)
        max_accel_data.append(exp_max_accel)

    return max_speed_data, max_accel_data


def plot_max_speed_accel(max_speed_data, max_accel_data, distance_bins):
    n_experiments = len(max_speed_data)
    n_bins = len(distance_bins) - 1

    # Plot max speed
    fig_speed, axes_speed = plt.subplots(nrows=n_bins, ncols=n_experiments, figsize=(n_experiments * 5, n_bins * 4), sharex=True, sharey=True)
    fig_accel, axes_accel = plt.subplots(nrows=n_bins, ncols=n_experiments, figsize=(n_experiments * 5, n_bins * 4), sharex=True, sharey=True)

    if n_experiments == 1:
        axes_speed = np.expand_dims(axes_speed, axis=1)
        axes_accel = np.expand_dims(axes_accel, axis=1)

    for exp_idx, (exp_speed_data, exp_accel_data) in enumerate(zip(max_speed_data, max_accel_data)):
        for bin_idx, (speed_data, accel_data) in enumerate(zip(exp_speed_data, exp_accel_data)):
            ax_speed = axes_speed[bin_idx, exp_idx]
            ax_accel = axes_accel[bin_idx, exp_idx]

            if speed_data:
                speed_before, speed_after = zip(*speed_data)
                ax_speed.scatter(speed_before, speed_after, alpha=0.6)
            
            if accel_data:
                accel_before, accel_after = zip(*accel_data)
                ax_accel.scatter(accel_before, accel_after, alpha=0.6)

            ax_speed.set_title(f"Exp {exp_idx+1}, Bin {distance_bins[bin_idx]}-{distance_bins[bin_idx+1]}")
            ax_accel.set_title(f"Exp {exp_idx+1}, Bin {distance_bins[bin_idx]}-{distance_bins[bin_idx+1]}")
            
            if bin_idx == n_bins - 1:
                ax_speed.set_xlabel('Max Speed Before Turn')
                ax_accel.set_xlabel('Max Accel Before Turn')
            if exp_idx == 0:
                ax_speed.set_ylabel('Max Speed After Turn')
                ax_accel.set_ylabel('Max Accel After Turn')
    
    accel_lims = [(-10, 0), (60, 175)]
    speed_lims = [(0, 0), (20, 35)]

    for ax_speed, ax_accel in zip(axes_speed.flatten(), axes_accel.flatten()):
        ax_speed.plot(speed_lims, speed_lims, 'k-', alpha=0.75, zorder=0)
        ax_accel.plot(accel_lims, accel_lims, 'k-', alpha=0.75, zorder=0)
            
    plt.tight_layout()
    save_plot(fig_speed, "max_speed_before_after_turns.pdf")
    save_plot(fig_accel, "max_accel_before_after_turns.pdf")



def pre_encounter_analysis():
    exp_trials = extract_trials()

    df_baseline = get_baseline_data_subjects()
    distance_bins = list(range(40, -5, -5))  # Generates [40, 35, 30, ..., 5, 0]

    # speed_data, turn_data, accel_data = measure_distance_based(exp_trials, distance_bins)

    # visualize_distance_bin_data(df_baseline,
    #                             speed_data, 
    #                             accel_data, 
    #                             turn_data, 
    #                             distance_bins)

    prominent_turns_data = measure_prominent_turns(exp_trials, df_baseline, distance_bins)
    plot_prominent_turns(prominent_turns_data, distance_bins)

    # max_speed_data, max_accel_data = measure_prominent_turns_speed_accel(exp_trials, df_baseline, distance_bins)
    # plot_max_speed_accel(max_speed_data, max_accel_data, distance_bins)


if __name__ == "__main__":
    pre_encounter_analysis()
