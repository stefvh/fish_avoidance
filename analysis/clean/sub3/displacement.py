from matplotlib import pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture

from core.experiment import Experiment
from core.study import study

from analysis.clean.reactions import extract_trials
from analysis.clean.sub3.help import save_plot

import seaborn as sns

def compute_min_post_phase_length(exp_trials):
    """
    Compute the minimum length of the post-phase (frames after end_idx) across all trials and experiments.
    
    Parameters:
        exp_trials (list of lists): Experimental data organized as a list of experiments, 
                                    each containing trials with "end_approach_index" and "data".
    
    Returns:
        int: The minimum number of frames available for the post-phase across all experiments and trials.
    """
    min_post_phase_length = float('inf')  # Start with a large value

    # Iterate over experiments and trials
    for trials in exp_trials:
        for trial in trials:
            trial_data = trial["data"]
            end_idx = trial["end_approach_index"]

            # Calculate available frames after end_idx
            post_length = len(trial_data) - (end_idx + 1)
            if post_length < min_post_phase_length:
                min_post_phase_length = post_length

    # Ensure the minimum is valid
    if min_post_phase_length <= 0:
        raise ValueError("No frames available for the post-phase in some trials.")

    return min_post_phase_length



def plot_post_displacement(exp_trials):
    """
    Plot displacement (scatter and density) in the post-phase for each experiment and combined.
    
    Parameters:
        exp_trials (list of lists): Experimental data organized as a list of experiments,
                                    each containing trials with "end_approach_index" and "data".
    """
    # Determine minimum post-phase length
    min_post_phase_length = 20
    
    # Prepare for plotting
    n_experiments = len(exp_trials)
    fig, axes = plt.subplots(n_experiments + 1, 2, figsize=(16, 4 * (n_experiments + 1)), sharex=False)

    # Initialize combined data
    combined_displacements = []
    combined_total_displacements = []
    combined_end_speeds = []

    # Iterate over experiments
    for exp_idx, trials in enumerate(exp_trials):
        experiment_displacements = []
        experiment_total_displacements = []

        for trial in trials:
            trial_data = trial["data"]
            end_idx = trial["end_approach_index"]

            # Extract consistent post-phase length
            post_phase_data = trial_data.iloc[end_idx + 1:end_idx + 1 + min_post_phase_length]

            # Get positions at start (A) and end (B)
            fish_start = post_phase_data.iloc[0][["x_fish", "y_fish"]].values
            fish_end = post_phase_data.iloc[-1][["x_fish", "y_fish"]].values
            robot_start = post_phase_data.iloc[0][["x_robot", "y_robot"]].values
            robot_end = post_phase_data.iloc[-1][["x_robot", "y_robot"]].values

            # Start
            dx_start = fish_start[0] - robot_start[0]
            dy_start = fish_start[1] - robot_start[1]

            # Displacements
            dx_end = fish_end[0] - robot_end[0]
            dy_end = fish_end[1] - robot_end[1]

            # Displacement
            dx_dsp = dx_end - dx_start
            dy_dsp = dy_end - dy_start
            total_displacement = np.sqrt(dx_dsp ** 2 + dy_dsp ** 2)

            # Mean speed
            speed_end = post_phase_data.iloc[-1]["speed_fish"]

            # Append to experiment-specific and combined data
            experiment_displacements.append((dx_dsp, dy_dsp))
            experiment_total_displacements.append(total_displacement)
            combined_displacements.append((dx_dsp, dy_dsp))
            combined_total_displacements.append(total_displacement)
            combined_end_speeds.append(speed_end)

        # Convert experiment displacements to arrays for easier processing
        x_disp, y_disp = zip(*experiment_displacements)

        # Left Column: Scatter and density plot for dx and dy
        ax = axes[exp_idx, 0]
        ax.scatter(x_disp, y_disp, color="orange", alpha=0.7, label=f"Experiment {exp_idx + 1}")
        sns.kdeplot(x=x_disp, y=y_disp, ax=ax, cmap="Blues", fill=True, alpha=0.5)
        ax.set_title(f"Displacement (dx, dy) - Experiment {exp_idx + 1}")
        ax.set_xlabel("dx (Fish-Robot)")
        ax.set_ylabel("dy (Fish-Robot)")
        ax.legend()
        # Mark (0,0) on the plot
        ax.axvline(0, color="red", linestyle="--")
        ax.axhline(0, color="red", linestyle="--")

        # Right Column: Total displacement scatter and density plot
        ax = axes[exp_idx, 1]
        sns.histplot(experiment_total_displacements, ax=ax, kde=True, color="orange", alpha=0.7, bins=20, label="Density")
        ax.set_title(f"Total Displacement - Experiment {exp_idx + 1}")
        ax.set_xlabel("Total Displacement (Fish-Robot)")
        ax.set_ylabel("Density")
        ax.legend()

    # Combined Plot
    combined_x_disp, combined_y_disp = zip(*combined_displacements)

    # Left Column: Combined dx and dy scatter/density
    ax = axes[-1, 0]
    ax.scatter(combined_x_disp, combined_y_disp, color="green", alpha=0.7, label="Combined")
    sns.kdeplot(x=combined_x_disp, y=combined_y_disp, ax=ax, cmap="Reds", fill=True, alpha=0.5)
    ax.set_title("Combined Displacement (dx, dy)")
    ax.set_xlabel("dx (Fish-Robot)")
    ax.set_ylabel("dy (Fish-Robot)")
    ax.legend()
    ax.axvline(0, color="red", linestyle="--")
    ax.axhline(0, color="red", linestyle="--")

    # Right Column: Combined total displacement
    ax = axes[-1, 1]
    sns.histplot(combined_total_displacements, ax=ax, kde=True, color="green", alpha=0.7, bins=20, label="Density")
    ax.set_title("Combined Total Displacement")
    ax.set_xlabel("Total Displacement (Fish-Robot)")
    ax.set_ylabel("Density")
    ax.legend()

    sns.histplot(combined_end_speeds, ax=ax, kde=True, color="blue", alpha=0.7, bins=20, label="Mean Speed")

    # Reshape data for GMM (GMM expects 2D array)
    data = np.array(combined_end_speeds).reshape(-1, 1)
    
    # Fit GMM
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(data)
    
    # Extract GMM parameters
    means = gmm.means_.flatten()
    variances = gmm.covariances_.flatten()
    weights = gmm.weights_.flatten()
    
    # Overlay GMM components
    x = np.linspace(min(data), max(data), 1000).flatten()
    y = np.zeros_like(x)
    for mean, var, weight in zip(means, variances, weights):
        component = weight * (1 / np.sqrt(2 * np.pi * var)) * np.exp(-((x - mean) ** 2) / (2 * var))
        y += component
        plt.plot(x, component, label=f"Gaussian (mean={mean:.2f})")
    
    # Total GMM distribution
    # Create second y-axis
    ax = ax.twinx()
    ax.plot(x, y, color="red", linewidth=2, label="GMM Total")

    # Final adjustments and save
    plt.tight_layout()
    save_plot(fig, "post_displacement_subplots.pdf")



def plot_time_until_deceleration(exp_trials, min_post_phase_length):
    """
    Plot density of time until deceleration per experiment and combined.
    
    Parameters:
        exp_trials (list of lists): Experimental data organized as a list of experiments,
                                    each containing trials with "end_approach_index" and "data".
        min_post_phase_length (int): Consistent length of the post-phase for analysis.
    """
    n_experiments = len(exp_trials)
    fig, axes = plt.subplots(n_experiments + 1, 1, figsize=(8, 4 * (n_experiments + 1)), sharex=True)

    combined_times = []

    for exp_idx, trials in enumerate(exp_trials):
        experiment_times = []

        for trial in trials:
            trial_data = trial["data"]
            end_idx = trial["end_approach_index"]

            # Extract consistent post-phase length
            post_phase_data = trial_data.iloc[end_idx + 1:end_idx + 1 + min_post_phase_length]

            # Determine time until deceleration
            decel_frames = np.where(post_phase_data["acceleration_fish"] <= 0)[0]

            if len(decel_frames) > 0:
                time_until_deceleration = decel_frames[0]  # First frame where deceleration occurs
            else:
                time_until_deceleration = -1  # No deceleration in the post-phase


            # # Find the frame of maximum speed
            # max_speed_idx = post_phase_data["speed_fish"].idxmax()
            # max_speed_relative_idx = max_speed_idx - post_phase_data.index[0]  # Relative to post-phase

            # # Check for deceleration after the maximum speed frame
            # decel_frames = np.where(post_phase_data.iloc[max_speed_relative_idx + 1:]["acceleration"] <= 0)[0]

            # if len(decel_frames) > 0:
            #     # First deceleration frame relative to the start of post-phase
            #     time_until_deceleration = max_speed_relative_idx + 1 + decel_frames[0]
            # else:
            #     # No deceleration found, assign -1
            #     time_until_deceleration = -1

            experiment_times.append(time_until_deceleration)
            combined_times.append(time_until_deceleration)

        # Plot for this experiment
        ax = axes[exp_idx]
        sns.histplot(
            experiment_times,
            kde=True,
            bins=20,
            ax=ax,
            color="blue",
            alpha=0.7,
            label=f"Experiment {exp_idx + 1}",
        )
        ax.set_title(f"Time Until Deceleration - Experiment {exp_idx + 1}")
        ax.set_xlabel("Time Until Deceleration (frames)")
        ax.set_ylabel("Density")
        ax.legend()

    # Combined plot
    ax = axes[-1]
    sns.histplot(
        combined_times,
        kde=True,
        bins=20,
        ax=ax,
        color="green",
        alpha=0.7,
        label="Combined",
    )
    ax.set_title("Time Until Deceleration - Combined")
    ax.set_xlabel("Time Until Deceleration (frames)")
    ax.set_ylabel("Density")
    ax.legend()

    # Final adjustments and save
    plt.tight_layout()
    save_plot(fig, "time_until_deceleration_subplots.pdf")


if __name__ == "__main__":
    exp_trials = extract_trials()
    plot_post_displacement(exp_trials)
    # plot_time_until_deceleration(exp_trials, min_post_phase_length=20)