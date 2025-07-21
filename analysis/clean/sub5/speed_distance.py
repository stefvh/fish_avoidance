# Analyse robot's approach speed vs encounter distance
# -> Hypothesis: Do robots with lower approach speed get closer to the fish?

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


from analysis.one.wrapper_fixed import run_plm_model
from core.experiment import Experiment
from core.study import study

from analysis.clean.reactions import extract_trials
from analysis.clean.sub5.help import save_plot


def get_speed_distance_by_index(trial_data, end_idx):
    np_speed_robot = trial_data["speed_robot"].to_numpy()
    begin_idx = max(0, end_idx - 10)
    speed = np.mean(np_speed_robot[begin_idx:end_idx])

    np_distance = trial_data["distance"].to_numpy()
    distance = np.min(np_distance[end_idx:end_idx+4])

    return speed, distance


def get_speed_distance_by_arb(trial_data):
    np_distance = trial_data["distance"].to_numpy()
    ef = np.argmin(np_distance)
    max_t = np_distance.shape[0]
    td = 10
    approach_phase = (0, max(0, ef - td))

    np_speed_robot = trial_data["speed_robot"].to_numpy()
    end = approach_phase[1]
    begin = max(0, end - 10)
    speed = np.mean(np_speed_robot[begin:end])
    distance = np_distance[ef]

    return speed, distance


def plot_speed_distance(exp_trials):
    # Initialize lists to hold combined angle_fr and fish_speed values
    a_distances = []
    a_speeds = []

    b_distances = []
    b_speeds = []
    
    # Calculate the number of experiments and set up the figure
    n_experiments = len(exp_trials)

    subject_ids = []
    trial_ids = []

    # Iterate over each experiment
    for exp_idx, trials in enumerate(exp_trials):
        for trial in trials:
            # Get data up to end of approach index
            end_idx = trial["end_approach_index"]
            trial_data = trial["data"]
            
            # Get the speed and distance values
            a_speed, a_distance = get_speed_distance_by_index(trial_data, end_idx)
            if a_speed is not None and a_distance is not None:
                a_speeds.append(a_speed)
                a_distances.append(a_distance)
                subject_ids.append(trial["experiment_id"])
                trial_ids.append(trial["m_trial_id"])

            # Get the speed and distance values
            b_speed, b_distance = get_speed_distance_by_arb(trial_data)
            if b_speed is not None and b_distance is not None:
                b_speeds.append(b_speed)
                b_distances.append(b_distance)
    
    # Create the figure and axis
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    # Plot the data
    axes[0].scatter(a_speeds, a_distances, color='blue', alpha=0.5)
    # label each scatter point with subject and trial id
    for i, txt in enumerate(trial_ids):
        axes[0].annotate(txt, (a_speeds[i], a_distances[i]))
    for i, txt in enumerate(subject_ids):
        axes[0].annotate("/" + str(txt), (a_speeds[i] + 0.4, a_distances[i]))
    axes[0].set_title("By approach end index")
    axes[0].set_xlabel("Approach Speed (cm/s)")
    axes[0].set_ylabel("Encounter Distance (cm)")

    axes[1].scatter(b_speeds, b_distances, color='red', alpha=0.5)
    axes[1].set_title("By phases")
    axes[1].set_xlabel("Approach Speed (cm/s)")
    axes[1].set_ylabel("Encounter Distance (cm)")
    
    # Save the plot
    plt.tight_layout()
    save_plot(fig, "approach_speed_encounter_distance.pdf")

    # Statistics
    df = pd.DataFrame({
        "speed": b_speeds,
        "distance": b_distances,
        "subject_id": subject_ids,
        "trial_id": trial_ids,
    })
    statistics(df)


def statistics(df: pd.DataFrame):
    result = run_plm_model(
        df, 
        "distance", 
        ["speed"], 
        ["subject_id", "trial_id"], 
        "within", 
        "twoways",
    )
    print(result)


if __name__ == "__main__":
    exp_trials = extract_trials()
    plot_speed_distance(exp_trials)