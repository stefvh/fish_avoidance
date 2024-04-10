import pandas as pd
from utils.parser import parse_args

from pre_process.experiment import Experiment, get_experiment
from utils.plotter import plot_multipdf

def annotate_row(ax, trial_id, min_robot_distance):
    x_textpad = -0.45
    annotate_color = "black"

    annotation_text = r'(' + f'{str(trial_id)}) '
    annotation_text += r'$d_{\min}=$' + f'{min_robot_distance:.1f}cm'
    
    ax.annotate(annotation_text, xy=(x_textpad, 0.45), xycoords="axes fraction",
            va='center', rotation="vertical", fontsize=12, weight='bold', color=annotate_color)

def plot_trial(ax_row, i, j, trial_id, results):
    k = i + j
    trial_results = results[results["trial_id"] == trial_id]

    relative_frame_numbers = trial_results["frame_number"] - trial_results["frame_number"].min()

    min_robot_distance = trial_results["robot_distance"].min()
    argmin_robot_distance = trial_results["robot_distance"].idxmin()
    x_rel_min = trial_results["frame_number"][argmin_robot_distance] - trial_results["frame_number"].min()
    for ax in ax_row:
        ax.axvline(x=x_rel_min, color="red", linestyle="-")

    approaching_frames = trial_results[trial_results["state"].str.contains("APPROACHING")]["frame_number"]
    if len(approaching_frames) > 0:
        peak_frame = approaching_frames.max()
        for ax in ax_row:
            ax.axvline(x=peak_frame - trial_results["frame_number"].min(), color="green", linestyle="-")

    column = 0

    # Plot robot distance
    ax = ax_row[column]
    ax.plot(relative_frame_numbers, trial_results["robot_distance_batch"])
    if j ==0:
        ax.set_title("Robot Distance")
    ax.set_xlabel("Frame Number")
    ax.set_ylabel("Distance (cm)")

    max_robot_distance = results["robot_distance"].max()
    ax.set_ylim(0, max_robot_distance)

    annotate_row(ax, trial_results["trial_id"].iloc[0], min_robot_distance)

    column +=1

    # Plot fish velocity
    ax = ax_row[column]
    ax.plot(relative_frame_numbers, trial_results["fish_velocity_batch"])
    if j ==0:
        ax.set_title("Fish Velocity")
    ax.set_xlabel("Frame Number")
    ax.set_ylabel("Velocity (cm/s)")

    max_fish_velocity = results["fish_velocity_batch"].max()
    ax.set_ylim(0, max_fish_velocity)

    column +=1

    # Plot robot distance slope
    ax = ax_row[column]
    ax.plot(relative_frame_numbers, trial_results["robot_distance_slopes"])
    if j ==0:
        ax.set_title("Distance Slope")
    ax.set_xlabel("Frame Number")
    ax.set_ylabel("Slope (cm/s)")

    max_robot_distance_slope = results["robot_distance_slopes"].max()
    min_robot_distance_slope = results["robot_distance_slopes"].min()
    ax.set_ylim(min_robot_distance_slope, max_robot_distance_slope)

    column +=1

    # Plot robot velocity
    ax = ax_row[column]
    ax.plot(relative_frame_numbers, trial_results["robot_velocity_batch"])
    if j ==0:
        ax.set_title("Robot Velocity")
    ax.set_xlabel("Frame Number")
    ax.set_ylabel("Velocity (cm/s)")

    max_robot_velocity = results["robot_velocity_batch"].max()
    ax.set_ylim(0, max_robot_velocity)

    column +=1

    # Plot robot state
    ax = ax_row[column]
    ax.plot(relative_frame_numbers, trial_results["state"])
    if j ==0:
        ax.set_title("Robot State")
    ax.set_xlabel("Frame Number")
    ax.set_ylabel("State")
    # set y-labels on right
    ax.yaxis.tick_right()

    column +=1




def run(experiment: Experiment):
    filepath = f"plotting/observation/peak_repulsion_ids/{experiment.experiment_datetime}.csv"
    cv_input = pd.read_csv(filepath)
    trial_ids = list(cv_input["trial_id"])

    results = experiment.results

    # 
    robot_states = experiment.robot_states
    results = results.merge(robot_states, on="frame_number", how="left")
    results = results.dropna(subset=["state"])

    plot_multipdf(
        len(trial_ids),
        5,
        5,
        lambda ax_row, i, j: plot_trial(
            ax_row=ax_row,
            i=i, 
            j=j,
            trial_id=trial_ids[i+j],
            results=results,
        ),
        f"plotting/hypotheses/peak_repulsion.pdf",
    )

if __name__ == "__main__":
    args = parse_args()
    variables = vars(args)

    experiment = get_experiment(**variables)
    run(experiment)
