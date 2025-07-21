from matplotlib import pyplot as plt
import numpy as np
from scipy import stats

from core.experiment import Experiment
from core.study import study

from analysis.full.get_peak_data import get_peaks_data
from analysis.full.run_reg_fe import get_sampled_dataframe

from scipy.stats import pearsonr, spearmanr


def plot_behaviors(behaviors):
    # Number of subjects
    num_subjects = len(behaviors['freezing'])

    # Calculate percentages
    freezing_percent = np.array(behaviors['freezing']) / np.array(behaviors['total_time']) * 100
    escape_percent = np.array(behaviors['escape']) / np.array(behaviors['total_time']) * 100
    wall_hugging_percent = np.array(behaviors['wall_hugging']) / np.array(behaviors['total_time']) * 100

    # Bar width and positions
    bar_width = 0.2
    indices = np.arange(num_subjects)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plotting bars
    bar1 = ax.bar(indices - bar_width, freezing_percent, bar_width, label='Freezing', color='#ff9999', edgecolor='black')  # Pink
    bar2 = ax.bar(indices, escape_percent, bar_width, label='Escape', color='#99ccff', edgecolor='black')  # Blue
    bar3 = ax.bar(indices + bar_width, wall_hugging_percent, bar_width, label='Wall Hugging', color='#99ff99', edgecolor='black')  # Green

    # Adding percentage labels in the center of each bar
    for bars in [bar1, bar2, bar3]:
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval / 2, f'{yval:.1f}%', ha='center', va='center', rotation=90)

    # Adding labels and title
    # ax.set_xlabel('Subjects')
    ax.set_ylabel('Percentage of Time (%)')
    ax.set_title('Behavioral Percentages for Each Subject')
    ax.set_xticks(indices)
    ax.set_xticklabels([f'Subject {i+1}' for i in range(num_subjects)])
    ax.legend()

    # Display plot
    fig.tight_layout()
    fig.savefig(f"{study.plot_output_folder}/individual_behaviors.pdf")


if __name__ == "__main__":
    data = study.data

    behaviors = {
        "freezing": [],
        "escape": [],
        "wall_hugging": [],
        "total_time": [],
    }

    experiment_ids = data["experiment_id"].unique()

    for experiment_id in experiment_ids:
        df = data[data["experiment_id"] == experiment_id]

        freezing = df[df["speed_fish"] < 2].shape[0]
        escape = df[df["rev"] >= 2].shape[0]
        wall_hugging = df[df["min_wall_distance_fish"] < 10].shape[0]

        behaviors["freezing"].append(freezing)
        behaviors["escape"].append(escape)
        behaviors["wall_hugging"].append(wall_hugging)
        behaviors["total_time"].append(df.shape[0])

    # plot_behaviors(behaviors)

    # Escape and wall-hugging data
    escape = np.array(behaviors['escape'])
    wall_hugging = np.array(behaviors['wall_hugging'])

    # Pearson's correlation
    pearson_corr, pearson_p = pearsonr(escape, wall_hugging)

    # Spearman's correlation
    spearman_corr, spearman_p = spearmanr(escape, wall_hugging)

    print(f"Pearson correlation: {pearson_corr:.2f} (p-value: {pearson_p:.4f})")
    print(f"Spearman correlation: {spearman_corr:.2f} (p-value: {spearman_p:.4f})")