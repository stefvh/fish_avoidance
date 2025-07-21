import numpy as np
import pandas as pd

from core.experiment import Experiment
from core.study import study

import matplotlib.pyplot as plt
import seaborn as sns

def create_behavior_heatmaps(results, apv_bins, distance_bins):
    subjects = results['experiment_id'].unique()
    behaviors = ['freezing', 'escape', 'wall_hugging']
    
    # Create the figure with subplots
    fig, axes = plt.subplots(nrows=len(subjects), ncols=len(behaviors), figsize=(15, 5*len(subjects)))
    
    for i, subject in enumerate(subjects):
        subject_data = results[results['experiment_id'] == subject]
        
        for j, behavior in enumerate(behaviors):
            # Bin the data for apv and distance
            subject_data['apv_bin'] = pd.cut(subject_data['avg_apv'], bins=apv_bins)
            subject_data['distance_bin'] = pd.cut(subject_data['avg_distance'], bins=distance_bins)
            
            # Create a pivot table to calculate the percentage of the behavior in each bin
            pivot_table = subject_data.pivot_table(index='apv_bin', 
                                                   columns='distance_bin', 
                                                   values=behavior, 
                                                   aggfunc='mean') * 100  # Convert to percentage
            
            # Plot the heatmap for each behavior for this subject
            ax = axes[i, j] if len(subjects) > 1 else axes[j]  # Adjust if there is only one subject
            sns.heatmap(pivot_table, ax=ax, cmap='coolwarm', cbar=True, annot=True, fmt=".1f")
            
            # Set the titles
            ax.set_title(f'Subject {subject} - {behavior.capitalize()}')
            ax.set_xlabel('Distance')
            ax.set_ylabel('APV')
    
    # Adjust layout
    plt.tight_layout()
    fig.savefig(f"{study.plot_output_folder}/behavior_heatmaps.pdf")

def create_population_behavior_heatmaps(results, apv_bins, distance_bins):
    behaviors = ['freezing', 'escape', 'wall_hugging']
    
    # Initialize dictionaries to hold the mean and std heatmap data
    mean_heatmap_data = {behavior: np.zeros((len(apv_bins) - 1, len(distance_bins) - 1)) for behavior in behaviors}
    std_heatmap_data = {behavior: np.zeros((len(apv_bins) - 1, len(distance_bins) - 1)) for behavior in behaviors}
    
    # Bin the data for all subjects
    results['apv_bin'] = pd.cut(results['avg_apv'], bins=apv_bins)
    results['distance_bin'] = pd.cut(results['avg_distance'], bins=distance_bins)
    
    for behavior in behaviors:
        # Create pivot tables for each behavior and compute mean and std across all subjects
        mean_pivot = results.pivot_table(index='apv_bin', 
                                         columns='distance_bin', 
                                         values=behavior, 
                                         aggfunc='mean') * 100  # Convert to percentage
        std_pivot = results.pivot_table(index='apv_bin', 
                                        columns='distance_bin', 
                                        values=behavior, 
                                        aggfunc='std') * 100  # Convert to percentage
        
        # Store the data
        mean_heatmap_data[behavior] = mean_pivot
        std_heatmap_data[behavior] = std_pivot

    # Create the figure with 2 rows (mean and std) and 3 columns (one per behavior)
    fig, axes = plt.subplots(nrows=2, ncols=len(behaviors), figsize=(15, 10))
    
    # Format the bin labels to show fewer digits (e.g., integers or rounded values)
    apv_bin_labels = [f"{int(bin.left)}-{int(bin.right)}" for bin in mean_pivot.index]
    distance_bin_labels = [f"{int(bin.left)}-{int(bin.right)}" for bin in mean_pivot.columns]


    for j, behavior in enumerate(behaviors):
        # Plot the mean heatmap
        sns.heatmap(mean_heatmap_data[behavior], ax=axes[0, j], cmap='coolwarm', cbar=True, annot=True, fmt=".1f",
                    xticklabels=distance_bin_labels, yticklabels=apv_bin_labels)
        axes[0, j].set_title(f'Mean {behavior.capitalize()}')
        axes[0, j].set_xlabel('Distance')
        axes[0, j].set_ylabel('APV')
        
        # Plot the std heatmap
        sns.heatmap(std_heatmap_data[behavior], ax=axes[1, j], cmap='coolwarm', cbar=True, annot=True, fmt=".1f",
                    xticklabels=distance_bin_labels, yticklabels=apv_bin_labels)
        axes[1, j].set_title(f'Std {behavior.capitalize()}')
        axes[1, j].set_xlabel('Distance')
        axes[1, j].set_ylabel('APV')
    
    # Adjust layout
    plt.tight_layout()
    fig.savefig(f"{study.plot_output_folder}/behavior_pop_heatmaps.pdf")


def get_results(start_frame_window, end_frame_window):
    results = []
    
    for exp in study.list_experiments:
        segments = exp.relative_motion_quantities.df_segments
        for segment in segments:
            df = segment.copy()
            df = df.dropna()

            if len(df) < 2 * start_frame_window + 2 * end_frame_window:
                continue

            # Loop through the dataset with different start frames
            for start_frame in range(min(df['frame']), max(df['frame']) - 2 * start_frame_window + 1):
                # Select frames within the window (start_frame to start_frame + frame_window)
                subset = df[(df['frame'] >= start_frame) & (df['frame'] < start_frame + start_frame_window)]
                
                avg_apv = subset['apv'].mean()
                avg_distance = subset['distance'].mean()

                # Select frames for averaging (start_frame + frame_window to start_frame + 2 * frame_window)
                later_frames = df[(df['frame'] >= start_frame + end_frame_window) & 
                                  (df['frame'] < start_frame + 2 * end_frame_window)]
                
                # Calculate averages of rev and speed_fish
                avg_rev = later_frames['rev'].max()
                avg_speed_fish = later_frames['speed_fish'].max()
                avg_min_wall_distance_fish = later_frames['min_wall_distance_fish'].mean()

                freezing = avg_speed_fish < 2
                escape = avg_rev >= 2
                wall_hugging = avg_min_wall_distance_fish < 10
                
                # Append the result as a dictionary to the results list
                results.append({
                    'experiment_id': exp.id,
                    'start_frame': start_frame,
                    'avg_apv': avg_apv,
                    'avg_distance': avg_distance,
                    'freezing': freezing,
                    'escape': escape,
                    'wall_hugging': wall_hugging
                })

    # Convert results into a DataFrame for easy viewing
    results_df = pd.DataFrame(results)
    return results_df


if __name__ == "__main__":
    range_apv = None
    range_distance = None
    bins = 10

    data = study.data
    distance = data['distance']
    apv = data['apv']

    # Define bin edges
    apv_bins = np.linspace(range_apv[0], range_apv[1], bins + 1) if range_apv else np.linspace(apv.min(), apv.max(), bins + 1)
    distance_bins = np.linspace(range_distance[0], range_distance[1], bins + 1) if range_distance else np.linspace(distance.min(), distance.max(), bins + 1)
    
    results = get_results(10, 8)
    create_population_behavior_heatmaps(results, apv_bins, distance_bins)
