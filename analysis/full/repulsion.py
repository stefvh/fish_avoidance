import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from core.experiment import Experiment
from core.study import study
from analysis.full.get_peak_data import get_peaks_data
from analysis.full.run_reg_fe import get_default_bound, get_iqr_bound, get_z_bound, get_sampled_dataframe


def plot_heatmap(z_values, x_labels, y_labels, filefolder, sub_title):
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.imshow(z_values, cmap='coolwarm', vmin=-1, vmax=1)

    # Add colorbar
    cbar = fig.colorbar(cax, ticks=[-1, -0.5, 0, 0.5, 1])
    cbar.set_label('z_values')

    # # Add annotations
    # for i in range(n_x_bins):
    #     for j in range(n_y_bins):
    #         ax.text(j, i, f'{int(z_n_samples[i, j])}\n{z_stds[i, j]:.2f}', 
    #                 ha='center', va='center', color='black')

    ax.set_xticks(np.arange(len(y_labels)))
    ax.set_yticks(np.arange(len(x_labels)))
    ax.set_xticklabels([f'{label:.1f}' if idx % 3 == 0 else '' for idx, label in enumerate(y_labels)])
    ax.set_yticklabels([f'{label:.1f}' for label in x_labels])

    ax.set_title('Heatmap')
    ax.set_xlabel('APV')
    ax.set_ylabel('Distance')

    plt.savefig(f"{filefolder}/{sub_title}.pdf")

def compute_and_plot_thresholds(z_means_total, df_min_distance, dx, df_min_apv, dy, df_max_apv, df_max_distance):
    thresholds_z = np.linspace(0, 1, 20)
    max_distance_thresholds = []
    min_apv_thresholds = []

    n_x_bins, n_y_bins = z_means_total.shape

    cond_stop_apv = False

    th_apv = 17.5
    th_distance = 7.5

    apv_bins = np.arange(n_y_bins) * dy + df_min_apv >= th_apv
    distance_bins = np.arange(n_x_bins) * dx + df_min_distance < th_distance
    selected_bins = z_means_total[distance_bins][:, apv_bins]
    a = selected_bins
    a = np.ma.array(a, mask=np.isnan(a))
    print(a)
    print(np.all(a >= 0.95))

    for threshold_z in [0.95]:
        min_apv_threshold = df_max_apv
        max_distance_threshold = df_min_distance

        for th_apv in np.linspace(df_min_apv, df_max_apv, 100):
            cond_stop_apv = False
            apv_bins = np.arange(n_y_bins) * dy + df_min_apv >= th_apv

            for th_distance in np.linspace(df_max_distance, df_min_distance, 100):
                if not cond_stop_apv:
                    distance_bins = np.arange(n_x_bins) * dx + df_min_distance < th_distance

                    # Check if all selected bins meet the threshold_z condition
                    selected_bins = z_means_total[distance_bins][:, apv_bins]
                    a = selected_bins
                    a = np.ma.array(a, mask=np.isnan(a))
                    if a.size > 0 and np.all(a >= threshold_z):
                        print(f"Threshold z: {threshold_z}")
                        print(f"Distance Threshold: {th_distance}")
                        print(f"APV Threshold: {th_apv}")
                        max_distance_threshold = th_distance
                        min_apv_threshold = th_apv
                        cond_stop_apv = True
                        break  # Break the inner loop if condition is met
            
            if cond_stop_apv:
                break

        max_distance_thresholds.append(max_distance_threshold)
        min_apv_thresholds.append(min_apv_threshold)

    # # Plotting
    # plt.figure(figsize=(12, 6))
    
    # # Plot 1: y-axis = threshold_z, x-axis = distance_threshold
    # plt.subplot(1, 2, 1)
    # plt.plot(max_distance_thresholds, thresholds_z, label='Distance Threshold')
    # plt.xlabel('Distance Threshold')
    # plt.ylabel('Threshold z')
    # plt.title('Distance Threshold vs. z Threshold')
    # plt.grid(True)
    # plt.legend()

    # # Plot 2: y-axis = threshold_z, x-axis = apv_threshold
    # plt.subplot(1, 2, 2)
    # plt.plot(min_apv_thresholds, thresholds_z, label='APV Threshold', color='orange')
    # plt.xlabel('APV Threshold')
    # plt.ylabel('Threshold z')
    # plt.title('APV Threshold vs. z Threshold')
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig("thresholds.pdf")


def plot_th_heatmap(z_means_total, df_min_distance, dx, df_min_apv, dy, df_max_apv, df_max_distance):
    n_x_bins, n_y_bins = z_means_total.shape

    distance_thresholds = np.linspace(df_min_distance, df_max_distance, 100)
    apv_thresholds = np.linspace(df_min_apv, df_max_apv, 100)

    heatmap_data = np.zeros((len(distance_thresholds), len(apv_thresholds)))

    for i, th_distance in enumerate(distance_thresholds):
        for j, th_apv in enumerate(apv_thresholds):
            distance_bins = np.arange(n_x_bins) * dx + df_min_distance < th_distance
            apv_bins = np.arange(n_y_bins) * dy + df_min_apv >= th_apv
            selected_bins = z_means_total[np.ix_(distance_bins, apv_bins)]
            if np.any(selected_bins):  # Check if there are any valid bins selected
                a = selected_bins  
                a = np.ma.array(a, mask=np.isnan(a))
                heatmap_data[i, j] = np.all(a >= 0.95)

    plt.figure(figsize=(10, 8))
    plt.imshow(heatmap_data, extent=[df_min_apv, df_max_apv, df_min_distance, df_max_distance], aspect='auto', origin='lower')
    plt.colorbar(label='Mean z_means')
    plt.xlabel('APV Threshold')
    plt.ylabel('Distance Threshold')
    plt.title('Heatmap of Mean z_means')
    
    plt.savefig("thresholds_hm.pdf")

def plot_experiment(data: pd.DataFrame, sample_time_points: pd.DataFrame):
    filefolder = f"{study.plot_output_folder}/repulsion"
    if not os.path.exists(filefolder):
        os.makedirs(filefolder)

    dataframe = get_sampled_dataframe(sample_time_points)

    df_min_distance = dataframe['distance'].min()
    df_max_distance = dataframe['distance'].max()

    df_min_apv = dataframe['apv'].min()
    df_max_apv = dataframe['apv'].max()
    
    dx = 2.5
    dy = 1.0

    n_x_bins = (df_max_distance - df_min_distance) / dx
    n_y_bins = (df_max_apv - df_min_apv) / dy
    n_x_bins = int(n_x_bins)
    n_y_bins = int(n_y_bins)

    z_means_total = np.full((n_x_bins, n_y_bins), np.nan)
    z_stds_total = np.full((n_x_bins, n_y_bins), np.nan)
    z_n_samples_total = np.zeros((n_x_bins, n_y_bins))

    experiment_ids = dataframe['experiment_id'].unique()
    
    for experiment_id in experiment_ids:
        df_experiment = dataframe[dataframe['experiment_id'] == experiment_id]

        z_means = np.full((n_x_bins, n_y_bins), np.nan)
        z_stds = np.full((n_x_bins, n_y_bins), np.nan)
        z_n_samples = np.zeros((n_x_bins, n_y_bins))

        for i in range(n_x_bins):
            for j in range(n_y_bins):
                x = df_min_distance + i * dx
                y = df_min_apv + j * dy
                df_dx_dy = df_experiment[
                    (df_experiment['distance'] >= x) & 
                    (df_experiment['distance'] < x + dx) &
                    (df_experiment['apv'] >= y) & 
                    (df_experiment['apv'] < y + dy)
                ]
                if len(df_dx_dy) == 0:
                    continue
                z_n_samples[i, j] = len(df_dx_dy)
                binary_rv_cos_rev = df_dx_dy['rv_cos_rev'].apply(lambda x: 1 if x > 0 else 0)
                z_means[i, j] = binary_rv_cos_rev.mean()
                z_stds[i, j] = binary_rv_cos_rev.std()

        # Calculate weighted averages
        total_samples = z_n_samples_total + z_n_samples
        nonzero_mask = total_samples > 0

        with np.errstate(divide='ignore', invalid='ignore'):
            weighted_means = np.divide(
                (np.nan_to_num(z_means_total) * z_n_samples_total + np.nan_to_num(z_means) * z_n_samples),
                total_samples,
                out=np.zeros_like(z_means_total),
                where=nonzero_mask
            )

        # Update totals
        z_means_total[nonzero_mask] = weighted_means[nonzero_mask]

        with np.errstate(divide='ignore', invalid='ignore'):
            z_stds_total[nonzero_mask] = np.sqrt(
                (np.nan_to_num(z_stds_total[nonzero_mask])**2 * z_n_samples_total[nonzero_mask] + 
                 np.nan_to_num(z_stds[nonzero_mask])**2 * z_n_samples[nonzero_mask]) / total_samples[nonzero_mask] +
                ((z_means_total[nonzero_mask] - np.nan_to_num(z_means[nonzero_mask]))**2 * 
                 z_n_samples_total[nonzero_mask] * z_n_samples[nonzero_mask]) / total_samples[nonzero_mask]
            )
        z_n_samples_total[nonzero_mask] = total_samples[nonzero_mask]

    x_labels = [df_min_distance + i * dx for i in range(n_x_bins)]
    y_labels = [df_min_apv + j * dy for j in range(n_y_bins)]
    # plot_heatmap(z_means_total, x_labels, y_labels, filefolder, "E=ALL_means_b_cos_rev")
    # plot_heatmap(z_stds_total, x_labels, y_labels, filefolder, "E=ALL_stds_b_cos_rev")

    # compute_and_plot_thresholds(z_means_total, df_min_distance, dx, df_min_apv, dy, df_max_apv, df_max_distance)
    plot_th_heatmap(z_means_total, df_min_distance, dx, df_min_apv, dy, df_max_apv, df_max_distance)

if __name__ == "__main__":
    data = study.data
    sample_time_points, _ = get_peaks_data(
        func_bound=None,
        func_multiplier=1,
        acc_max_height=0,
        vel_max_height=0,
        acc_prominence=15,
        do_group=False,
    )

    plot_experiment(data, sample_time_points)
