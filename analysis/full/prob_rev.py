import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from core.experiment import Experiment
from core.study import study
from analysis.full.get_peak_data import get_peaks_data
from analysis.full.run_reg_fe import get_default_bound, get_iqr_bound, get_z_bound, get_sampled_dataframe


def compute_probabilities(dataframe):
    filefolder = f"{study.plot_output_folder}/prob_rev"
    if not os.path.exists(filefolder):
        os.makedirs(filefolder)

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

    min_samples = 30
    probabilities = np.full((n_x_bins, n_y_bins), np.nan)
    for i in range(n_x_bins):
        for j in range(n_y_bins):
            x = df_min_distance + i * dx
            y = df_min_apv + j * dy
            df_dxy = dataframe[
                (dataframe['distance'] < x) & 
                (dataframe['apv'] >= y) 
            ]
            if len(df_dxy) < min_samples:
                continue
            values = df_dxy['rv_cos_rev'].apply(lambda x: 1 if x > 0 else 0)
            # values = df_dxy['rv_speed_fish']
            probability = values.mean()
            probabilities[i,j] = probability

    fig, ax = plt.subplots()
    ax.imshow(probabilities, cmap='viridis', interpolation='nearest')

    x_labels = np.arange(df_min_distance, df_max_distance, dx)
    y_labels = np.arange(df_min_apv, df_max_apv, dy)

    apv_zero_idx = np.argmin(np.abs(y_labels))
    ax.axvline(apv_zero_idx, color='red', linestyle='dashed', linewidth=1)


    ax.set_xticks(np.arange(len(y_labels)))
    ax.set_yticks(np.arange(len(x_labels)))
    ax.set_xticklabels([f'{label:.1f}' if idx % 4 == 0 else '' for idx, label in enumerate(y_labels)])
    ax.set_yticklabels([f'{label:.1f}' if idx % 2 == 0 else '' for idx, label in enumerate(x_labels)])

    # show colorbar
    cbar = fig.colorbar(ax.imshow(probabilities, cmap='viridis', interpolation='nearest'))

    ax.set_title(r"$P(\cos({\text{rev}}) > 0 \mid \text{distance} < D, \text{apv} > A)$")
    ax.set_xlabel("APV Threshold $A")
    ax.set_ylabel("Distance Threshold $D$")
    ax.set_aspect('auto')
    plt.savefig(f"{filefolder}/probabilities.pdf")


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
    dataframe = get_sampled_dataframe(sample_time_points)

    cos_rev = dataframe['cos_rev']
    distance = dataframe['distance']
    # print(cos_rev.describe())

    # plot multivariate distribution of cos_rev and distance
    # fig, ax = plt.subplots()
    # ax.scatter(distance, cos_rev, s=1)
    # ax.scatter(distance, dataframe["cos_apv"], s=1, color="orange")
    # ax.set_title(r"$\cos(\text{rev})$ vs. distance")
    # ax.set_xlabel("Distance")
    # ax.set_ylabel(r"$\cos(\text{rev})$")
    # plt.savefig(f"{study.plot_output_folder}/prob_rev/cos_rev_distance.pdf")

    # # i want it as a 3D plot with density bins
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # hist, xedges, yedges = np.histogram2d(distance, cos_rev, bins=50, density=True)

    # xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
    # xpos = xpos.ravel()

    # ypos = ypos.ravel()
    # zpos = 0

    # dx = dy = 0.5 * np.ones_like(zpos)
    # dz = hist.ravel()

    # ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')
    # ax.set_title(r"$\cos(\text{rev})$ vs. distance")
    # ax.set_xlabel("Distance")
    # ax.set_ylabel(r"$\cos(\text{rev})$")
    # plt.savefig(f"{study.plot_output_folder}/prob_rev/cos_rev_distance_3d.pdf")

    # # plot distribution of cos_rev
    # fig, ax = plt.subplots()
    # ax.hist(dataframe['rev'], bins=100, density=True)
    # ax.set_title(r"$\cos(\text{rev})$ * speed_fish")
    # ax.set_xlabel("Value")
    # ax.set_ylabel("Density")
    # plt.savefig(f"{study.plot_output_folder}/prob_rev/cos_rev_distr.pdf")

    # t_delta = sample_time_points["response_vel_time"] - sample_time_points["predictor_time"]
    # # print(t_delta.describe())
    # mean_t_delta = t_delta.mean()
    # rv_t_delta = int(round(mean_t_delta, 0))

    # dataframe = data.copy()
    # dataframe['rv_cos_rev'] = np.nan

    # # Group by 'experiment_id'
    # grouped = dataframe.groupby('experiment_id')

    # # Iterate over each group
    # for experiment_id, group in grouped:
    #     # Sort the group by 'frame' to ensure correct shifting
    #     group = group.sort_values(by='frame').reset_index(drop=True)
        
    #     for i, row in group.iterrows():
    #         # Calculate the target frame
    #         target_frame = row['frame'] + rv_t_delta
            
    #         # Find the row with the target frame
    #         target_row = group[group['frame'] == target_frame]
            
    #         if not target_row.empty:
    #             # Set the 'rv_cos_rev' value for the current row
    #             dataframe.at[row.name, 'rv_cos_rev'] = target_row.iloc[0]['cos_rev']

    # # Drop rows where 'rv_cos_rev' is NaN
    # dataframe = dataframe.dropna(subset=['rv_cos_rev'])

    compute_probabilities(dataframe)
