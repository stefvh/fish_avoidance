import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn

from analysis.one.time_points import get_frame_points
from core.experiment import Experiment
from core.study import study

from scipy.signal import find_peaks, peak_prominences

# import utils.plotter as plotter

output_folder = study.plot_output_folder
output_folder += "/peak_moments"
os.makedirs(output_folder, exist_ok=True)


# def get_peak_acc_frames(df: pd.DataFrame, prominence=18.5):
#     frames = df["frame"].to_numpy()
#     accelerations_fish = df["acceleration_fish"].to_numpy()
#     peaks, _ = find_peaks(accelerations_fish, prominence=prominence, height=0)
#     peak_frames = frames[peaks]

#     return peaks, peak_frames


# def get_peak_vel_frames(df: pd.DataFrame, peaks_acc, window=9):
#     frames = df["frame"].to_numpy()
#     speeds_fish = df["speed_fish"].to_numpy()

#     peak_frames = []
#     for peak_acc in peaks_acc:
#         sf_window = speeds_fish[peak_acc : peak_acc + window]
#         if sf_window.shape[0] == 0:
#             continue
#         peak_frame = frames[peak_acc + sf_window.argmax()]
#         peak_frames.append(peak_frame)

#     return peak_frames


# def get_peak_jerk_frames(df: pd.DataFrame, peaks_acc):
#     frames = df["frame"].to_numpy()
#     jerks_fish = np.diff(df["acceleration_fish"].to_numpy()) / 0.04
#     jerks_fish = np.append(jerks_fish, 0)

#     peak_frames = []
#     for peak_acc in peaks_acc:
#         i = peak_acc
#         while i > 0 and jerks_fish[i] <= jerks_fish[i - 1]:
#             i -= 1
#         if i > 0:
#             peak_frames.append(frames[i])

#     return peak_frames


# def get_frame_points(prominence=18.5):
#     data = study.data
#     experiment_ids = study.experiment_ids

#     experiment_id_points = []
#     predictor_time_points = []
#     response_vel_time_points = []
#     response_acc_time_points = []

#     for i, experiment_id in enumerate(experiment_ids):
#         df = data[data["experiment_id"] == experiment_id]

#         peaks_acc, peak_acc_frames = get_peak_acc_frames(df, prominence)
#         peak_vel_frames = get_peak_vel_frames(df, peaks_acc)
#         peak_jerk_frames = get_peak_jerk_frames(df, peaks_acc)

#         experiment_id_points.extend([experiment_id] * len(peak_acc_frames))
#         predictor_time_points.extend(peak_jerk_frames)
#         response_vel_time_points.extend(peak_vel_frames)
#         response_acc_time_points.extend(peak_acc_frames)

#     results = pd.DataFrame(
#         {
#             "experiment_id": experiment_id_points,
#             "predictor_time": predictor_time_points,
#             "response_vel_time": response_vel_time_points,
#             "response_acc_time": response_acc_time_points,
#         }
#     )
#     return results


def _plot_distribution(ax, data, experiment_id, y_label, y_lim, y_color):
    if experiment_id is not None:
        data = data[data["experiment_id"] == experiment_id]
    y_max = data[y_label].max()
    y_n_samples = data[y_label].shape[0]
    if y_lim is not None:
        data = data[data[y_label].between(-y_lim, y_lim)]

    y_data = data[y_label]

    y_data.plot.hist(ax=ax, bins=20, color=y_color)
    # seaborn.kdeplot(y_data, ax=ax, color='black')
    ax.annotate(
        f"Max: {y_max:.2f}",
        xy=(0.65, 0.65),
        xycoords="axes fraction",
        ha="center",
        va="center",
    )
    ax.annotate(
        f"N: {y_n_samples}",
        xy=(0.65, 0.55),
        xycoords="axes fraction",
        ha="center",
        va="center",
    )

    if experiment_id is not None:
        ax.set_title(f"Experiment {experiment_id}")
    else:
        ax.set_title("All data")

    return ax


def _plot_distribution_row(row_axes, data, y_label, y_lim, y_color):
    ids = [None] + study.experiment_ids

    for j, experiment_id in enumerate(ids):
        _plot_distribution(row_axes[j], data, experiment_id, y_label, y_lim, y_color)

    # annotate row
    textpad = -0.45
    annotate_color = "black"
    annotation_text = y_label
    row_axes[0].annotate(
        annotation_text,
        xy=(textpad, 0.45),
        xycoords="axes fraction",
        va="center",
        rotation="vertical",
        fontsize=12,
        weight="bold",
        color=annotate_color,
    )


def plot_study_distributions():
    data = study.data
    y_labels = ["distance", "speed_fish", "acceleration_fish"]
    y_colors = ["tab:blue", "tab:orange", "tab:green"]
    y_lims = [100, 25, 50]
    fig, axes = plt.subplots(3, 6)
    fig.set_figheight(12)
    fig.set_figwidth(30)
    fig.subplots_adjust(hspace=0.5)

    for i in range(3):
        _plot_distribution_row(axes[i], data, y_labels[i], y_lims[i], y_colors[i])

    fig.savefig(f"{output_folder}/study_distributions.pdf", bbox_inches="tight")


def plot_peak_distributions():
    data_distance, data_speed_fish, data_acceleration_fish = get_predictor_response_data()

    y_labels = ["distance", "speed_fish", "acceleration_fish"]
    y_datas = [data_distance, data_speed_fish, data_acceleration_fish]
    y_colors = ["tab:blue", "tab:orange", "tab:green"]
    y_lims = [100, 25, 50]
    fig, axes = plt.subplots(3, 6)
    fig.set_figheight(12)
    fig.set_figwidth(30)
    fig.subplots_adjust(hspace=0.5)

    for i in range(3):
        _plot_distribution_row(axes[i], y_datas[i], y_labels[i], y_lims[i], y_colors[i])

    fig.savefig(f"{output_folder}/peak_distributions.pdf", bbox_inches="tight")


def _plot_scatter(ax, experiment_id, x_data, y_data, x_label, y_label, y_lim, y_color):
    if experiment_id is not None:
        x_data = x_data[x_data["experiment_id"] == experiment_id]
        y_data = y_data[y_data["experiment_id"] == experiment_id]

    x_data = x_data[x_label]
    y_data = y_data[y_label]

    ax.scatter(x_data, y_data, color=y_color, s=5)

    n_samples = x_data.shape[0]
    ax.annotate(
        f"N: {n_samples}",
        xy=(0.65, 0.65),
        xycoords="axes fraction",
        ha="center",
        va="center",
    )

    ax.set_ylim(0, y_lim)

    if experiment_id is not None:
        ax.set_title(f"Experiment {experiment_id}")
    else:
        ax.set_title("All data")

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    return ax


def _plot_scatter_row(axes_row, x_data, y_data, x_label, y_label, y_lim, y_color):
    ids = [None] + study.experiment_ids

    for j, experiment_id in enumerate(ids):
        _plot_scatter(
            axes_row[j], experiment_id, x_data, y_data, x_label, y_label, y_lim, y_color
        )

    # annotate row
    textpad = -0.45
    annotate_color = "black"
    annotation_text = y_label
    axes_row[0].annotate(
        annotation_text,
        xy=(textpad, 0.45),
        xycoords="axes fraction",
        va="center",
        rotation="vertical",
        fontsize=12,
        weight="bold",
        color=annotate_color,
    )


def plot_peak_scatters(x_label = "distance", x_data = None):
    print(f"Plotting peak scatters for {x_label}")
    data_predictors, data_speed_fish, data_acceleration_fish = get_predictor_response_data()
    if x_data is None:
        x_data = data_predictors

    y_labels = ["speed_fish", "acceleration_fish"]
    y_datas = [data_speed_fish, data_acceleration_fish]
    y_lims = [30, 75]
    y_colors = ["tab:orange", "tab:green"]

    fig, axes = plt.subplots(2, 6)
    fig.set_figheight(8)
    fig.set_figwidth(30)
    fig.subplots_adjust(hspace=0.5)

    for i in range(2):
        _plot_scatter_row(
            axes[i],
            x_data,
            y_datas[i],
            x_label,
            y_labels[i],
            y_lims[i],
            y_colors[i],
        )

    fig.savefig(f"{output_folder}/peak_scatters_{x_label}.pdf", bbox_inches="tight")


def get_predictor_response_data():
    prominence = 18.5

    frames = get_frame_points(acc_prominence=prominence)
    data = study.data

    data_predictors = pd.merge(
        data,
        frames,
        left_on=["experiment_id", "frame"],
        right_on=["experiment_id", "predictor_time"],
    )
    data_resp_sf = pd.merge(
        data,
        frames,
        left_on=["experiment_id", "frame"],
        right_on=["experiment_id", "response_vel_time"],
    )
    data_resp_af = pd.merge(
        data,
        frames,
        left_on=["experiment_id", "frame"],
        right_on=["experiment_id", "response_acc_time"],
    )

    return data_predictors, data_resp_sf, data_resp_af


def prepare_regression():
    data_pred, data_resp_speed, data_resp_acc = get_predictor_response_data()

    # data_pred['rel_speed_robot'] = data_pred['speed_robot'] / data_pred['distance']
    # data_pred['int_speed_robot'] = data_pred['speed_robot'] * data_pred['cos_apv']
    # data_pred['dist_speed_robot'] = data_pred['speed_robot'] * (80 -  data_pred['distance'])
    data_pred['min_wall_distance_fish'] = data_pred['wall_distances_fish'].apply(lambda x: x.min())

    # rename speed_fish of data_pred as p_speed_fish
    data_pred = data_pred.rename(columns={'speed_fish': 'p_speed_fish'})

    indicators = [
        'experiment_id',
    ]
    predictors = [
        'predictor_time',
        'distance',
        'speed_robot',
        'cos_apv',
        'cos_rev',
        # 'rel_speed_robot',
        # 'int_speed_robot',
        # 'dist_speed_robot',
        'min_wall_distance_fish',
        'p_speed_fish',
    ]
    data_pred = data_pred[indicators + predictors]
    data_resp_speed = data_resp_speed[['speed_fish', 'response_vel_time']]
    data_resp_acc = data_resp_acc[['acceleration_fish', 'response_acc_time']]

    # reset index for all
    data_pred.reset_index(drop=True, inplace=True)
    data_resp_speed.reset_index(drop=True, inplace=True)
    data_resp_acc.reset_index(drop=True, inplace=True)

    # for each experiment_id, make a new column that goes 1, 2, 3... (like predictor_time)
    data_pred['time'] = data_pred.groupby('experiment_id').cumcount() + 1
    # create new column p_time that is predictor_time - predictor_time.min() for each experiment_id
    data_pred['p_time'] = data_pred.groupby('experiment_id')['predictor_time'].transform(lambda x: x - x.min())

    # merge on index
    data = pd.concat([data_pred, data_resp_speed, data_resp_acc], axis=1)

    # save data
    data.to_csv(f"{output_folder}/regression_data.csv", index=False)

    return predictors, data_pred

if __name__ == "__main__":
    # plot_study_distributions()
    # plot_peak_distributions()
    predictors, data_pred = prepare_regression()
    # for predictor in predictors:
    #     plot_peak_scatters(predictor, data_pred)
