import numpy as np
import pandas as pd

from core.experiment import Experiment
from core.study import study

from scipy.signal import find_peaks, peak_prominences


def get_peak_acc_frames(df: pd.DataFrame, height, prominence):
    frames = df["frame"].to_numpy()
    accelerations_fish = df["acceleration_fish"].to_numpy()
    peaks, _ = find_peaks(accelerations_fish, height=height, prominence=prominence)
    peak_frames = frames[peaks]

    return peaks, peak_frames


def get_peak_vel_frames(df: pd.DataFrame, peaks_acc, height):
    frames = df["frame"].to_numpy()
    speeds_fish = df["speed_fish"].to_numpy()

    peak_frames = []
    peak_valid = []
    for peak_acc in peaks_acc:
        i = peak_acc
        while i < len(speeds_fish) and speeds_fish[i] <= speeds_fish[i + 1]:
            i += 1
        peak_frames.append(frames[i])
        if i < len(speeds_fish) and speeds_fish[i] > height:
            peak_valid.append(True)
        else:
            peak_valid.append(False)

    return np.array(peak_frames), np.array(peak_valid)


def get_peak_jerk_frames(df: pd.DataFrame, peaks_acc):
    frames = df["frame"].to_numpy()
    jerks_fish = np.diff(df["acceleration_fish"].to_numpy()) / 0.04
    jerks_fish = np.append(jerks_fish, 0)

    peak_frames = []
    peak_valid = []
    for peak_acc in peaks_acc:
        i = peak_acc
        while i > 0 and jerks_fish[i] <= jerks_fish[i - 1]:
            i -= 1
        peak_frames.append(frames[i])
        if i > 0:
            peak_valid.append(True)
        else:
            peak_valid.append(False)

    return np.array(peak_frames), np.array(peak_valid)


def group_peak_vel_frames(df, peak_vel_frames, dy, dx):
    speeds_fish = df["speed_fish"].to_numpy()

    # group if (1) frames are consecutively close to each other by < dx frames and (2) the difference in speed between the frames is < dy
    groups = []
    group = []
    for i in range(1, len(peak_vel_frames)):
        if peak_vel_frames[i] - peak_vel_frames[i - 1] < dx and abs(speeds_fish[i] - speeds_fish[i - 1]) < dy:
            group.append(peak_vel_frames[i])
        else:
            groups.append(group)
            group = [peak_vel_frames[i]]

def get_frame_points(
    func_bound=None,
    func_multiplier=0,
    acc_max_height=0,
    vel_max_height=0,
    acc_prominence=0,
):
    experiment_id_points = []
    predictor_time_points = []
    response_vel_time_points = []
    response_acc_time_points = []

    acc_height_points = []
    vel_height_points = []
    acc_prominence_points = []

    for exp in study.list_experiments:
        if func_bound is not None:
            acc_func_bound = func_bound(exp.relative_motion_quantities.df["acceleration_fish"], func_multiplier)
            acc_height = min(acc_max_height, acc_func_bound)
        else:
            acc_height = acc_max_height

        if func_bound is not None:
            vel_func_bound = func_bound(exp.relative_motion_quantities.df["speed_fish"], func_multiplier)
            vel_height = min(vel_max_height, vel_func_bound)
        else:
            vel_height = vel_max_height

        segments = exp.relative_motion_quantities.df_segments
        for segment in segments:
            df = segment.copy()
            df = df.dropna()

            peaks_acc, peak_acc_frames = get_peak_acc_frames(df, acc_height, acc_prominence)
            peak_vel_frames, peak_vel_valid = get_peak_vel_frames(df, peaks_acc, vel_height)
            peak_jerk_frames, peak_jerk_valid = get_peak_jerk_frames(df, peaks_acc)

            peak_valid = np.logical_and(peak_vel_valid, peak_jerk_valid)
            # remove peaks_frames that are not valid
            peak_acc_frames = peak_acc_frames[peak_valid]
            peak_vel_frames = peak_vel_frames[peak_valid]
            peak_jerk_frames = peak_jerk_frames[peak_valid]

            if len(peak_acc_frames) > 0:
                experiment_id_points.extend([exp.id] * len(peak_acc_frames))
                predictor_time_points.extend(peak_jerk_frames)
                response_vel_time_points.extend(peak_vel_frames)
                response_acc_time_points.extend(peak_acc_frames)

                acc_height_points.extend([acc_height] * len(peak_acc_frames))
                vel_height_points.extend([vel_height] * len(peak_acc_frames))
                acc_prominence_points.extend([acc_prominence] * len(peak_acc_frames))

    results = pd.DataFrame(
        {
            "experiment_id": experiment_id_points,
            "predictor_time": predictor_time_points,
            "response_vel_time": response_vel_time_points,
            "response_acc_time": response_acc_time_points,
            "acc_height": acc_height_points,
            "vel_height": vel_height_points,
            "acc_prominence": acc_prominence_points,
        }
    )

    return results
