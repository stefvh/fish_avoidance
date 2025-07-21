import numpy as np
import pandas as pd

from core.experiment import Experiment
from core.study import study

from scipy.signal import find_peaks, peak_prominences


def get_peaks_acc(df: pd.DataFrame, height, prominence):
    frames = df["frame"].to_numpy()
    accelerations_fish = df["acceleration_fish"].to_numpy()
    peaks, _ = find_peaks(accelerations_fish, height=height, prominence=prominence)
    peak_frames = frames[peaks]
    peak_values = accelerations_fish[peaks]

    return peaks, peak_frames, peak_values


def get_peaks_vel(df: pd.DataFrame, peaks_acc, height):
    frames = df["frame"].to_numpy()
    speeds_fish = df["speed_fish"].to_numpy()

    peak_indexes = []
    peak_frames = []
    peak_valid = []
    for peak_acc in peaks_acc:
        i = peak_acc
        while i < len(speeds_fish) and speeds_fish[i] <= speeds_fish[i + 1]:
            i += 1
        peak_indexes.append(i)
        peak_frames.append(frames[i])
        if i < len(speeds_fish) and speeds_fish[i] > height:
            peak_valid.append(True)
        else:
            peak_valid.append(False)
    
    peak_values = speeds_fish[peak_indexes]

    return np.array(peak_frames), np.array(peak_valid), peak_values


def get_peaks_jerk(df: pd.DataFrame, peaks_acc):
    frames = df["frame"].to_numpy()
    jerks_fish = np.diff(df["acceleration_fish"].to_numpy()) / 0.04
    jerks_fish = np.append(jerks_fish, 0)

    peak_indexes = []
    peak_frames = []
    peak_valid = []
    for peak_acc in peaks_acc:
        i = peak_acc
        while i > 0 and jerks_fish[i] <= jerks_fish[i - 1]:
            i -= 1
        peak_indexes.append(i)
        peak_frames.append(frames[i])
        if i > 0:
            peak_valid.append(True)
        else:
            peak_valid.append(False)
    
    peak_values = jerks_fish[peak_indexes]

    return np.array(peak_frames), np.array(peak_valid), peak_values


def get_dips_acc(df: pd.DataFrame, peaks_acc):
    frames = df["frame"].to_numpy()
    accelerations_fish = df["acceleration_fish"].to_numpy()

    peak_indexes = []
    peak_frames = []
    peak_valid = []
    for peak_acc in peaks_acc:
        i = peak_acc
        while i < len(accelerations_fish) - 1 and accelerations_fish[i] >= accelerations_fish[i + 1]:
            i += 1
        peak_indexes.append(i)
        peak_frames.append(frames[i])
        if i < len(accelerations_fish):
            peak_valid.append(True)
        else:
            peak_valid.append(False)

    return np.array(peak_frames), np.array(peak_valid)


def get_peak_groups(peak_frames, peak_values, dy, dx):
    groups = []
    group = [0]
    for i in range(1, len(peak_frames)):
        if peak_frames[i] - peak_frames[i - 1] < dx and abs(peak_values[i] - peak_values[i - 1]) < dy:
            group.append(i)
        else:
            groups.append(group)
            group = [i]
    
    return groups

def get_peaks_data(
    func_bound=None,
    func_multiplier=0,
    acc_max_height=0,
    vel_max_height=0,
    acc_prominence=0,
    do_group=False,
    lead=None,
):
    experiment_id_points = []
    predictor_time_points = []
    response_vel_time_points = []
    response_acc_time_points = []
    response_dst_time_points = []

    response_vel_values = []
    response_acc_values = []

    acc_height_points = []
    vel_height_points = []
    acc_prominence_points = []

    for exp in study.list_experiments:
        exp_df = exp.relative_motion_quantities.df
        exp_df = exp_df.dropna()
        if func_bound is not None:
            acc_func_bound = func_bound(exp_df["acceleration_fish"], func_multiplier)
            acc_height = min(acc_max_height, acc_func_bound)
        else:
            acc_height = acc_max_height

        if func_bound is not None:
            vel_func_bound = func_bound(exp_df["speed_fish"], func_multiplier)
            vel_height = min(vel_max_height, vel_func_bound)
        else:
            vel_height = vel_max_height

        speed_fish_std = exp.relative_motion_quantities.df["speed_fish"].std()

        segments = exp.relative_motion_quantities.df_segments
        for segment in segments:
            df = segment.copy()
            df = df.dropna()

            peaks_acc, peak_acc_frames, peak_acc_values = get_peaks_acc(df, acc_height, acc_prominence)
            peak_vel_frames, peak_vel_valid, peak_vel_values = get_peaks_vel(df, peaks_acc, vel_height)
            peak_jerk_frames, peak_jerk_valid, peak_jerk_values = get_peaks_jerk(df, peaks_acc)
            dip_acc_frames, dip_acc_valid = get_dips_acc(df, peaks_acc)

            peak_valid = np.logical_and(peak_vel_valid, peak_jerk_valid)
            peak_valid = np.logical_and(peak_valid, dip_acc_valid)
            # remove peaks_frames that are not valid
            peak_acc_frames = peak_acc_frames[peak_valid]
            peak_vel_frames = peak_vel_frames[peak_valid]
            peak_jerk_frames = peak_jerk_frames[peak_valid]
            dip_acc_frames = dip_acc_frames[peak_valid]
            # remove peak values that are not valid
            peak_acc_values = peak_acc_values[peak_valid]
            peak_vel_values = peak_vel_values[peak_valid]
            peak_jerk_values = peak_jerk_values[peak_valid]

            if do_group:
                groups = get_peak_groups(peak_vel_frames, peak_vel_values, speed_fish_std, 15)
                for group in groups:
                    # for each group: the lowest jerk peak is the predictor time
                    # the highest velocity peak is the response time
                    predictor_time = peak_jerk_frames[group[0]]
                    response_vel_i = np.argmax(peak_vel_values[group])
                    response_vel_value = peak_vel_values[group][response_vel_i]
                    response_vel_time = peak_vel_frames[group][response_vel_i]
                    response_acc_i = np.argmax(peak_acc_values[group])
                    response_acc_value = peak_acc_values[group][response_acc_i]
                    response_acc_time = peak_acc_frames[group][response_acc_i]
                    response_dst_time = dip_acc_frames[group[-1]]

                    experiment_id_points.append(exp.id)
                    predictor_time_points.append(predictor_time)
                    response_vel_time_points.append(response_vel_time)
                    response_acc_time_points.append(response_acc_time)
                    response_vel_values.append(response_vel_value)
                    response_acc_values.append(response_acc_value)
                    response_dst_time_points.append(response_dst_time)
            else:
                experiment_id_points.extend([exp.id] * len(peak_acc_frames))
                predictor_time_points.extend(peak_jerk_frames)
                response_vel_time_points.extend(peak_vel_frames)
                response_acc_time_points.extend(peak_acc_frames)
                response_vel_values.extend(peak_vel_values)
                response_acc_values.extend(peak_acc_values)
                response_dst_time_points.extend(dip_acc_frames)
        
        acc_height_points.append(acc_height)
        vel_height_points.append(vel_height)
        acc_prominence_points.append(acc_prominence)

    results = pd.DataFrame(
        {
            "experiment_id": experiment_id_points,
            "predictor_time": predictor_time_points,
            "response_vel_time": response_vel_time_points,
            "response_acc_time": response_acc_time_points,
            "response_vel_value": response_vel_values,
            "response_acc_value": response_acc_values,
            "response_dst_time": response_dst_time_points,
        }
    )
    parameters = {
            "experiment_id": study.experiment_ids,
            "acc_height": acc_height_points,
            "vel_height": vel_height_points,
            "acc_prominence": acc_prominence_points,
    }

    return results, parameters
