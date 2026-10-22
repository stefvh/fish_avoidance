import pandas as pd
from analysis.full.get_peak_data import get_peaks_data
from analysis.one.time_points import get_frame_points

from core.study import study

if __name__ == '__main__':
    acc_prominence = 18.5

    frames = get_frame_points(acc_prominence=acc_prominence)
    peaks_data, _ = get_peaks_data(acc_prominence=acc_prominence)

    columns =  ['experiment_id', 'predictor_time', 'response_vel_time', 'response_acc_time']
    c_frames = frames[columns]
    c_peaks_data = peaks_data[columns]

    print(c_frames.shape)

    # check if any difference between the dataframes
    print(c_frames.equals(c_peaks_data))

    data = study.data

    data_resp_sf = pd.merge(
        data,
        frames,
        left_on=["experiment_id", "frame"],
        right_on=["experiment_id", "response_vel_time"],
    )

    sf = data_resp_sf[['experiment_id', 'predictor_time', 'speed_fish']]
    # rename peaks_data speed_max_fish column as speed_fish
    peaks_data = peaks_data.rename(columns={'max_speed_fish': 'speed_fish'})
    peaks_data = peaks_data[['experiment_id', 'predictor_time', 'speed_fish']]
    # check if any difference between the dataframes
    print(sf.equals(peaks_data))
