import datetime
from enum import Enum


class Type(Enum):
    STRAIGHT = 1
    HALF = 2
    SIDEWAYS = 3


def get_valid_trials():
    trials = []

    def set_trial(id, start_time, end_time, type):
        trial = {"id": id, "start_time": start_time, "end_time": end_time, "type": type, "results": None}
        trials.append(trial)

    set_trial(
        1,
        datetime.time(0, 1, 19, 500000),
        datetime.time(0, 1, 24, 00000),
        Type.STRAIGHT,
    )
    set_trial(
        2, 
        datetime.time(0, 1, 55, 000000), 
        datetime.time(0, 1, 58, 000000),  # datetime.time(0, 1, 57, 000000), 
        Type.HALF
    )
    set_trial(
        3, datetime.time(0, 2, 7, 000000), datetime.time(0, 2, 11, 750000), Type.HALF
    )
    set_trial(
        4,
        datetime.time(0, 2, 18, 000000),
        datetime.time(0, 2, 20, 500000), # datetime.time(0, 2, 20, 000000),
        Type.STRAIGHT,
    )
    set_trial(
        7,
        datetime.time(0, 2, 53, 900000),
        datetime.time(0, 2, 59, 500000),
        Type.SIDEWAYS,
    )
    set_trial(
        8, datetime.time(0, 3, 6, 500000), datetime.time(0, 3, 11, 000000), Type.HALF
    )
    set_trial(
        12,
        datetime.time(0, 4, 2, 500000),
        datetime.time(0, 4, 11, 000000),
        Type.SIDEWAYS,
    )
    set_trial(
        20,
        datetime.time(0, 5, 42, 000000),
        datetime.time(0, 5, 44, 500000),
        Type.SIDEWAYS,
    )
    set_trial(
        21,
        datetime.time(0, 5, 51, 000000),
        datetime.time(0, 5, 54, 500000),
        Type.STRAIGHT,
    )
    set_trial(
        23, datetime.time(0, 6, 24, 000000), datetime.time(0, 6, 26, 500000), Type.HALF
    )
    set_trial(
        24,
        datetime.time(0, 6, 35, 250000),
        datetime.time(0, 6, 38, 750000),
        Type.STRAIGHT,
    )
    set_trial(
        25, 
        datetime.time(0, 6, 51, 250000), 
        datetime.time(0, 6, 54, 500000),  # datetime.time(0, 6, 54, 000000), 
        Type.HALF
    )
    set_trial(
        29,
        datetime.time(0, 7, 43, 00000),
        datetime.time(0, 7, 49, 250000),
        Type.SIDEWAYS,
    )
    set_trial(
        33,
        datetime.time(0, 8, 38, 10000),
        datetime.time(0, 8, 40, 500000), # datetime.time(0, 8, 40, 000000),
        Type.SIDEWAYS,
    )
    set_trial(
        34,
        datetime.time(0, 8, 47, 00000),
        datetime.time(0, 8, 50, 500000),
        Type.STRAIGHT,
    )
    set_trial(
        35,
        datetime.time(0, 8, 56, 00000),
        datetime.time(0, 8, 59, 750000),
        Type.SIDEWAYS,
    )
    # set_trial(37, datetime.time(0,9,18,75000), datetime.time(0,9,19,500000), 'STRAIGHT')
    set_trial(
        38,
        datetime.time(0, 9, 30, 0000),
        datetime.time(0, 9, 32, 750000),
        Type.SIDEWAYS,
    )
    set_trial(
        39,
        datetime.time(0, 9, 43, 0000),
        datetime.time(0, 9, 44, 750000),
        Type.SIDEWAYS,
    )
    set_trial(
        42,
        datetime.time(0, 10, 12, 5000),
        datetime.time(0, 10, 16, 100000),
        Type.SIDEWAYS,
    )
    set_trial(
        43,
        datetime.time(0, 10, 21, 5000),
        datetime.time(0, 10, 24, 5000), # datetime.time(0, 10, 22, 50000),
        Type.STRAIGHT,
    )
    return trials

# trials = get_valid_trials()
# # export trials to csv
# import csv

# with open('post_process/labels/JUNE 13 (SET 2)/20230613-165310.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(["id", "start_time", "end_time"])
#     for trial in trials:
#         writer.writerow([trial["id"], trial["start_time"], trial["end_time"]])
