from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.experiment import Experiment

import numpy as np

from utils.parser import get_default_parser

def parse_args():
    parser = get_default_parser()
    parser.add_argument("-sf", "--begin_fn", type=int)
    parser.add_argument("-ef", "--end_fn", type=int)
    parser.add_argument("-ss", "--begin_str", type=str)
    parser.add_argument("-es", "--end_str", type=str)
    return parser.parse_args()


def suggest_poses_from_to_frame(poses: np.ndarray, begin: int, end: int, type: str):
    begin_pose = poses[begin]

    new_poses = []
    str_new_poses = []
    for frame in range(begin, end + 1):
        # interpolate the position
        x = begin_pose[0] + (poses[end][0] - begin_pose[0]) * (frame - begin) / (end - begin)
        y = begin_pose[1] + (poses[end][1] - begin_pose[1]) * (frame - begin) / (end - begin)
        theta = begin_pose[2] + (poses[end][2] - begin_pose[2]) * (frame - begin) / (end - begin)
        new_poses.append([x, y, theta])
        if type == "robot":
            str_new_poses.append(f"{frame},\"robot\",{x},{y},{theta}")
        elif type == "fish":
            str_new_poses.append(f"{frame},\"fish\",{x},{y},{theta}")
        else:
            pass
    
    return np.array(new_poses), str_new_poses

def from_frame_number(experiment: Experiment, begin: int, end: int, out_print: bool):
    new_robot_poses, str_new_robot_poses = suggest_poses_from_to_frame(experiment.robot_poses, begin, end, "robot")
    new_fish_poses, str_new_fish_poses = suggest_poses_from_to_frame(experiment.fish_poses, begin, end, "fish")
    
    if out_print:
        print("\n".join(str_new_robot_poses))
        print("\n".join(str_new_fish_poses))

    return np.array(new_robot_poses), np.array(new_fish_poses)


def from_string(begin: str, end: str):
    # begin is for example equal to the string, "6572,"fish",-10.45,-47.05,0.31"
    # separate each field
    begin_fields = begin.split(",")
    end_fields = end.split(",")

    type = begin_fields[1]

    b_frame = int(begin_fields[0])
    e_frame = int(end_fields[0])

    new_poses = []
    for frame in range(b_frame, e_frame + 1):
        # interpolate the position
        x = float(begin_fields[2]) + (float(end_fields[2]) - float(begin_fields[2])) * (frame - b_frame) / (e_frame - b_frame)
        y = float(begin_fields[3]) + (float(end_fields[3]) - float(begin_fields[3])) * (frame - b_frame) / (e_frame - b_frame)
        theta = float(begin_fields[4]) + (float(end_fields[4]) - float(begin_fields[4])) * (frame - b_frame) / (e_frame - b_frame)
        new_poses.append(f"{frame},\"{type}\",{x},{y},{theta}")
    
    print("\n".join(new_poses))

    return new_poses


if __name__ == "__main__":
    args = parse_args()
    if args.begin_fn and args.end_fn:
        from core.experiment import Experiment, get_experiment
        experiment = get_experiment(**vars(args))
        if not experiment:
            raise ValueError("Experiment not found: ", **vars(args))
        from_frame_number(experiment, args.begin_fn, args.end_fn, True)
    elif args.begin_str and args.end_str:
        from_string(args.begin_str, args.end_str)
    else:
        print("Please provide either frame numbers or strings")