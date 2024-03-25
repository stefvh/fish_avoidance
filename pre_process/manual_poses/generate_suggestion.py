import argparse

from pre_process.experiment import Experiment, get_experiment
from utils.parser import get_default_parser

def parse_args():
    parser = get_default_parser()
    parser.add_argument("-bf", "--begin_fn", type=int)
    parser.add_argument("-ef", "--end_fn", type=int)
    parser.add_argument("-bs", "--begin_str", type=str)
    parser.add_argument("-es", "--end_str", type=str)
    return parser.parse_args()


def from_frame_number(experiment: Experiment, begin: int, end: int):
    begin_robot_pose = experiment.robot_poses[begin]    # np.array([x, y, theta])
    end_robot_pose = experiment.robot_poses[end]        # np.array([x, y, theta])

    new_robot_poses = []
    for frame in range(begin, end + 1):
        # interpolate the position
        x = begin_robot_pose[0] + (end_robot_pose[0] - begin_robot_pose[0]) * (frame - begin) / (end - begin)
        y = begin_robot_pose[1] + (end_robot_pose[1] - begin_robot_pose[1]) * (frame - begin) / (end - begin)
        theta = begin_robot_pose[2] + (end_robot_pose[2] - begin_robot_pose[2]) * (frame - begin) / (end - begin)
        new_robot_poses.append(f"{frame},\"robot\",{x},{y},{theta}")
    
    print("\n".join(new_robot_poses))
    
    begin_fish_pose = experiment.fish_poses[begin]  # np.array([x, y, theta])
    end_fish_pose = experiment.fish_poses[end]      # np.array([x, y, theta])

    new_fish_poses = []
    for frame in range(begin, end + 1):
        # interpolate the position
        x = begin_fish_pose[0] + (end_fish_pose[0] - begin_fish_pose[0]) * (frame - begin) / (end - begin)
        y = begin_fish_pose[1] + (end_fish_pose[1] - begin_fish_pose[1]) * (frame - begin) / (end - begin)
        theta = begin_fish_pose[2] + (end_fish_pose[2] - begin_fish_pose[2]) * (frame - begin) / (end - begin)
        new_fish_poses.append(f"{frame},\"fish\",{x},{y},{theta}")

    print("\n".join(new_fish_poses))


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


if __name__ == "__main__":
    args = parse_args()
    if args.begin_fn and args.end_fn:
        experiment = get_experiment(**vars(args))
        if not experiment:
            raise ValueError("Experiment not found: ", **vars(args))
        from_frame_number(experiment, args.begin_fn, args.end_fn)
    elif args.begin_str and args.end_str:
        from_string(args.begin_str, args.end_str)
    else:
        print("Please provide either frame numbers or strings")