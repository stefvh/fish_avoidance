import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--begin", type=str)
    parser.add_argument("-e", "--end", type=str)
    return parser.parse_args()


def run(begin, end):
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
    run(args.begin, args.end)    