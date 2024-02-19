import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--set_id", type=str, default="JUNE 13 (SET 2)")
    parser.add_argument("--experiment_id", type=str, default="20230613-165310")

    args = parser.parse_args()
    return args

def run(vars):
    folder = "data/RoboFish_Commander_Data/" + vars["set_id"] + "/" + vars["experiment_id"]
    return folder


if __name__ == "__main__":
    args = parse_args()
    variables = vars(args)