import argparse

def get_default_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment_id", type=int)
    parser.add_argument("-pf", "--parent_folder", type=str)
    parser.add_argument("-ed", "--experiment_datetime", type=str)

    return parser

def parse_args():
    parser = get_default_parser()
    args = parser.parse_args()
    return args
