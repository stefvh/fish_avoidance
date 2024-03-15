import argparse

def parser_default():
    parser = argparse.ArgumentParser()
    parser.add_argument("-pf", "--parent_folder", type=str, default="June13_2")
    parser.add_argument(
        "-ed", "--experiment_datetime", type=str, default="20230613-165310"
    )

    return parser

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-pf", "--parent_folder", type=str, default="June13_2")
    parser.add_argument(
        "-ed", "--experiment_datetime", type=str, default="20230613-165310"
    )

    args = parser.parse_args()
    return args
