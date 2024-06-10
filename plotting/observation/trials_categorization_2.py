import os
import post_process.categorization as cat
import post_process.feature_extraction as fe
from core.experiment import Experiment, get_experiment
from utils.parser import parse_args

def category_labels(approach_velocity, stop_distance, stop_location, post_behavior):
    if approach_velocity == "STEADY_APPROACH":
        if stop_distance == "CLOSE_STOP":
            if stop_location == "INTERSECT_STOP":
                if post_behavior == "NORMAL_POST":
                    return "A"
                else:
                    return "B"
            else:
                if post_behavior == "NORMAL_POST":
                    return "C"
                else:
                    return "D"
        else:
            if stop_location == "INTERSECT_STOP":
                if post_behavior == "NORMAL_POST":
                    return "1"
                else:
                    return "2"
            else:
                if post_behavior == "NORMAL_POST":
                    return "3"
                else:
                    return "E"
    else:
        if stop_distance == "CLOSE_STOP":
            if stop_location == "INTERSECT_STOP":
                if post_behavior == "NORMAL_POST":
                    return "F"
                else:
                    return "G"
            else:
                if post_behavior == "NORMAL_POST":
                    return "4"
                else:
                    return "H"
        else:
            if stop_location == "INTERSECT_STOP":
                if post_behavior == "NORMAL_POST":
                    return "5"
                else:
                    return "6"
            else:
                if post_behavior == "NORMAL_POST":
                    return "I"
                else:
                    return "J"

def get_categories():
    A = [0, 1, 5, 6, 26, 34]
    B = [4, 21, 24, 27, 28, 43]
    C = [32]
    D = [2, 9, 10, 13, 40, 41]
    E = [23, 25, 36]
    F = [33]
    G = [37]
    H = [11, 12]
    I = [3, 7, 8, 20, 29, 38, 39]
    J = [22, 35, 42]

    return (A, B, C, D, E, F, G, H, I, J)

def get_trial_manual_category(trial_id):
    categories = get_categories()
    for i, category in enumerate(categories):
        if trial_id in category:
            return str.upper(chr(65 + i))
    
    return -1
        

def run(experiment: Experiment):
    output_folder = experiment.input_folder + "/plots"
    sub_folder = "observation"
    output_filename = "trials_categorization_2.csv"
    if not os.path.exists(f"{output_folder}/{sub_folder}"):
        os.makedirs(f"{output_folder}/{sub_folder}")
    
    results = experiment.results
    trial_ids = results.trial_id.unique()

    with open(f"{output_folder}/{sub_folder}/{output_filename}", "w") as f:
        f.write("trial_id,label,flabel,approach_velocity,stop_distance,stop_location,post_behavior\n")
        for trial_id in trial_ids:
            trial_results = results[results.trial_id == trial_id]
            window_frame = fe.extract_approach_window_frame_by_follow_metric(trial_results)
            approach_velocity = cat.categorize_approach_velocity(trial_results, window_frame=window_frame)
            stop_distance = cat.categorize_stop_distance(trial_results)
            stop_location = cat.categorize_stop_location(trial_results, window_frame=window_frame)
            post_behavior = cat.categorize_post_behavior(trial_results, window_frame=window_frame)
            label = category_labels(approach_velocity, stop_distance, stop_location, post_behavior)
            flabel = get_trial_manual_category(trial_id)
            f.write(f"{trial_id},{label},{flabel},{approach_velocity},{stop_distance},{stop_location},{post_behavior}\n")

if __name__ == "__main__":
    args = parse_args()
    variables = vars(args)

    experiment = get_experiment(**variables)
    run(experiment)
