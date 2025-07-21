from datetime import datetime
import os
import pandas as pd

from analysis.full.get_peak_data import get_peaks_data
from analysis.full.run_reg_fe import compute_response_distance, get_default_bound, get_iqr_bound, get_z_bound
from analysis.one.wrapper_mixed import run_lmer_model
from core.study import study

output_folder = "results/regressions"
os.makedirs(output_folder, exist_ok=True)

def get_regression_results():
    given_parameters = [
        {"func_bound": "DEFAULT", "func_multiplier": 0, "acc_max_height": 0, "vel_max_height": 0, "acc_prominence": 5},
        {"func_bound": "DEFAULT", "func_multiplier": 0, "acc_max_height": 0, "vel_max_height": 0, "acc_prominence": 10},
        {"func_bound": "DEFAULT", "func_multiplier": 0, "acc_max_height": 0, "vel_max_height": 0, "acc_prominence": 12.5},
        {"func_bound": "DEFAULT", "func_multiplier": 0, "acc_max_height": 0, "vel_max_height": 0, "acc_prominence": 15},
        {"func_bound": "DEFAULT", "func_multiplier": 0, "acc_max_height": 0, "vel_max_height": 0, "acc_prominence": 17.5},
        {"func_bound": "DEFAULT", "func_multiplier": 0, "acc_max_height": 0, "vel_max_height": 0, "acc_prominence": 18.5},
        {"func_bound": "DEFAULT", "func_multiplier": 0, "acc_max_height": 0, "vel_max_height": 0, "acc_prominence": 20},
        # {"func_bound": "IQR", "func_multiplier": 1, "acc_max_height": 20, "vel_max_height": 15, "acc_prominence": 0},
        # {"func_bound": "IQR", "func_multiplier": 1.25, "acc_max_height": 20, "vel_max_height": 15, "acc_prominence": 0},
        # {"func_bound": "IQR", "func_multiplier": 1.5, "acc_max_height": 20, "vel_max_height": 15, "acc_prominence": 0},
        # {"func_bound": "Z", "func_multiplier": 1, "acc_max_height": 20, "vel_max_height": 15, "acc_prominence": 0},
        # {"func_bound": "Z", "func_multiplier": 2, "acc_max_height": 20, "vel_max_height": 15, "acc_prominence": 0},
        # {"func_bound": "Z", "func_multiplier": 3, "acc_max_height": 20, "vel_max_height": 15, "acc_prominence": 0},
    ]
    # extend parameters
    groupings = [False]
    _parameters = []
    for parameter in given_parameters:
        for grouping in groupings:
            _parameter = parameter.copy()
            _parameter["do_group"] = grouping
            _parameters.append(_parameter)
    given_parameters = _parameters
    
    derived_parameters = []
    results = []
    for given_parameter in given_parameters:
        if given_parameter["func_bound"] == "IQR":
            func_bound = get_iqr_bound
        elif given_parameter["func_bound"] == "Z":
            func_bound = get_z_bound
        else:
            func_bound = get_default_bound
        params = given_parameter.copy()
        params["func_bound"] = func_bound
        samples, derived_parameter = get_peaks_data(
            **params
        )
        derived_parameters.append(derived_parameter)

        # Filter data to samples
        full_data = study.data.copy()
        dataframe = pd.merge(
            full_data,
            samples,
            left_on=["experiment_id", "frame"],
            right_on=["experiment_id", "predictor_time"],
        )

        # Edit data
        dataframe["min_wall_distance_fish"] = dataframe["wall_distances_fish"].apply(lambda x: x.min())

        response_distances = compute_response_distance(dataframe, full_data)
        dataframe["r_dist_fish"] = response_distances

        dataframe = dataframe.rename(columns={'speed_fish': 'p_speed_fish'})
        dataframe = dataframe.rename(columns={'response_vel_value': 'r_speed_fish'})
        dataframe = dataframe.rename(columns={'acceleration_fish': 'p_acc_fish'})
        dataframe = dataframe.rename(columns={'response_acc_value': 'r_acc_fish'})

        dataframe["d_speed_fish"] = dataframe["r_speed_fish"] - dataframe["p_speed_fish"]

        max_speed_robot = dataframe["speed_robot"].max()
        dataframe["apv"] = dataframe.apply(lambda x: (max_speed_robot - x["speed_robot"]) * x["cos_apv"] if x["cos_apv"] < 0 else x["speed_robot"] * x["cos_apv"], axis=1)
        dataframe["c_apv"] = dataframe["cos_apv"] * dataframe["speed_robot"]

        # Run model
        response_var = "r_speed_fish"
        result = run_lmer_model(
            dataframe,
            response_var=response_var,
            fixed_effects=["distance", "c_apv", "cos_rev", "min_wall_distance_fish", "p_speed_fish"],
            random_effects=["experiment_id"],
        )
        results.append(result)

    return given_parameters, derived_parameters, results, response_var

def create_me_csv(given_parameters, derived_parameters, model_summaries, response_var):
    n_models = len(given_parameters)
    n_rows_per_model = max(len(given_parameters[0]), len(derived_parameters[0]), len(model_summaries[0]))

    header = [
        "gp_name", "gp_value", 
        "dp_experiment_id", "dp_acc_height", "dp_vel_height",
        "coeff_name", "coeff_estimate", "coeff_p_value",
        "statistics_name", "statistics_value",
    ]
    rows = []
    for i in range(n_models):
        given_parameter = given_parameters[i]
        gp_names = list(given_parameter.keys())
        gp_values = list(given_parameter.values())

        derived_parameter = derived_parameters[i]
        dp_experiment_ids = derived_parameter["experiment_id"]
        dp_acc_height = derived_parameter["acc_height"]
        dp_vel_height = derived_parameter["vel_height"]

        model_summary = model_summaries[i]
        coeff_names = list(model_summary["coefficients"].keys())
        coeff_estimates = [model_summary["coefficients"][name]["estimate"] for name in coeff_names]
        coeff_p_values = [model_summary["coefficients"][name]["p_sign_values"] for name in coeff_names]
        statistics_names = ["marginal_r_squared",
                             "conditional_r_squared",
                             "repeatability", 
                             "repeatability_ci",
                             "repeatability_p"]
        statistics_values = [model_summary[name] for name in statistics_names]

        gp_names += [""] * (n_rows_per_model - len(gp_names))
        gp_values += [""] * (n_rows_per_model - len(gp_values))
        dp_experiment_ids += [""] * (n_rows_per_model - len(dp_experiment_ids))
        dp_acc_height += [""] * (n_rows_per_model - len(dp_acc_height))
        dp_vel_height += [""] * (n_rows_per_model - len(dp_vel_height))
        coeff_names += [""] * (n_rows_per_model - len(coeff_names))
        coeff_estimates += [""] * (n_rows_per_model - len(coeff_estimates))
        coeff_p_values += [""] * (n_rows_per_model - len(coeff_p_values))
        statistics_values += [""] * (n_rows_per_model - len(statistics_values))
        statistics_names += [""] * (n_rows_per_model - len(statistics_names))

        for j in range(n_rows_per_model):
            row = [
                gp_names[j], gp_values[j],
                dp_experiment_ids[j], dp_acc_height[j], dp_vel_height[j],
                coeff_names[j], coeff_estimates[j], coeff_p_values[j],
                statistics_names[j], statistics_values[j]
            ]
            rows.append(row)
        
        rows.append(["-------"] * len(header))

    df = pd.DataFrame(rows, columns=header)
    # df.to_csv(f"{output_folder}/param_reg_results.csv", index=False)
    now = datetime.now()
    filename = f"ME_{response_var}_" + now.strftime("%Y%m%d_%H%M%S") + ".csv"
    df.to_csv(f"{output_folder}/{filename}", index=False)

if __name__ == '__main__':
    gp, dp, ms, rv = get_regression_results()
    create_me_csv(gp, dp, ms, rv)