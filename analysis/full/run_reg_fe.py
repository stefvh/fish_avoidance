import os
import numpy as np
import pandas as pd

from core.experiment import Experiment
from core.study import study

from scipy.signal import find_peaks, peak_prominences
from scipy.stats import invgauss
from analysis.one.time_points import get_frame_points
from analysis.full.get_peak_data import get_peaks_data
from analysis.one.wrapper_fixed import run_plm_model

from datetime import datetime

# output_folder = study.plot_output_folder
# output_folder += "/regression_data"
output_folder = "results/regressions"
os.makedirs(output_folder, exist_ok=True)


def get_iqr_bound(values: pd.Series, multiplier: float):
    q3 = values.quantile(0.75)
    q1 = values.quantile(0.25)
    iqr = q3 - q1
    upper_bound = q3 + multiplier * iqr
    return upper_bound

def get_z_bound(values: pd.Series, multiplier: float):
    values = values.to_numpy()
    params = invgauss.fit(values)
    mu, loc, scale = params

    mean = invgauss.mean(mu, loc, scale)
    std_dev = invgauss.std(mu, loc, scale)

    upper_bound = mean + multiplier * std_dev
    return upper_bound

def get_default_bound(values: pd.Series, multiplier: float):
    return 0

def compute_response_distance(sample_data: pd.DataFrame, full_data: pd.DataFrame):
    predictor_data = sample_data.copy()
    predictor_data = predictor_data.rename(columns={"x_fish": "x_fish_predictor", "y_fish": "y_fish_predictor"})
    xy_fish_data = full_data[["experiment_id", "frame", "x_fish", "y_fish"]]
    merged_data = pd.merge(
        predictor_data,
        xy_fish_data,
        left_on=["experiment_id", "response_dst_time"],
        right_on=["experiment_id", "frame"],
        suffixes=("", "_response")
    )
    merged_data = merged_data.rename(columns={"x_fish": "x_fish_response", "y_fish": "y_fish_response"})
    merged_data["response_distance"] = np.sqrt(
        (merged_data["x_fish_predictor"] - merged_data["x_fish_response"]) ** 2 +
        (merged_data["y_fish_predictor"] - merged_data["y_fish_response"]) ** 2
    )

    return merged_data["response_distance"]

def run_model(dataframe: pd.DataFrame):
    result = run_plm_model(
        dataframe,
        response_var="speed_fish",
        predictor_vars=["distance", "speed_robot", "cos_apv", "cos_rev", "min_wall_distance_fish", "p_speed_fish"],
        indexes=["experiment_id"],
        model_type="within",
        effect="individual"
    )
    
    return result

def params_to_dict(sample_params: pd.DataFrame):
    # Initialize the dictionary
    result_dict = {}

    # Iterate over the rows of the DataFrame
    for _, row in sample_params.iterrows():
        exp_id = row['experiment_id']
        for col in sample_params.columns:
            if col != 'experiment_id':
                key = f"{col}_{exp_id}"
                result_dict[key] = row[col]

    return result_dict


def get_sampled_dataframe(samples: pd.DataFrame):
    full_data = study.data.copy()
    dataframe = pd.merge(
        full_data,
        samples,
        left_on=["experiment_id", "frame"],
        right_on=["experiment_id", "predictor_time"],
    )
    dataframe.reset_index(drop=True, inplace=True)

    rv_data = pd.merge(
        full_data,
        samples,
        left_on=["experiment_id", "frame"],
        right_on=["experiment_id", "response_vel_time"],
    )
    rv_data = rv_data[full_data.columns]
    rv_data = rv_data.rename(columns=lambda x: 'rv_' + x)
    rv_data.reset_index(drop=True, inplace=True)
    rv_dataframe = pd.concat([dataframe, rv_data], axis=1)
    
    ra_data = pd.merge(
        full_data,
        samples,
        left_on=["experiment_id", "frame"],
        right_on=["experiment_id", "response_acc_time"],
    )
    ra_data = ra_data[full_data.columns]
    ra_data = ra_data.rename(columns=lambda x: 'ra_' + x)
    dataframe.reset_index(drop=True, inplace=True)
    ra_data.reset_index(drop=True, inplace=True)
    ra_dataframe = pd.concat([dataframe, ra_data], axis=1)

    return rv_dataframe, ra_dataframe


def get_regression_results():
    given_parameters = [
        {"func_bound": "DEFAULT", "func_multiplier": 0, "acc_max_height": 0, "vel_max_height": 0, "acc_prominence": 5},
        # {"func_bound": "DEFAULT", "func_multiplier": 0, "acc_max_height": 0, "vel_max_height": 0, "acc_prominence": 10},
        # {"func_bound": "DEFAULT", "func_multiplier": 0, "acc_max_height": 0, "vel_max_height": 0, "acc_prominence": 12.5},
        # {"func_bound": "DEFAULT", "func_multiplier": 0, "acc_max_height": 0, "vel_max_height": 0, "acc_prominence": 15},
        # {"func_bound": "DEFAULT", "func_multiplier": 0, "acc_max_height": 0, "vel_max_height": 0, "acc_prominence": 17.5},
        # {"func_bound": "DEFAULT", "func_multiplier": 0, "acc_max_height": 0, "vel_max_height": 0, "acc_prominence": 18.5},
        # {"func_bound": "DEFAULT", "func_multiplier": 0, "acc_max_height": 0, "vel_max_height": 0, "acc_prominence": 20},

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

    leads = [8]
    _parameters = []
    for parameter in given_parameters:
        for lead in leads:
            _parameter = parameter.copy()
            _parameter["lead"] = lead
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
        vel_dataframe, acc_dataframe = get_sampled_dataframe(samples)

        # Edit data
        vel_dataframe["min_wall_distance_fish"] = vel_dataframe["wall_distances_fish"].apply(lambda x: x.min())

        response_distances = compute_response_distance(vel_dataframe, full_data)
        vel_dataframe["r_dist_fish"] = response_distances

        vel_dataframe = vel_dataframe.rename(columns={'speed_fish': 'p_speed_fish'})
        vel_dataframe = vel_dataframe.rename(columns={'response_vel_value': 'r_speed_fish'})
        vel_dataframe = vel_dataframe.rename(columns={'acceleration_fish': 'p_acc_fish'})
        vel_dataframe = vel_dataframe.rename(columns={'response_acc_value': 'r_acc_fish'})

        vel_dataframe["d_speed_fish"] = vel_dataframe["r_speed_fish"] - vel_dataframe["p_speed_fish"]

        max_speed_robot = vel_dataframe["speed_robot"].max()
        # dataframe["apv"] = dataframe.apply(lambda x: (max_speed_robot - x["speed_robot"]) * x["cos_apv"] if x["cos_apv"] < 0 else x["speed_robot"] * x["cos_apv"], axis=1)
        # dataframe["c_apv"] = dataframe["cos_apv"] * dataframe["speed_robot"]

        vel_dataframe["rv_rea"] = vel_dataframe["rv_ax_fish"] * vel_dataframe["rv_x_fr_n"] + vel_dataframe["rv_ay_fish"] * vel_dataframe["rv_y_fr_n"]
        acc_dataframe["ra_rea"] = acc_dataframe["ra_ax_fish"] * acc_dataframe["ra_x_fr_n"] + acc_dataframe["ra_ay_fish"] * acc_dataframe["ra_y_fr_n"]

        lead = 8 # given_parameter["lead"]
        # TEST
        # timed_df = full_data.copy()
        # # Step 1: Create a new time_sec variable
        # timed_df['time_sec'] = np.floor(timed_df['time'] / 5)
        # # Step 2: Aggregate data by experiment_id and time_sec
        # agg_df = timed_df.groupby(['experiment_id', 'time_sec']).agg(
        #     rev=('rev', 'mean'),
        #     distance=('distance', 'mean'),
        #     apv=('apv', 'mean'),
        #     min_wall_distance_fish=('min_wall_distance_fish', 'mean'),
        # ).reset_index()

        # # Step 3: Create the time_trend column within each experiment_id group
        # agg_df['time_trend'] = agg_df.groupby('experiment_id')['time_sec'].transform(
        #     lambda x: (x - x.min()) / (x.max() - x.min())
        # )
        # print(agg_df)
        # # Step 4: Shift the 'rev' column for lead
        # agg_df['rev_lead'] = agg_df.groupby('experiment_id')['rev'].shift(-lead)
        # # Run the model
        response_var = "rev"

        full_data["speed_ratio"] = full_data["speed_fish"] / full_data["speed_robot"]

        print(full_data.shape)
        # Run model
        response_var = f"lead({response_var}, {lead})"
        result = run_plm_model(
            full_data,
            response_var=response_var,
            predictor_vars=[
                "distance", 
                "apv",
                # "cos_rev", 
                "min_wall_distance_fish",
                # "p_speed_fish",
            ],
            indexes=["experiment_id"],
            model_type="within",
            effect="individual"
        )
        results.append(result)

    return given_parameters, derived_parameters, results, response_var

def create_fe_csv(given_parameters, derived_parameters, model_summaries, response_var):
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
        coeff_p_values = [model_summary["coefficients"][name]["significance"] for name in coeff_names]
        statistics_names = ["r_squared", 
                            "adj_r_squared", 
                            "f_statistic", 
                            "p_value",
                            "repeatibility"]
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
    filename = f"FE_{response_var}_" + now.strftime("%Y%m%d_%H%M%S") + ".csv"
    df.to_csv(f"{output_folder}/{filename}", index=False)

if __name__ == '__main__':
    gp, dp, ms, rv = get_regression_results()
    create_fe_csv(gp, dp, ms, rv)