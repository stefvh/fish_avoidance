import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
import pandas as pd

from core.experiment import Experiment
from core.study import study
from analysis.full.get_peak_data import get_peaks_data
from analysis.full.run_reg_fe import get_sampled_dataframe


pandas2ri.activate()

def _run_model(dataframe, response_var, predictor_vars, indexes, model_type, effect, family):
    pass

def run_lin_model(dataframe, response_var, predictor_vars, indexes, model_type, effect):
    pass

def run_log_model(dataframe, response_var, predictor_vars, indexes, model_type, effect):
    r = robjects.r
    
    # Load required R libraries
    r('library(base)')
    r('library(stats)')
    r('library(pglm)')
    # r('library(sandwich)')
    # r('library(lmtest)')

    # Convert pandas DataFrame to R DataFrame
    r_dataframe = pandas2ri.py2rpy(dataframe)
    robjects.globalenv['data'] = r_dataframe

    # Construct the formula
    formula = f"{response_var} ~ " + " + ".join(predictor_vars)

    # Create the plm model command
    indexes = "c(" + ", ".join([f'"{index}"' for index in indexes]) + ")"
    model_cmd = f'''
    model <- plm({formula},
        data = data,
        family = binomial(link = "logit"),
        index = {indexes},
        model = "{model_type}",
        effect = "{effect}"
    )
    '''

    # Execute the model command
    r(model_cmd)

    # Print summary of the model
    summary_cmd = 'summary(model)'
    summary = r(summary_cmd)

    print(summary)

    return summary


if __name__ == '__main__':
    # Example
    sample_time_points, _ = get_peaks_data(
        func_bound=None,
        func_multiplier=0,
        acc_max_height=0,
        vel_max_height=0,
        acc_prominence=17.5,
        do_group=False,
    )
    df, _ = get_sampled_dataframe(
        sample_time_points
    )
    # df.to_csv("test_log_rev_data.csv", index=False)

    response_var = "rv_cos_rev"
    predictor_vars = ["distance", "apv", "min_wall_distance_fish"]
    indexes = ["experiment_id"]
    model_type = "within"
    effect = "individual"

    run_log_model(df, response_var, predictor_vars, indexes, model_type, effect)

