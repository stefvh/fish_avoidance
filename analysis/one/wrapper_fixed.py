import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
import pandas as pd

pandas2ri.activate()

def run_plm_model(dataframe, response_var, predictor_vars, indexes, model_type, effect):
    r = robjects.r
    
    # Load required R libraries
    r('library(plm)')
    r('library(sandwich)')
    r('library(lmtest)')

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

    # Cluster standard errors
    vcov_cluster_cmd = 'vcov_cluster <- vcovHC(model, type = "HC1", cluster = "group")'
    r(vcov_cluster_cmd)

    # Perform coeftest with clustered standard errors
    coeftest_cmd = 'coeftest_result <- coeftest(model, vcov_cluster)'
    r(coeftest_cmd)

    # Extract relevant data from coeftest
    coeftest_result = r('coeftest_result')
    coeftest_result = np.array(coeftest_result)
    coef_names = r('rownames(coeftest_result)')
    coef_names = list(coef_names)
    coef_values = coeftest_result[:, 0]
    std_errors = coeftest_result[:, 1]
    t_values = coeftest_result[:, 2]
    p_values = coeftest_result[:, 3]
    signif_values = []
    for p_value in p_values:
        str = f"{p_value:.3f}"
        if p_value < 0.001:
            str += "(***)"
        elif p_value < 0.01:
            str += "(**)"
        elif p_value < 0.05:
            str += "(*)"
        elif p_value < 0.1:
            str += "(.)"
        else:
            str += "( )"
        signif_values.append(str)

    # Extract additional model summary data
    print(summary)
    r_squared, adj_r_squared = summary.rx2('r.squared')
    f_statistic = summary.rx2('fstatistic')

    f_str = f"F = {f_statistic[1][0]}, df1 = {f_statistic[2][0]}, df2 = {f_statistic[2][1]}"
    p_value = f_statistic[3][0]

    # Calculate repeatability
    repeatability_cmd = '''
    individual_effects <- fixef(model)
    var_individual_effects <- var(individual_effects)
    residuals <- residuals(model)
    var_residuals <- var(residuals)
    total_variance <- var_individual_effects + var_residuals
    repeatability <- var_individual_effects / total_variance
    repeatability
    '''
    repeatability = r(repeatability_cmd)[0]


    # Convert extracted data to Python dictionary
    model_summary = {
        "coefficients": {coef_names[i]: {
            "estimate": coef_values[i],
            "std_error": std_errors[i],
            "t_value": t_values[i],
            "p_value": p_values[i],
            "significance": signif_values[i],
        } for i in range(len(coef_names))},
        "r_squared": r_squared,
        "adj_r_squared": adj_r_squared,
        "f_statistic": f_str,
        "p_value": p_value,
        "repeatibility": repeatability,
    }

    return model_summary


if __name__ == '__main__':
    # Example
    filename = "data/0_DEV/June13_2/ALL/plots/peak_moments/regression_data.csv"
    data = pd.read_csv(filename)
    response_var = "speed_fish"
    predictor_vars = ["distance", "speed_robot", "cos_apv", "cos_rev", "min_wall_distance_fish", "p_speed_fish"]
    indexes = ["experiment_id"]
    model_type = "within"
    effect = "individual"

    run_plm_model(data, response_var, predictor_vars, indexes, model_type, effect)

