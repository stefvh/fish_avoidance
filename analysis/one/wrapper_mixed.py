import re
import numpy as np
import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri

pandas2ri.activate()

def run_lmer_model(dataframe, response_var, fixed_effects, random_effects):
    r = robjects.r
    
    # Load required R libraries
    r('library(lme4)')
    r('library(lmerTest)')
    r('library(MuMIn)')
    r('library(rptR)')

    # Convert pandas DataFrame to R DataFrame
    r_dataframe = pandas2ri.py2rpy(dataframe)
    robjects.globalenv['data'] = r_dataframe

    # Construct the formula
    formula = f"{response_var} ~ " + " + ".join(fixed_effects)
    formula += " + " + " + ".join([f"(1|{effect})" for effect in random_effects])

    # Create the plm model command
    model_cmd = f'''
    model <- lmer({formula},
        data = data,
    )
    '''

    # Execute the model command
    r(model_cmd)

    # Print summary of the model
    summary_cmd = 'summary(model)'
    summary = r(summary_cmd)

    print(summary)

    # Extract the coefficient names and values from the summary
    summary_text = r('capture.output(summary(model))')
    summary_text = "\n".join(summary_text)
    
    coef_pattern = re.compile(r'Fixed effects:\n([\s\S]+?)\n\n')
    match = coef_pattern.search(summary_text)
    coef_lines = match.group(1).strip().split("\n")
    coef_names = []
    coef_values = []
    std_errors = []
    df_values = []
    t_values = []
    p_sign_values = []

    for line in coef_lines[1:len(fixed_effects)+2]:
        print(line)
        parts = re.split(r'\s+', line)
        coef_names.append(parts[0])
        coef_values.append(float(parts[1]))
        std_errors.append(float(parts[2]))
        df_values.append(float(parts[3]))
        t_values.append(float(parts[4]))
        p_sign_values.append(parts[6])

    # Calculate R-squared
    r_squared = r('r.squaredGLMM(model)')
    r_squared = r_squared[0]

    # Calcualte repeatability with rptR
    rpt_cmd = f'''
    rpt_result <- rpt({formula}, data = data, grname = "experiment_id", datatype = "Gaussian", nboot = 100, npermut = 0)
    '''
    r(rpt_cmd)
    rpt_result = r('rpt_result')
    rpt_value = rpt_result.rx2('R')[0][0]
    rpt_ci = rpt_result.rx2('CI_emp')
    rpt_p = rpt_result.rx2('P')[0][0]
    rpt_ci = (rpt_ci[0][0], rpt_ci[1][0])

    # Convert extracted data to Python dictionary
    model_summary = {
        "coefficients": {coef_names[i]: {
            "estimate": coef_values[i],
            "std_error": std_errors[i],
            "df": df_values[i],
            "t_value": t_values[i],
            "p_sign_values": p_sign_values[i],
        } for i in range(len(coef_names))},
        "marginal_r_squared": r_squared[0],
        "conditional_r_squared": r_squared[1],
        "repeatability": rpt_value,
        "repeatability_ci": rpt_ci,
        "repeatability_p": rpt_p,
    }

    return model_summary


if __name__ == '__main__':
    # Example
    filename = "data/0_DEV/June13_2/ALL/plots/peak_moments/regression_data.csv"
    data = pd.read_csv(filename)
    response_var = "speed_fish"
    fixed_effects = ["distance", "speed_robot", "cos_apv", "cos_rev", "min_wall_distance_fish", "p_speed_fish"]
    random_effects = ["experiment_id"]

    run_lmer_model(data, response_var, fixed_effects, random_effects)

