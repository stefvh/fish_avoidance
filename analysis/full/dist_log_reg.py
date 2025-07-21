import numpy as np
from core.experiment import Experiment
from core.study import study

from analysis.full.get_peak_data import get_peaks_data
from analysis.full.run_reg_fe import get_sampled_dataframe

from analysis.one.wrapper_fixed_gen import run_pglm_model
from analysis.one.wrapper_fixed import run_plm_model

if __name__ == "__main__":
    data = study.data
    sample_time_points, _ = get_peaks_data(
        func_bound=None,
        func_multiplier=1,
        acc_max_height=0,
        vel_max_height=0,
        acc_prominence=12.5,
        do_group=False,
    )
    rv_df, ra_df = get_sampled_dataframe(sample_time_points)

    # get frames from rv_df where rv_rev > 1
    frames = rv_df['frame']
    rv_rev = rv_df['rv_rev']
    frames_rv_rev = frames[rv_rev > 1]

    print(frames_rv_rev)

    lr_data = data.copy()
    lr_data['b_lr'] = 0
    lr_data.loc[lr_data['frame'].isin(frames_rv_rev), 'b_lr'] = 1

    run_pglm_model(
        dataframe=lr_data,
        response_var="b_lr",
        predictor_vars=["distance"],
        indexes=["experiment_id"],
        model_type="within",
        effect="individual",
    )