import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri

pandas2ri.activate()

def wald_test_plm(
    dataframe,
    response_var,
    predictors_restricted,   # model0 predictors (no interaction)
    predictors_full,         # model1 predictors (with interaction)
    indexes,
    model_type="within",
    effect="individual",
    vcov_type="HC1",
    cluster="group",
    method=None,             # None => plm default (Arellano)
):
    r = robjects.r
    r('library(plm)')
    r('library(lmtest)')

    # data -> R
    robjects.globalenv['data'] = pandas2ri.py2rpy(dataframe)

    # build formulas
    f0 = f"{response_var} ~ " + " + ".join(predictors_restricted)
    f1 = f"{response_var} ~ " + " + ".join(predictors_full)

    idx = "c(" + ", ".join([f'"{i}"' for i in indexes]) + ")"

    # fit both models
    r(f'''
        model0 <- plm({f0}, data=data, index={idx}, model="{model_type}", effect="{effect}")
        model1 <- plm({f1}, data=data, index={idx}, model="{model_type}", effect="{effect}")
    ''')

    # robust vcov function for waldtest
    if method is None:
        # plm::vcovHC default method is arellano (so we omit it)
        r(f'''
            vcov_fun <- function(x) vcovHC(x, type="{vcov_type}", cluster="{cluster}")
        ''')
    else:
        r(f'''
            vcov_fun <- function(x) vcovHC(x, method="{method}", type="{vcov_type}", cluster="{cluster}")
        ''')

    # Wald test of nested models (robust)
    r('wt <- waldtest(model0, model1, vcov=vcov_fun)')

    # Extract: waldtest returns an anova-like table; take 2nd row (model1 vs model0)
    # Columns typically include: Res.Df, Df, F (or Chisq), Pr(>F) ...
    wt = r('wt')
    wt_np = np.array(wt)

    # last column is p-value; the test stat is usually in the second-to-last
    p_value = float(wt_np[1, -1])
    test_stat = float(wt_np[1, -2])

    print("Wald test statistic:", test_stat)
    print("Wald p-value:", p_value)
    print(str(wt))

    # Return the printed table too (optional)
    return {
        "test_stat": test_stat,
        "p_value": p_value,
        "table_str": str(wt),
    }