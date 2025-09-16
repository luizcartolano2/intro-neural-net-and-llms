import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import shapiro

import pandas as pd


def convert_numpy_bools(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_bools(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_bools(v) for v in obj]
    elif hasattr(obj, 'item'):  # np.bool_, np.int64, etc.
        return obj.item()
    else:
        return obj


def analyse_regression(model, df_indep):
    results = {}

    # --- resíduos ---
    waste = model.resid

    # 1.4 Normalidade dos resíduos (Shapiro-Wilk)
    stat, p = shapiro(waste)
    results['normalidade'] = {
        'statistica': stat,
        'p_valor': p,
        'normalidade_aceita': p > 0.05
    }

    # 1.5 Homocedasticidade (Breusch-Pagan)
    bp = het_breuschpagan(waste, model.model.exog)
    results['homocedasticidade'] = {
        'LM_stat': bp[0],
        'LM_p_valor': bp[1],
        'F_stat': bp[2],
        'F_p_valor': bp[3],
        'homocedastico': bp[1] > 0.05 and bp[3] > 0.05
    }

    # 1.6 Linearidade (Ramsey RESET)
    reset = linear_reset(model, power=2, use_f=True)
    results['linearidade'] = {
        'F_stat': reset.fvalue,
        'p_valor': reset.pvalue,
        'linear': reset.pvalue > 0.05
    }

    # 1.7 Autocorrelação (Durbin-Watson)
    dw = durbin_watson(waste)
    results['autocorrelacao'] = {
        'durbin_watson': dw,
        'sem_autocorrelacao': 1.5 < dw < 2.5  # regra prática
    }

    # 1.8 Multicolinearidade (VIF)
    X_com_const = sm.add_constant(df_indep)
    vif_df = pd.DataFrame()
    vif_df['variavel'] = X_com_const.columns
    vif_df['VIF'] = [variance_inflation_factor(X_com_const.values, i)
                     for i in range(X_com_const.shape[1])]
    results['multicolinearidade'] = {
        'vif': vif_df.to_dict(orient='records'),
        'sem_multicolinearidade': all(v < 5 for v in vif_df['VIF'] if v != float('inf'))
    }

    return convert_numpy_bools(results)
