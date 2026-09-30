"""Time-series pricing diagnostics with explicit covariance conventions."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import f as f_distribution
import statsmodels.api as sm


def pricing_regression(y: pd.Series, factors: pd.DataFrame, hac_lags: int = 4) -> dict:
    """OLS coefficients; classical, White HC0 and NW/Bartlett HAC inference.

    HAC/HC0 p-values use asymptotic normal inference. Classical p-values use t.
    y must already be an excess return, or a zero-investment return spread.
    """
    joint = pd.concat([y.rename("__target__"), factors], axis=1).dropna()
    if not joint.index.is_unique:
        raise ValueError("regression dates must be unique")
    x = sm.add_constant(joint[factors.columns], has_constant="add").astype(float)
    if len(joint) <= len(x.columns) + hac_lags or np.linalg.matrix_rank(x) < x.shape[1]:
        raise ValueError("insufficient observations or collinear regressors")
    fit = sm.OLS(joint["__target__"].astype(float), x).fit()
    hc0 = fit.get_robustcov_results(cov_type="HC0", use_t=False)
    hac = fit.get_robustcov_results(cov_type="HAC", maxlags=hac_lags, use_correction=False, use_t=False)
    return {
        "n": int(fit.nobs), "alpha_pct": float(fit.params.iloc[0] * 100),
        "t_classical": float(fit.tvalues.iloc[0]), "p_classical": float(fit.pvalues.iloc[0]),
        "t_hc0": float(hc0.tvalues[0]), "p_hc0": float(hc0.pvalues[0]),
        "t_hac": float(hac.tvalues[0]), "p_hac": float(hac.pvalues[0]),
        "ci_hac_low_pct": float(hac.conf_int()[0, 0] * 100),
        "ci_hac_high_pct": float(hac.conf_int()[0, 1] * 100),
        "r_squared": float(fit.rsquared), "adjusted_r_squared": float(fit.rsquared_adj),
        "hac_lags": hac_lags,
        **{f"beta_{name}": float(fit.params[name]) for name in factors.columns},
    }


def grs_test(excess_returns: pd.DataFrame, factors: pd.DataFrame) -> dict:
    """Classical GRS F test, using ML covariance matrices (division by T).

    The exact F reference requires iid multivariate-normal residuals, conditional
    on factors. This is not a heteroskedasticity/autocorrelation robust test.
    Reject singular residual systems instead of silently using a pseudoinverse.
    """
    if set(excess_returns.columns) & set(factors.columns):
        raise ValueError("test assets and factor labels must be distinct")
    joined = pd.concat([excess_returns, factors], axis=1).dropna()
    y = joined[excess_returns.columns].to_numpy(float)
    z = joined[factors.columns].to_numpy(float)
    t, n = y.shape
    k = z.shape[1]
    if t <= n + k or not joined.index.is_unique:
        raise ValueError("insufficient observations or duplicate dates for GRS")
    x = np.column_stack([np.ones(t), z])
    if np.linalg.matrix_rank(x) != k + 1:
        raise ValueError("collinear factor matrix")
    b = np.linalg.lstsq(x, y, rcond=None)[0]
    e = y - x @ b
    sigma = e.T @ e / t
    centered = z - z.mean(axis=0)
    omega = centered.T @ centered / t
    if np.linalg.matrix_rank(sigma) < n or np.linalg.cond(sigma) > 1e10:
        raise ValueError("singular or ill-conditioned residual covariance; assets may span a factor")
    alpha = b[0]
    denominator = 1 + z.mean(axis=0) @ np.linalg.solve(omega, z.mean(axis=0))
    statistic = (t - n - k) / n * (alpha @ np.linalg.solve(sigma, alpha)) / denominator
    return {"F": float(statistic), "p_value": float(f_distribution.sf(statistic, n, t-n-k)),
            "n_months": t, "n_assets": n, "n_factors": k, "df_numerator": n, "df_denominator": t-n-k}
