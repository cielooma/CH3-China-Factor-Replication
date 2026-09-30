import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from quant_research.repro.statistics import grs_test, pricing_regression


def test_single_asset_grs_matches_classical_intercept_t_squared():
    rng = np.random.default_rng(77)
    f = pd.DataFrame(rng.normal(size=(200, 2)), columns=["f1", "f2"])
    y = pd.Series(.1 + .3*f.f1 - .4*f.f2 + rng.normal(size=200), name="asset")
    fit = sm.OLS(y, sm.add_constant(f)).fit()
    result = grs_test(y.to_frame(), f)
    assert result["F"] == pytest.approx(fit.tvalues["const"]**2)
    assert result["p_value"] == pytest.approx(fit.pvalues["const"])


def test_grs_rejects_mechanically_redundant_assets():
    rng = np.random.default_rng(8)
    factors = pd.DataFrame({"value": rng.normal(size=100)})
    growth = rng.normal(size=100)
    assets = pd.DataFrame({"growth": growth, "value_leg": growth+factors.value})
    with pytest.raises(ValueError, match="singular"):
        grs_test(assets, factors)


def test_hac_uncertainty_is_larger_for_persistent_residuals():
    rng = np.random.default_rng(10)
    eps = rng.normal(size=1000)
    for t in range(1, len(eps)):
        eps[t] += .85 * eps[t-1]
    got = pricing_regression(pd.Series(eps+.4), pd.DataFrame(index=range(1000)))
    assert abs(got["t_hac"]) < abs(got["t_classical"])
    assert got["ci_hac_low_pct"] < got["alpha_pct"] < got["ci_hac_high_pct"]
