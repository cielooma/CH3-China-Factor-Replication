"""Regression checks for paper-specific definitions and formation-time integrity."""
import numpy as np
import pandas as pd
import pytest
from ch3_fixtures import exposure_matrix
from quant_research.factors import Ch3Spec, build_ch3_factors, validate_ch3_result
from quant_research.pit import PointInTimePanel


def panel(m, f):
    return PointInTimePanel.from_frames(m, f, source="test", data_version="test_v1")


def test_valuation_denominator_is_separate_from_size_and_weights():
    m, f = exposure_matrix()
    m["valuation_me"] = m["me"] * 2
    a = build_ch3_factors(panel(m, f), Ch3Spec())
    b = build_ch3_factors(panel(m, f), Ch3Spec(valuation_cap_field="valuation_me"))
    np.testing.assert_allclose(b.assignments.ep, a.assignments.ep / 2)
    np.testing.assert_allclose(b.assignments.bm, a.assignments.bm / 2)
    pd.testing.assert_series_equal(a.assignments.me, b.assignments.me)
    pd.testing.assert_series_equal(a.assignments.size_group, b.assignments.size_group)


def test_missing_future_return_never_changes_formation_membership_or_breakpoints():
    m, f = exposure_matrix()
    before = build_ch3_factors(panel(m, f), Ch3Spec())
    t = before.factor_returns.index[0]
    sid = before.assignments.iloc[0].security_id
    m.loc[(m.security_id == sid) & (m.observation_date == t), "ret_1m"] = np.nan
    after = build_ch3_factors(panel(m, f), Ch3Spec())
    cols = ["security_id", "size_group", "value_group", "bm_group"]
    pd.testing.assert_frame_equal(before.assignments[cols], after.assignments[cols])
    assert before.diagnostics.loc[t, "breakpoints"] == after.diagnostics.loc[t, "breakpoints"]
    assert after.factor_returns.loc[t].isna().any()
    assert "assignments contain missing next-month returns" in validate_ch3_result(after)


def test_all_negative_ep_are_growth_even_above_thirty_percent():
    m, f = exposure_matrix()
    ids = sorted(m.security_id.unique())[20:50]
    f.loc[f.security_id.isin(ids), "earnings_ttm"] = -10
    result = build_ch3_factors(panel(m, f), Ch3Spec())
    assert len(result.assignments.query('ep < 0')) / len(result.assignments) > .3
    assert set(result.assignments.query('ep < 0').value_group) == {"G"}


def test_reported_earnings_are_not_silently_replaced_by_ttm():
    m, f = exposure_matrix()
    f["earnings_reported"] = f.earnings_ttm * 3
    a = build_ch3_factors(panel(m, f), Ch3Spec())
    b = build_ch3_factors(panel(m, f), Ch3Spec(earnings_field="earnings_reported"))
    np.testing.assert_allclose(b.assignments.ep, a.assignments.ep * 3)
    with pytest.raises(ValueError, match="missing earnings field"):
        build_ch3_factors(panel(m, f.drop(columns="earnings_reported")), Ch3Spec(earnings_field="earnings_reported"))


def test_financial_revision_not_visible_before_its_release():
    m, f = exposure_matrix()
    first = f.iloc[[0]].copy()
    first["available_at"] = pd.Timestamp("2001-02-15")
    first["trade_date"] = first.available_at
    first["earnings_ttm"] = 999
    p = panel(m, pd.concat([f, first], ignore_index=True))
    req = pd.DataFrame({"security_id": [first.security_id.iloc[0]] * 2,
                        "formation_date": pd.to_datetime(["2001-01-31", "2001-02-28"])})
    got = p.available_fundamentals(req)
    assert got.earnings_ttm.iloc[0] != 999
    assert got.earnings_ttm.iloc[1] == 999


def test_reference_requires_complete_sample_and_risk_free():
    m, f = exposure_matrix()
    result = build_ch3_factors(panel(m, f), Ch3Spec(require_full_sample=True))
    assert "factor months do not cover the complete configured sample" in validate_ch3_result(result)
    with pytest.raises(ValueError, match="missing required risk-free"):
        build_ch3_factors(panel(m.drop(columns="rf_1y_deposit"), f), Ch3Spec(require_risk_free=True))


def test_tied_ep_keeps_growth_and_value_disjoint():
    m, f = exposure_matrix()
    caps = m.drop_duplicates("security_id").set_index("security_id").me
    f["earnings_ttm"] = f.security_id.map(caps) * .02
    result = build_ch3_factors(panel(m, f), Ch3Spec())
    first = result.assignments.query('realization_month == @result.factor_returns.index[0]')
    assert (first.value_group == "G").sum() == int(.3 * len(first))
    assert (first.value_group == "V").sum() == int(.3 * len(first))


def test_realized_month_uses_calendar_label_with_last_trading_day_inputs():
    m, f = exposure_matrix()
    for col in ["observation_date", "available_at", "trade_date"]:
        m[col] -= pd.Timedelta(days=2)
    result = build_ch3_factors(panel(m, f), Ch3Spec(sample_start="2001-02-28", sample_end="2001-03-31", require_full_sample=True))
    assert list(result.factor_returns.index) == list(pd.to_datetime(["2001-02-28", "2001-03-31"]))
    assert result.factor_returns.formation_month.iloc[0] == pd.Timestamp("2001-01-29")
    assert not validate_ch3_result(result)
