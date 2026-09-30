from pathlib import Path
import runpy
import pandas as pd
import pytest

convert = runpy.run_path(str(Path(__file__).resolve().parents[1]/"scripts/prepare_ch3_licensed_panel.py"))["convert"]


def exports():
    m = pd.DataFrame({"security_id": ["600001.SH"], "date": ["1999-12-31"],
        "S_DQ_CLOSE": [10], "S_SHARE_TOTALA": [100], "TOT_SHR": [150], "ret_1m": [5],
        "listing_months": [30], "trading_days_12m": [200], "trading_days_1m": [20], "rf_monthly": [.002]})
    f = pd.DataFrame({"security_id": ["600001.SH"], "REPORT_PERIOD": ["1999-06-30"],
        "ANN_DT": ["1999-08-20"], "NET_PROFIT_AFTER_DED_NR_LP": [20],
        "TOT_SHRHLDR_EQY_EXCL_MIN_INT": [500]})
    return m, f


def test_explicit_unit_conversion_preserves_two_share_denominators():
    m, f = exports()
    p, audit = convert(m, f, share_multiplier=10000, money_multiplier=10000, return_unit="percent")
    assert p.market.me.iloc[0] == 10000000
    assert p.market.valuation_me.iloc[0] == 15000000
    assert p.market.ret_1m.iloc[0] == .05
    assert p.market.rf_monthly.iloc[0] == .002
    assert p.fundamentals.earnings_reported.iloc[0] == 200000


def test_adapter_does_not_invent_unavailable_profit():
    m, f = exports()
    f["NET_PROFIT_AFTER_DED_NR_LP"] = float("nan")
    p, audit = convert(m, f, share_multiplier=1, money_multiplier=1, return_unit="decimal")
    assert pd.isna(p.fundamentals.earnings_reported.iloc[0])
    assert audit["missing_reported_profit"] == 1


def test_adapter_rejects_announcement_before_report_end():
    m, f = exports()
    f["ANN_DT"] = "1999-05-30"
    with pytest.raises(ValueError):
        convert(m, f, share_multiplier=1, money_multiplier=1, return_unit="percent")
