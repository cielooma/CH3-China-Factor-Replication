"""Convert explicitly specified monthly WIND-style exports to reference-v2 inputs.

This is an offline adapter, not a database client. The input return must already
include dividends and corporate actions. No missing return is filled with zero.
"""
import argparse
import json
import hashlib
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from quant_research.pit import PointInTimePanel


def convert(market, financial, *, share_multiplier, money_multiplier, return_unit):
    required_m = ["security_id", "date", "S_DQ_CLOSE", "S_SHARE_TOTALA", "TOT_SHR", "ret_1m",
                  "listing_months", "trading_days_12m", "trading_days_1m", "rf_monthly"]
    required_f = ["security_id", "REPORT_PERIOD", "ANN_DT", "NET_PROFIT_AFTER_DED_NR_LP",
                  "TOT_SHRHLDR_EQY_EXCL_MIN_INT"]
    for frame, columns in [(market, required_m), (financial, required_f)]:
        missing = sorted(set(columns)-set(frame.columns))
        if missing:
            raise ValueError(f"missing vendor fields: {missing}")
    m, f = market.copy(), financial.copy()
    for df in [m, f]:
        df["security_id"] = df.security_id.astype(str).str.strip()
    # Include only paper-universe exchange prefixes. No current-constituent list.
    m = m.loc[m.security_id.str.match(r"^(60|30|00)\d{4}(\.(SH|SZ))?$")].copy()
    f = f.loc[f.security_id.isin(m.security_id)].copy()
    m["observation_date"] = pd.to_datetime(m["date"].astype(str))
    m["available_at"] = m.observation_date
    m["trade_date"] = m.observation_date
    for c in required_m[2:]:
        m[c] = pd.to_numeric(m[c], errors="raise")
    for c in required_f[3:]:
        f[c] = pd.to_numeric(f[c], errors="raise")
    m["me"] = m.S_DQ_CLOSE * m.S_SHARE_TOTALA * share_multiplier
    m["valuation_me"] = m.S_DQ_CLOSE * m.TOT_SHR * share_multiplier
    if return_unit == "percent":
        m["ret_1m"] /= 100
    if (m.ret_1m.dropna() < -1).any():
        raise ValueError("simple stock return below -100%; check units/corporate actions")
    f["observation_date"] = pd.to_datetime(f.REPORT_PERIOD.astype(str))
    f["available_at"] = pd.to_datetime(f.ANN_DT.astype(str))
    f["trade_date"] = f.available_at
    # No undocumented fallback or TTM transformation: missing profit stays missing.
    f["earnings_reported"] = f.NET_PROFIT_AFTER_DED_NR_LP * money_multiplier
    f["book_equity"] = f.TOT_SHRHLDR_EQY_EXCL_MIN_INT * money_multiplier
    for df in [m, f]:
        df["source"] = "wind_export"
        df["data_version"] = "wind_ch3_v2"
    p = PointInTimePanel.from_frames(m, f, source="wind_export", data_version="wind_ch3_v2")
    audit = {"stock_months": len(m), "financial_vintages": len(f), "securities": m.security_id.nunique(),
             "span": [str(m.observation_date.min()), str(m.observation_date.max())],
             "missing_returns": int(m.ret_1m.isna().sum()),
             "missing_reported_profit": int(f.earnings_reported.isna().sum()),
             "share_multiplier": share_multiplier, "money_multiplier": money_multiplier,
             "input_return_unit": return_unit, "rf_unit": "monthly decimal",
             "earnings_rule": "latest reported profit, no undocumented TTM or missing-profit fallback",
             "unverified": ["historical universe completeness", "return corporate-action adjustments", "historical financial vintages and announcement timestamps"]}
    return p, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market-export", type=Path, required=True)
    parser.add_argument("--financial-export", type=Path, required=True)
    parser.add_argument("--share-multiplier", type=float, required=True, help="1 for shares, 10000 for ten-thousand shares")
    parser.add_argument("--money-multiplier", type=float, required=True)
    parser.add_argument("--return-unit", choices=["decimal", "percent"], required=True)
    parser.add_argument("--output", type=Path, default=ROOT/"private_data")
    args = parser.parse_args()
    if args.share_multiplier <= 0 or args.money_multiplier <= 0:
        parser.error("unit multipliers must be positive")
    p, audit = convert(pd.read_csv(args.market_export, dtype={"security_id": str}),
                       pd.read_csv(args.financial_export, dtype={"security_id": str}),
                       share_multiplier=args.share_multiplier, money_multiplier=args.money_multiplier,
                       return_unit=args.return_unit)
    args.output.mkdir(parents=True, exist_ok=True)
    p.market.to_csv(args.output/"ch3_monthly_market.csv", index=False)
    p.fundamentals.to_csv(args.output/"ch3_fundamentals.csv", index=False)
    audit["input_sha256"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [args.market_export, args.financial_export]}
    (args.output/"ch3_data_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False))
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
