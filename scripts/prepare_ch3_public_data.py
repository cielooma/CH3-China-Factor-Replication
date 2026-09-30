"""Normalize the archived official MONTHLY workbooks; never compound daily spreads.

Run with a Python environment providing pandas and openpyxl. Original workbooks
are preserved, and every normalized file is recorded with its source checksum.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "author_data" / "verified_20260925"
SOURCES = {
    "CH3_factors_monthly_202608.xlsx": "https://en.mingshiim.com/static/CH3_factors_monthly_202608.xlsx",
    "Size_Value_Six_Portifolio_202608.xlsx": "https://en.mingshiim.com/static/Size_Value_Six_Portifolio_202608.xlsx",
    "paper.pdf": "https://en.mingshiim.com/static/sizevaluechina.pdf",
    "database.html": "https://en.mingshiim.com/database",
    "CH_4_fac_update_20211231.csv": "https://finance.wharton.upenn.edu/~stambaug/CH_4_fac_update_20211231.csv",
}


def read_monthly(path):
    frame = pd.read_excel(path, sheet_name="Returnseries")
    dates = pd.to_datetime(frame.pop("mnthdt").astype(str), format="%Y%m%d")
    frame.index = dates.dt.to_period("M").dt.to_timestamp("M")
    frame.index.name = "date"
    frame = frame.sort_index().apply(pd.to_numeric, errors="raise")
    if frame.index.has_duplicates or frame.isna().any().any():
        raise ValueError(f"Duplicate months or missing values in {path}")
    if not pd.period_range(frame.index[0], frame.index[-1], freq="M").equals(frame.index.to_period("M")):
        raise ValueError(f"Missing calendar months in {path}")
    return frame


def main():
    factors = read_monthly(DATA / "CH3_factors_monthly_202608.xlsx").rename(
        columns={"mktrf": "MKT", "rf_mon": "RF_MONTHLY"})
    portfolios = read_monthly(DATA / "Size_Value_Six_Portifolio_202608.xlsx")
    # Workbooks are in decimal returns. Cross-check common factors against the
    # explicitly percentage-labelled archived CH-4 file; SMB differs by design.
    old = pd.read_csv(DATA / "CH_4_fac_update_20211231.csv", skiprows=9)
    old.index = pd.to_datetime(old.pop("mnthdt").astype(str), format="%Y%m%d")
    gaps = {}
    for public, archival in [("MKT", "mktrf"), ("VMG", "VMG"), ("RF_MONTHLY", "rf_mon")]:
        joined = pd.concat([factors[public], old[archival] / 100], axis=1).dropna()
        gaps[public] = float((joined.iloc[:, 0] - joined.iloc[:, 1]).abs().max())
        if gaps[public] > 0.00011:
            raise ValueError(f"Unit/version cross-check failed for {public}: {gaps[public]}")
    factors.to_csv(DATA / "ch3_monthly_official.csv", float_format="%.10g")
    portfolios.to_csv(DATA / "size_value_portfolios.csv", float_format="%.10g")
    files = {name: {"url": url, "sha256": hashlib.sha256((DATA / name).read_bytes()).hexdigest(),
                    "bytes": (DATA / name).stat().st_size} for name, url in SOURCES.items()}
    for name in ["ch3_monthly_official.csv", "size_value_portfolios.csv"]:
        files[name] = {"sha256": hashlib.sha256((DATA / name).read_bytes()).hexdigest()}
    manifest = {"prepared_at_utc": datetime.now(timezone.utc).isoformat(),
                "download_date_local": "2026-09-25", "files": files, "unit": "decimal",
                "monthly_rows": len(factors), "span": [str(factors.index[0].date()), str(factors.index[-1].date())],
                "common_CH4_max_absolute_gap": gaps,
                "warning": "CH4 SMB is NOT CH3 SMB. Portfolio workbook has four legs plus two spreads; no middle legs. Post-2016 continuity not independently verified."}
    (DATA / "provenance.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(json.dumps({k:v for k,v in manifest.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
