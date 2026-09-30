"""Deterministic synthetic A-share panel with a known factor structure.

Purpose: let the entire CH-3 pipeline — point-in-time validation, universe
screens, 2x3 sorts, next-month alignment, reporting — run end to end, in
public, with no licensed data.  The generator embeds *latent* MKT, SMB, VMG
and HML_BM factors into stock returns, so a correct factor construction should
recover them.  That makes the smoke test a real test instead of a tautology.

It proves the plumbing, never an investment conclusion.  Real conclusions
require the licensed panel described in
``03_reproductions/rep01_ch3/data_dictionary.md``.

The panel deliberately contains the awkward cases the screens exist for:
stocks that list mid-sample, months with too few trading days, missing
earnings before the first announcement, and a spread of market caps wide
enough that dropping the smallest 30% changes the answer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Approximate PBoC one-year deposit rate, in percent per year, used only to
# give the synthetic panel a realistic time-varying risk-free rate.  The real
# reproduction reads the rate from the licensed panel.
DEPOSIT_RATE_1Y_BY_YEAR: dict[int, float] = {
    1997: 5.67,
    1998: 3.78,
    1999: 2.25,
    2000: 2.25,
    2001: 2.25,
    2002: 1.98,
    2003: 1.98,
    2004: 2.25,
    2005: 2.25,
    2006: 2.52,
    2007: 4.14,
    2008: 3.60,
    2009: 2.25,
    2010: 2.75,
    2011: 3.50,
    2012: 3.00,
    2013: 3.00,
    2014: 2.75,
    2015: 2.00,
    2016: 1.50,
    2017: 1.50,
    2018: 1.50,
}
DEFAULT_DEPOSIT_RATE = 1.50


@dataclass(frozen=True)
class SyntheticPanelSpec:
    """Knobs of the data-generating process, all recorded in the manifest."""

    seed: int = 20260831
    n_stocks: int = 900
    start: str = "2000-01"
    end: str = "2016-12"
    data_version: str = "synthetic_ch3_v1"
    source: str = "synthetic_generator"
    warmup_months: int = 15
    formation_buffer_months: int = 1
    # Volatilities follow the authors' reported Table 3 moments (MKT 8.09%,
    # SMB 4.52%, VMG 3.75% per month), so a recovered series can be compared
    # against a known target.  The latent *means* are deliberately small: a
    # large persistent size premium compounds into market caps and slowly
    # relocates every small stock into the big group, which is a property of the
    # fixture rather than of the factor construction.  See
    # ``size_mean_reversion`` for the alternative.
    market_mean: float = 0.0060
    smb_mean: float = 0.0040
    vmg_mean: float = 0.0050
    hml_mean: float = 0.0030
    market_vol: float = 0.0810
    smb_vol: float = 0.0452
    vmg_vol: float = 0.0375
    hml_vol: float = 0.0400
    idiosyncratic_vol: float = 0.070
    beta_dispersion: float = 0.22
    size_dispersion: float = 1.20
    size_mean_reversion: float = 0.0
    size_loading: float = 0.58
    value_loading: float = 0.52
    bm_loading: float = 0.41
    leaky_fraction: float = 0.02
    leaky_trade_fraction: float = 0.01

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "n_stocks": self.n_stocks,
            "start": self.start,
            "end": self.end,
            "data_version": self.data_version,
            "source": self.source,
            "warmup_months": self.warmup_months,
            "formation_buffer_months": self.formation_buffer_months,
            "market_mean": self.market_mean,
            "smb_mean": self.smb_mean,
            "vmg_mean": self.vmg_mean,
            "hml_mean": self.hml_mean,
            "market_vol": self.market_vol,
            "smb_vol": self.smb_vol,
            "vmg_vol": self.vmg_vol,
            "hml_vol": self.hml_vol,
            "idiosyncratic_vol": self.idiosyncratic_vol,
            "beta_dispersion": self.beta_dispersion,
            "size_dispersion": self.size_dispersion,
            "size_mean_reversion": self.size_mean_reversion,
            "size_loading": self.size_loading,
            "value_loading": self.value_loading,
            "bm_loading": self.bm_loading,
            "leaky_fraction": self.leaky_fraction,
            "leaky_trade_fraction": self.leaky_trade_fraction,
        }


@dataclass
class SyntheticPanelBundle:
    market: pd.DataFrame
    fundamentals: pd.DataFrame
    author_factors: pd.DataFrame
    market_leaky: pd.DataFrame
    truth: dict[str, Any] = field(default_factory=dict)


def _month_ends(periods: pd.PeriodIndex) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(periods.to_timestamp(how="end")).normalize().astype(
        "datetime64[ns]"
    )


def _deposit_rate_percent(timestamp: pd.Timestamp) -> float:
    return DEPOSIT_RATE_1Y_BY_YEAR.get(int(timestamp.year), DEFAULT_DEPOSIT_RATE)


def simulate_ch3_panel(spec: SyntheticPanelSpec | None = None) -> SyntheticPanelBundle:
    """Simulate a monthly panel whose true factor structure is known."""

    spec = spec or SyntheticPanelSpec()
    rng = np.random.default_rng(spec.seed)

    start_period = pd.Period(spec.start, freq="M")
    end_period = pd.Period(spec.end, freq="M")
    if start_period > end_period:
        raise ValueError("start must not be later than end")
    sim_periods = pd.period_range(start_period - spec.warmup_months, end_period, freq="M")
    months = _month_ends(sim_periods)
    n_months = len(months)
    n_stocks = spec.n_stocks
    if n_stocks < 30:
        raise ValueError("n_stocks must be at least 30 for a 2x3 sort to mean anything")

    # ------------------------------------------------------------ latent truth
    deposit_rate = np.array([_deposit_rate_percent(month) for month in months])
    rf_monthly = (1.0 + deposit_rate / 100.0) ** (1.0 / 12.0) - 1.0

    latent = pd.DataFrame(
        {
            "MKT": rng.normal(spec.market_mean, spec.market_vol, n_months),
            "SMB": rng.normal(spec.smb_mean, spec.smb_vol, n_months),
            "VMG": rng.normal(spec.vmg_mean, spec.vmg_vol, n_months),
            "HML_BM": rng.normal(spec.hml_mean, spec.hml_vol, n_months),
        },
        index=months,
    )

    # ------------------------------------------------------- stock parameters
    log_me_initial = rng.normal(23.0, spec.size_dispersion, n_stocks)
    beta = np.clip(rng.normal(1.00, spec.beta_dispersion, n_stocks), 0.20, 2.00)
    ep_level = np.clip(rng.normal(0.050, 0.035, n_stocks), -0.100, 0.250)
    bm_level = np.clip(rng.normal(0.550, 0.280, n_stocks), 0.050, 1.600)

    def zscore(values: np.ndarray) -> np.ndarray:
        std = values.std(ddof=0)
        return (values - values.mean()) / (std if std > 0 else 1.0)

    z_size = zscore(log_me_initial)
    z_ep = zscore(ep_level)
    z_bm = zscore(bm_level)
    s_load = spec.size_loading * (-z_size) + rng.normal(0.0, 0.15, n_stocks)
    v_load = spec.value_loading * z_ep + rng.normal(0.0, 0.15, n_stocks)
    b_load = spec.bm_loading * z_bm + rng.normal(0.0, 0.15, n_stocks)

    # ------------------------------------------------------------- returns
    idio = rng.normal(0.0, spec.idiosyncratic_vol, size=(n_months, n_stocks))
    excess = (
        np.outer(latent["MKT"].to_numpy(), beta)
        + np.outer(latent["SMB"].to_numpy(), s_load)
        + np.outer(latent["VMG"].to_numpy(), v_load)
        + np.outer(latent["HML_BM"].to_numpy(), b_load)
        + idio
    )
    returns = excess + rf_monthly[:, None]

    # Market cap accumulates returns, but real firms also issue shares, split
    # and get delisted, so log size is mean-reverting rather than a random walk.
    # Without this pull, a positive size premium gradually moves every small
    # stock into the big group and the size sort empties out — an artefact of
    # the fixture, not of the factor construction.
    log_me = np.empty((n_months, n_stocks))
    log_me[0] = log_me_initial + np.log1p(returns[0])
    for index in range(1, n_months):
        drifted = log_me[index - 1] + np.log1p(returns[index])
        log_me[index] = drifted + spec.size_mean_reversion * (
            np.median(drifted) - drifted
        )
    me = np.exp(log_me)

    # ------------------------------------------------------ trading frictions
    # Negative offsets let stocks list *during* the sample, which is what makes
    # the listing and trading-activity screens bind.
    listing_offset = rng.integers(-12, 40, n_stocks)
    is_financial = rng.random(n_stocks) < 0.08
    st_pressure = 1.0 / (1.0 + np.exp(z_size))
    is_st_path = rng.random((n_months, n_stocks)) < (0.010 + 0.050 * st_pressure)

    security_ids = [
        f"{600000 + 7 * i:06d}.SH" if i % 2 == 0 else f"{i:06d}.SZ" for i in range(n_stocks)
    ]

    suspended_month = rng.random((n_months, n_stocks)) < 0.03
    trading_days_1m = np.where(
        suspended_month,
        rng.integers(0, 15, (n_months, n_stocks)),
        rng.integers(17, 23, (n_months, n_stocks)),
    )

    # ------------------------------------------------------- monthly market
    # One vectorised block instead of a DataFrame per month.
    panel_index = np.arange(1, n_months)
    n_panel = len(panel_index)
    listing_months = listing_offset[None, :] + panel_index[:, None] + 1
    trading_days_12m = np.minimum(244, np.maximum(listing_months, 0) * 20) - rng.integers(
        0, 6, (n_panel, n_stocks)
    )
    deposit_percent = np.array([_deposit_rate_percent(month) for month in months[1:]])

    market_all = pd.DataFrame(
        {
            "security_id": np.tile(np.array(security_ids), n_panel),
            "observation_date": np.repeat(months[1:].to_numpy(), n_stocks),
            "ret_1m": returns[1:].reshape(-1),
            "me": me[1:].reshape(-1),
            "listing_months": listing_months.reshape(-1),
            "trading_days_12m": trading_days_12m.reshape(-1),
            "trading_days_1m": trading_days_1m[1:].reshape(-1),
            "is_st": is_st_path[1:].reshape(-1),
            "is_financial": np.tile(is_financial, n_panel),
            "rf_1y_deposit": np.repeat(deposit_percent, n_stocks),
        }
    )
    market_all["trade_date"] = market_all["observation_date"]
    market_all["available_at"] = market_all["observation_date"]
    market_all["source"] = spec.source
    market_all["data_version"] = spec.data_version
    # A stock cannot trade before it lists.
    market_all = market_all.loc[market_all["listing_months"] >= 1].reset_index(drop=True)

    # ---------------------------------------------------------- fundamentals
    quarter_periods = pd.period_range(sim_periods[0], end_period, freq="Q")
    quarter_ends = _month_ends(quarter_periods)
    month_position = {timestamp: position for position, timestamp in enumerate(months)}
    quarter_positions = np.array([month_position[timestamp] for timestamp in quarter_ends])

    n_quarters = len(quarter_ends)
    ep_quarterly = np.empty((n_quarters, n_stocks))
    ep_quarterly[0] = ep_level
    for index in range(1, n_quarters):
        ep_quarterly[index] = (
            0.94 * ep_quarterly[index - 1]
            + 0.06 * ep_level
            + rng.normal(0.0, 0.006, n_stocks)
        )
    np.clip(ep_quarterly, -0.200, 0.400, out=ep_quarterly)

    book_quarterly = bm_level * (1.0 + rng.normal(0.0, 0.04, size=(n_quarters, n_stocks)))
    lag_days = rng.integers(25, 120, size=(n_quarters, n_stocks))

    day_ns = 86_400_000_000_000
    quarter_end_ns = quarter_ends.to_numpy().astype("datetime64[ns]").astype("int64")
    announcement_ns = quarter_end_ns[:, None] + lag_days.astype("int64") * day_ns

    me_at_quarter = me[quarter_positions]
    earnings_quarter = ep_quarterly * me_at_quarter / 4.0
    ttm_earnings = (
        earnings_quarter[3:]
        + earnings_quarter[2:-1]
        + earnings_quarter[1:-2]
        + earnings_quarter[:-3]
    )
    # A trailing-twelve-month figure is only knowable once the last of its four
    # quarters has been announced.
    ttm_available_ns = np.maximum.reduce(
        [
            announcement_ns[3:],
            announcement_ns[2:-1],
            announcement_ns[1:-2],
            announcement_ns[:-3],
        ]
    )
    ttm_book = book_quarterly[3:] * me_at_quarter[3:]

    n_ttm = n_quarters - 3
    fundamentals = pd.DataFrame(
        {
            "security_id": np.tile(np.array(security_ids), n_ttm),
            "observation_date": np.repeat(quarter_ends[3:].to_numpy(), n_stocks),
            "available_at": ttm_available_ns.reshape(-1).astype("datetime64[ns]"),
            "earnings_ttm": ttm_earnings.reshape(-1),
            "book_equity": ttm_book.reshape(-1),
        }
    )
    fundamentals["trade_date"] = fundamentals["available_at"]
    fundamentals["source"] = spec.source
    fundamentals["data_version"] = spec.data_version

    # ------------------------------------------------------- trim to sample
    factor_window = _month_ends(pd.period_range(start_period, end_period, freq="M"))
    sample_start, sample_end = factor_window[0], factor_window[-1]
    panel_start = _month_ends(
        pd.period_range(start_period - spec.formation_buffer_months, start_period, freq="M")
    )[0]
    market = market_all.loc[
        (market_all["observation_date"] >= panel_start)
        & (market_all["observation_date"] <= sample_end)
    ].reset_index(drop=True)

    author_factors = latent.loc[
        (latent.index >= sample_start) & (latent.index <= sample_end)
    ].copy()
    author_factors.index.name = "date"
    # The FF-3 size factor sorts on B/M rather than E/P, but it is still a size
    # factor, so in a one-size-factor world it must load on the same latent SMB.
    # Publishing it here lets the smoke test check SMB_FF as well.
    author_factors["SMB_FF"] = author_factors["SMB"]
    latent_columns = ["MKT", "SMB", "VMG", "HML_BM"]

    truth = {
        "spec": spec.to_dict(),
        "n_market_rows": int(len(market)),
        "n_fundamental_rows": int(len(fundamentals)),
        "n_securities": int(market["security_id"].nunique()),
        "panel_span": [
            str(market["observation_date"].min().date()),
            str(market["observation_date"].max().date()),
        ],
        "factor_span": [str(sample_start.date()), str(sample_end.date())],
        "latent_factor_means": {
            column: float(author_factors[column].mean()) for column in latent_columns
        },
        "latent_factor_vols": {
            column: float(author_factors[column].std(ddof=1)) for column in latent_columns
        },
        "note": (
            "Synthetic data with an embedded factor structure. Recovers the plumbing, "
            "not an investment conclusion."
        ),
    }
    return SyntheticPanelBundle(
        market=market,
        fundamentals=fundamentals,
        author_factors=author_factors,
        market_leaky=_inject_leaks(market, spec, rng),
        truth=truth,
    )


def _inject_leaks(
    market: pd.DataFrame,
    spec: SyntheticPanelSpec,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Build a deliberately leaky twin of the market table.

    Two violations, the two that matter most in practice:

    * ``available_at`` earlier than ``observation_date`` — the value is used
      before it exists (future information).
    * ``trade_date`` earlier than ``available_at`` — traded on a number that
      had not been published yet.
    """

    leaky = market.copy()
    n_rows = len(leaky)
    if n_rows == 0:
        return leaky

    n_future = max(1, int(spec.leaky_fraction * n_rows))
    n_early_trade = max(1, int(spec.leaky_trade_fraction * n_rows))
    future_rows = rng.choice(n_rows, size=min(n_future, n_rows), replace=False)
    leaky.loc[future_rows, "available_at"] = (
        leaky.loc[future_rows, "observation_date"] - pd.Timedelta(days=3)
    )

    remaining = np.setdiff1d(np.arange(n_rows), future_rows)
    if len(remaining) >= n_early_trade:
        early_rows = rng.choice(remaining, size=n_early_trade, replace=False)
        leaky.loc[early_rows, "trade_date"] = (
            leaky.loc[early_rows, "available_at"] - pd.Timedelta(days=5)
        )
    return leaky


def write_panel_bundle(bundle: SyntheticPanelBundle, output_dir: str | Path) -> dict[str, Path]:
    """Write the CSV bundle and a manifest describing how it was produced."""

    from quant_research.repro.manifest import fingerprint

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "market": directory / "ch3_monthly_market.csv",
        "fundamentals": directory / "ch3_fundamentals.csv",
        "author_factors": directory / "ch3_author_factors.csv",
        "market_leaky": directory / "ch3_monthly_market_leaky.csv",
    }
    # Six significant digits keep the files small without losing any precision
    # that matters for a market cap or a return.
    bundle.market.to_csv(paths["market"], index=False, float_format="%.6g")
    bundle.fundamentals.to_csv(paths["fundamentals"], index=False, float_format="%.6g")
    bundle.author_factors.to_csv(paths["author_factors"], float_format="%.6g")
    bundle.market_leaky.to_csv(paths["market_leaky"], index=False, float_format="%.6g")

    manifest_path = directory / "ch3_synthetic_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "generator": "quant_research.synthetic.simulate_ch3_panel",
                "spec": bundle.truth.get("spec", {}),
                "truth": bundle.truth,
                "files": {
                    name: {
                        "path": path.name,
                        "fingerprint": fingerprint(path.read_text(encoding="utf-8")),
                    }
                    for name, path in paths.items()
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    paths["manifest"] = manifest_path
    return paths
