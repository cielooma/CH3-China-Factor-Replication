"""CH-3: the Liu–Stambaugh–Yuan (2019) three-factor model for Chinese A-shares.

The model is a Fama–French-style 2x3 construction with two China-specific
choices that are the whole point of the paper:

1. **The smallest 30% of stocks are dropped.**  In China a near-empty shell
   company keeps a valuable listing permit, so the smallest firms carry a
   reverse-merger option rather than a size premium.  83% of reverse mergers
   involve shells from the smallest 30%, and roughly 30% of a typical shell's
   market value is the permit itself.  Leaving them in corrupts every factor.
2. **Value is measured by earnings-to-price (E/P), not book-to-market.**
   Book-to-market is a weak value signal in China; E/P sorts better.  The
   engine also builds the full B/M (FF-3) counterpart on the same universe so
   the two can be compared directly, which is the control exercise in §5.3.

Every construction choice below is traceable to a sentence in the paper; the
ones the paper leaves open are marked ``[OPEN]`` and default to the literal
reading.  See ``03_reproductions/rep01_ch3/preregistration.md``.

Timing, stated once and enforced by construction: every input is read from the
point-in-time panel at formation month ``t``, portfolios are formed at the end
of ``t``, and the return attributed to a factor observation is realised over
month ``t+1``.  A factor row is therefore *dated by the month it earns*, not by
the month it was formed, and the formation month is kept alongside it.  Nothing
here reads a full-sample statistic: breakpoints are recomputed inside each
cross-section, so a number produced for month ``t`` cannot depend on any
observation after ``t``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from quant_research.pit.panel import PointInTimePanel

SIZE_GROUPS = ("S", "B")
VALUE_GROUPS = ("V", "N", "G")
BM_GROUPS = ("H", "M", "L")
CH3_PORTFOLIOS = ("SV", "SN", "SG", "BV", "BN", "BG")
#: The B/M control legs.  The FF_ prefix is part of the label, so a filter for
#: "SM" can only ever return the CH-3 small/middle cell.
FF_PORTFOLIOS = ("FF_SH", "FF_SM", "FF_SL", "FF_BH", "FF_BM", "FF_BL")
FACTOR_COLUMNS = ("MKT", "SMB", "VMG", "HML_BM", "SMB_FF")

# Appendix A.1 lists the screens; the NBER working paper adds a portfolio-width
# requirement of "at least 50 stocks in all portfolios".
EXCLUSION_ORDER = (
    "invalid_market_cap",
    "invalid_valuation_cap",
    "missing_next_return",
    "insufficient_listing_history",
    "insufficient_trading_history",
    "suspended_or_st",
    "financial_firm",
    "missing_earnings",
    "non_positive_earnings",
    "smallest_30pct",
)


@dataclass(frozen=True)
class Ch3Spec:
    """The frozen CH-3 recipe.  Every field is part of the research record.

    Defaults reproduce Liu–Stambaugh–Yuan (2019) as closely as the paper
    states it.  Fields the paper does *not* pin down (``exclude_st``,
    ``breakpoint_weighting``, ``market_cap_lag_months``) default to the value
    that keeps the recipe literal and are marked ``[OPEN]``: they are reported
    as robustness variants, never tuned on the sample.
    """

    experiment_id: str = "EXP-CH3-001"
    strategy_name: str = "ch3_reproduction_v1"
    paper: str = "Liu, Stambaugh, Yuan (2019), Size and value in China, JFE 134(1):48-69"
    data_version: str = "synthetic_ch3_v1"
    sample_start: str = "2000-01-31"
    sample_end: str = "2016-12-31"
    min_listing_months: int = 6
    min_trading_days_12m: int = 120
    min_trading_days_1m: int = 15
    min_stocks_per_portfolio: int = 50
    exclude_smallest_fraction: float = 0.30
    exclude_smallest_by: str = "count"
    size_cut_first: bool = False  # [OPEN] the paper does not state the screen order
    market_cap_lag_months: int = 0  # [OPEN] "end of the previous month"
    exclude_st: bool = False  # [OPEN] the paper never mentions ST
    exclude_financials: bool = False  # the paper includes financial firms
    size_breakpoint: str = "median"
    breakpoint_weighting: str = "equal"  # [OPEN] paper does not state it
    value_signal: str = "ep"
    earnings_field: str = "earnings_ttm"  # legacy synthetic schema; reference v2 uses reported profit
    valuation_cap_field: str = "me"  # reference v2 explicitly requires price * ALL share classes
    value_breakpoints: tuple[float, float] = (0.30, 0.70)
    require_positive_earnings: bool = False  # negative E/P stocks stay, as growth
    negative_ep_policy: str = "force_growth"
    book_to_market_control: bool = True
    require_positive_book_equity: bool = True  # FF convention
    market_universe: str = "eligible"
    weighting: str = "value_weighted"
    risk_free_field: str = "rf_1y_deposit"
    risk_free_is_monthly_decimal: bool = False
    require_risk_free: bool = False
    require_full_sample: bool = False
    risk_free_annual: float = 0.0
    seed: int = 20260831

    def __post_init__(self) -> None:
        object.__setattr__(self, "value_breakpoints", tuple(self.value_breakpoints))

    def validate(self) -> None:
        if pd.Timestamp(self.sample_start) > pd.Timestamp(self.sample_end):
            raise ValueError("sample_start must not be later than sample_end")
        if not 0.0 <= self.exclude_smallest_fraction < 1.0:
            raise ValueError("exclude_smallest_fraction must be in [0, 1)")
        if self.exclude_smallest_by not in {"count", "market_cap"}:
            raise ValueError("exclude_smallest_by must be 'count' or 'market_cap'")
        if self.size_breakpoint not in {"median"}:
            raise ValueError("size_breakpoint must be 'median'")
        if self.breakpoint_weighting not in {"equal", "value"}:
            raise ValueError("breakpoint_weighting must be 'equal' or 'value'")
        if self.value_signal not in {"ep"}:
            raise ValueError("value_signal must be 'ep'")
        if self.negative_ep_policy not in {"force_growth", "rank_only"}:
            raise ValueError("negative_ep_policy must be force_growth or rank_only")
        low, high = self.value_breakpoints
        if not 0.0 < low < high < 1.0:
            raise ValueError("value_breakpoints must satisfy 0 < low < high < 1")
        if self.market_universe not in {"all", "eligible"}:
            raise ValueError("market_universe must be 'all' or 'eligible'")
        if self.weighting not in {"value_weighted", "equal_weighted"}:
            raise ValueError("weighting must be 'value_weighted' or 'equal_weighted'")
        for field_name in (
            "min_listing_months",
            "min_trading_days_12m",
            "min_trading_days_1m",
            "min_stocks_per_portfolio",
            "market_cap_lag_months",
        ):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} cannot be negative")
        if self.risk_free_annual <= -1.0:
            raise ValueError("risk_free_annual must be greater than -100%")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["value_breakpoints"] = list(self.value_breakpoints)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Ch3Spec":
        known = {item.name for item in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        unknown = sorted(set(payload) - known)
        if unknown:
            raise ValueError(
                f"unknown CH-3 config keys: {unknown}. Frozen configs must be explicit; "
                "a typo silently changing the recipe is worse than a crash."
            )
        return cls(**dict(payload))

    def fingerprint(self) -> str:
        from quant_research.repro.manifest import fingerprint

        return fingerprint(self.to_dict())

    def monthly_risk_free(self) -> float:
        return float((1.0 + self.risk_free_annual) ** (1.0 / 12.0) - 1.0)


@dataclass
class Ch3Result:
    factor_returns: pd.DataFrame
    portfolio_returns: pd.DataFrame
    assignments: pd.DataFrame
    universe_log: pd.DataFrame
    diagnostics: pd.DataFrame
    spec: Ch3Spec
    notes: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        frame = self.factor_returns
        summary: dict[str, Any] = {
            "experiment_id": self.spec.experiment_id,
            "n_months": int(len(frame)),
            "date_range": [str(frame.index.min().date()), str(frame.index.max().date())]
            if len(frame)
            else [],
            "config_fingerprint": self.spec.fingerprint(),
            "notes": list(self.notes),
            "factors": {},
        }
        for column in frame.columns:
            if column not in FACTOR_COLUMNS:
                continue
            series = frame[column].dropna()
            if series.empty:
                summary["factors"][column] = {"observations": 0}
                continue
            standard_deviation = float(series.std(ddof=1)) if len(series) > 1 else 0.0
            summary["factors"][column] = {
                "observations": int(len(series)),
                "mean_monthly": float(series.mean()),
                "std_monthly": standard_deviation,
                "annualised_mean": float(series.mean() * 12.0),
                "t_stat": float(series.mean() / standard_deviation * np.sqrt(len(series)))
                if standard_deviation > 0
                else None,
                "min": float(series.min()),
                "max": float(series.max()),
            }

        correlations = frame.loc[:, [c for c in FACTOR_COLUMNS if c in frame.columns]].corr()
        summary["correlations"] = {
            left: {right: float(correlations.loc[left, right]) for right in correlations.columns}
            for left in correlations.index
        }
        return summary


# --------------------------------------------------------------------- helpers
def _ordered_by(frame: pd.DataFrame, column: str, *, ascending: bool) -> list:
    """Deterministic ordering by a sort variable, ties broken by security id.

    Ties matter in real A-share data: suspended stocks carry a stale price
    forward, so month-end market caps repeat.  Ordering by ``security_id`` as
    the tie-break makes the split reproducible instead of dependent on the
    order the vendor serialised its file in.
    """

    ordered = frame.sort_values(
        [column, "security_id"], ascending=[ascending, True], kind="mergesort"
    )
    return list(ordered.index)


def _take_low_group(
    frame: pd.DataFrame,
    order: Sequence,
    fraction: float,
    weighting: str,
) -> list:
    """The lowest ``fraction`` of a deterministically ordered cross-section.

    With ``weighting='equal'`` the split is by count; with ``'value'`` it is by
    cumulative market-capitalisation share.  Both are computed positionally, so
    a tied block is never pushed wholesale to one side — which is what a
    ``value <= quantile`` rule does when many stocks share a market cap.
    """

    n = len(order)
    if n == 0 or fraction <= 0:
        return []
    weights = frame.loc[list(order), "me"].to_numpy(dtype="float64")
    if weighting == "value" and weights.sum() > 0:
        cumulative = np.cumsum(weights) / weights.sum()
        size = int(np.searchsorted(cumulative, fraction, side="right"))
        if size == 0:
            size = 1
    else:
        size = int(np.floor(fraction * n))
    return list(order[:size])


def _prepare_market(market: pd.DataFrame, market_cap_lag_months: int = 0) -> pd.DataFrame:
    """Attach the month grid, the next-month realised return and the size variable."""

    frame = market.copy()
    frame = frame.sort_values(["security_id", "observation_date"], kind="mergesort")
    month = frame["observation_date"].dt
    frame["month_ordinal"] = month.year * 12 + month.month

    duplicated = frame.duplicated(subset=["security_id", "month_ordinal"])
    if duplicated.any():
        example = frame.loc[duplicated, ["security_id", "observation_date"]].iloc[0]
        raise ValueError(
            "market table must hold at most one row per security and month; "
            f"found more than one for {example['security_id']} in "
            f"{example['observation_date'].date()}"
        )

    grouped = frame.groupby("security_id", sort=False)
    frame["next_month_ordinal"] = grouped["month_ordinal"].shift(-1)
    frame["realization_date"] = grouped["observation_date"].shift(-1)
    frame["ret_next_1m"] = grouped["ret_1m"].shift(-1)
    consecutive = frame["next_month_ordinal"].eq(frame["month_ordinal"] + 1)
    frame.loc[~consecutive, ["realization_date", "ret_next_1m"]] = np.nan

    # The paper excludes and splits on "market capitalization at the end of the
    # previous month".  The formation month IS the previous month relative to
    # the month the return is earned, so lag 0 is the literal reading; lag 1
    # (the month before the sort) is the preregistered robustness variant.
    # The lagged observation must be exactly `lag` months earlier: without that
    # guard a stock with a data gap silently gets a stale market cap.
    if market_cap_lag_months > 0:
        shifted_me = grouped["me"].shift(market_cap_lag_months)
        shifted_ordinal = grouped["month_ordinal"].shift(market_cap_lag_months)
        stale = shifted_ordinal.ne(frame["month_ordinal"] - market_cap_lag_months)
        frame["size_me"] = shifted_me.mask(stale)
    else:
        frame["size_me"] = frame["me"]
    return frame.reset_index(drop=True)


def _apply_size_cut(
    frame: pd.DataFrame,
    reason: pd.Series,
    spec: Ch3Spec,
    *,
    require_earnings: bool,
) -> None:
    """Drop the smallest ``exclude_smallest_fraction`` of the eligible names."""

    if spec.exclude_smallest_fraction <= 0:
        return
    candidates = frame.loc[reason == ""]
    if require_earnings:
        candidates = candidates.loc[candidates["ep"].notna()]
    if candidates.empty:
        return

    if spec.exclude_smallest_by == "count":
        order = _ordered_by(candidates, "size_me", ascending=True)
        n_drop = int(np.floor(spec.exclude_smallest_fraction * len(order)))
        dropped = order[:n_drop]
    else:
        # "30% of capitalisation" reads as the smallest names that together
        # account for 30% of the cross-section's market value.
        order = _ordered_by(candidates, "size_me", ascending=True)
        weights = candidates.loc[order, "size_me"].to_numpy(dtype="float64")
        if weights.sum() <= 0:
            return
        cumulative = np.cumsum(weights) / weights.sum()
        n_drop = int(np.searchsorted(cumulative, spec.exclude_smallest_fraction, side="right"))
        dropped = order[:n_drop]
    reason.loc[list(dropped)] = "smallest_30pct"


def _screen_cross_section(
    cross: pd.DataFrame,
    spec: Ch3Spec,
    *,
    has_listing: bool,
    has_trading_days: bool,
    has_st: bool,
    has_financial: bool,
) -> pd.DataFrame:
    """Attach a single, deterministic exclusion reason to every stock-month.

    The order below is the frozen screen order from Appendix A.1.  It matters:
    a stock can fail several screens, and the reason recorded decides which
    data problem gets investigated first.  The paper's screens come first; the
    optional robustness screens come last.
    """

    frame = cross.copy()
    reason = pd.Series("", index=frame.index, dtype="object")

    def flag(mask: pd.Series, label: str) -> None:
        reason.loc[(reason == "") & mask.fillna(False)] = label

    flag(frame["size_me"].isna() | (frame["size_me"] <= 0), "invalid_market_cap")
    # Formation membership must not depend on next month's data availability.
    # Missing realised returns invalidate affected portfolios, not their sorts.
    flag(~np.isfinite(frame["valuation_me"]) | (frame["valuation_me"] <= 0), "invalid_valuation_cap")

    # The paper's screen order is not stated.  The default applies the data
    # quality screens first and the shell-value cut last, so "30% of the
    # investable universe" is cut; `size_cut_first` cuts 30% of the raw
    # tradable universe instead.  Both are preregistered; neither is tuned.
    if spec.size_cut_first:
        _apply_size_cut(frame, reason, spec, require_earnings=False)

    if spec.min_listing_months > 0:
        if not has_listing:
            raise ValueError(
                "spec.min_listing_months requires a 'listing_months' column in the market table"
            )
        flag(frame["listing_months"] < spec.min_listing_months, "insufficient_listing_history")
    if spec.min_trading_days_12m > 0 or spec.min_trading_days_1m > 0:
        if not has_trading_days:
            raise ValueError(
                "the trading-activity screen requires 'trading_days_12m' and "
                "'trading_days_1m' columns in the market table"
            )
        flag(
            (frame["trading_days_12m"] < spec.min_trading_days_12m)
            | (frame["trading_days_1m"] < spec.min_trading_days_1m),
            "insufficient_trading_history",
        )
    if spec.exclude_st:
        if not has_st:
            raise ValueError("spec.exclude_st requires an 'is_st' column in the market table")
        flag(frame["is_st"].fillna(False).astype(bool), "suspended_or_st")
    if spec.exclude_financials:
        if not has_financial:
            raise ValueError(
                "spec.exclude_financials requires an 'is_financial' column in the market table"
            )
        flag(frame["is_financial"].fillna(False).astype(bool), "financial_firm")

    # E/P is NaN whenever either input is missing, so this single flag covers a
    # missing announcement and an unusable market cap.  Labelling on
    # `earnings_ttm` alone would leave an ep-gated row with no reason recorded,
    # which is exactly the kind of untraceable exclusion the log exists to stop.
    flag(frame["ep"].isna(), "missing_earnings")
    if spec.require_positive_earnings:
        flag(frame["earnings_signal"] <= 0, "non_positive_earnings")

    if not spec.size_cut_first:
        _apply_size_cut(frame, reason, spec, require_earnings=True)

    frame["exclusion_reason"] = reason
    frame["in_universe"] = reason == ""
    return frame


def _assign_groups(
    eligible: pd.DataFrame,
    spec: Ch3Spec,
    *,
    has_book: bool,
) -> tuple[pd.DataFrame, dict[str, float], dict[str, int]]:
    """Size split at the median, E/P split 30/40/30, then the B/M control.

    Every split is positional (see ``_take_low_group``) rather than a
    ``value <= quantile`` comparison, because real month-end market caps repeat
    and a comparison rule pushes an entire tied block to one side.
    """

    frame = eligible.copy()
    low_q, high_q = spec.value_breakpoints
    weighting = spec.breakpoint_weighting

    size_order = _ordered_by(frame, "size_me", ascending=True)
    small = _take_low_group(frame, size_order, 0.50, weighting)
    frame["size_group"] = np.where(frame.index.isin(small), "S", "B")

    # Top 30% of E/P is value, bottom 30% is growth.  The middle group is
    # labelled N (not the paper's M) so it can never be confused with the B/M
    # control's middle group in the same frame.
    ep_ascending = _ordered_by(frame, "ep", ascending=True)
    # Reverse the total order so tied cutoffs cannot put a name in both tails.
    ep_descending = list(reversed(ep_ascending))
    growth = _take_low_group(frame, ep_ascending, low_q, weighting)
    value = _take_low_group(frame, ep_descending, 1.0 - high_q, weighting)
    frame["value_group"] = np.select(
        [frame.index.isin(growth), frame.index.isin(value)],
        ["G", "V"],
        default="N",
    )
    if spec.negative_ep_policy == "force_growth":
        # Explicit statement in section 5.1. Keep the all-universe breakpoints;
        # this override may make G larger than 30% (documented convention).
        frame.loc[frame["ep"] < 0, "value_group"] = "G"
    frame["ep_group"] = frame["value_group"]
    frame["portfolio"] = frame["size_group"] + frame["value_group"]

    bm_low = bm_high = float("nan")
    frame["bm"] = np.nan
    frame["bm_group"] = pd.NA
    frame["bm_portfolio"] = pd.NA
    if spec.book_to_market_control and has_book:
        bm = frame["book_equity"] / frame["valuation_me"]
        if spec.require_positive_book_equity:
            bm = bm.where(frame["book_equity"] > 0)
        frame["bm"] = bm
        bm_valid = frame["bm"].notna()
        if bm_valid.any():
            bm_frame = frame.loc[bm_valid]
            bm_ascending = _ordered_by(bm_frame, "bm", ascending=True)
            bm_descending = list(reversed(bm_ascending))
            bm_low_group = _take_low_group(bm_frame, bm_ascending, low_q, weighting)
            bm_high_group = _take_low_group(bm_frame, bm_descending, 1.0 - high_q, weighting)
            # Top 30% of B/M is high (H), bottom 30% is low (L).
            frame["bm_group"] = pd.Series(pd.NA, index=frame.index, dtype="object")
            frame.loc[bm_low_group, "bm_group"] = "L"
            frame.loc[bm_high_group, "bm_group"] = "H"
            frame.loc[bm_valid & frame["bm_group"].isna(), "bm_group"] = "M"
            # The FF_ prefix travels with the label: anyone filtering for "SM"
            # in this frame must get the CH-3 small/middle cell, never the B/M
            # small/middle one.
            frame["bm_portfolio"] = np.where(
                frame["bm_group"].notna(),
                "FF_" + frame["size_group"] + frame["bm_group"].astype("object"),
                pd.NA,
            )
            if len(bm_low_group):
                bm_low = float(frame.loc[bm_low_group, "bm"].max())
            if len(bm_high_group):
                bm_high = float(frame.loc[bm_high_group, "bm"].min())

    breakpoints = {
        "size_cut_me": float(frame.loc[small, "size_me"].max()) if small else float("nan"),
        "ep_cut_low": float(frame.loc[growth, "ep"].max()) if growth else float("nan"),
        "ep_cut_high": float(frame.loc[value, "ep"].min()) if value else float("nan"),
        "bm_cut_low": bm_low,
        "bm_cut_high": bm_high,
    }
    counts = frame["portfolio"].value_counts().to_dict()
    return frame, breakpoints, {str(key): int(value) for key, value in counts.items()}


def _portfolio_return(group: pd.DataFrame, weighting: str) -> tuple[float, float]:
    """Return of a formed portfolio; invalid outcomes are never renormalised."""

    # Do not silently renormalise surviving names after a missing/invalid return.
    valid = (np.isfinite(group["me"]) & (group["me"] > 0)
             & np.isfinite(group["ret_next_1m"]) & (group["ret_next_1m"] >= -1))
    if not valid.all():
        return float("nan"), float(group["me"].sum())
    usable = group
    if usable.empty:
        return float("nan"), 0.0
    if weighting == "value_weighted":
        weights = usable["me"] / usable["me"].sum()
    else:
        weights = pd.Series(1.0 / len(usable), index=usable.index)
    return (
        float((weights * usable["ret_next_1m"]).sum()),
        float(usable["me"].sum()),
    )


def _mean_of(returns: Mapping[str, float], keys: Sequence[str]) -> float:
    """Average a set of portfolio legs, refusing to average a partial set.

    A factor leg that is missing means the month's factor is not defined.  If
    the missing leg were simply dropped, the survivors' weights would silently
    increase and SMB could move by 50% or more with nothing in the output
    saying so.  ``NaN`` propagates instead, and ``validate_ch3_result`` turns
    it into a failure.
    """

    values = []
    for key in keys:
        if key not in returns or np.isnan(returns[key]):
            return float("nan")
        values.append(returns[key])
    return float(np.mean(values)) if values else float("nan")


def _resolve_risk_free(
    by_month: pd.Series | None,
    realization_ordinal: int,
    spec: Ch3Spec,
) -> tuple[float, str]:
    """The one-year deposit rate for the month the return is earned in.

    Returns the monthly decimal rate and a label naming its source, so the
    diagnostic output always says whether MKT is a genuine excess return or a
    total return produced by the scalar fallback.
    """

    if by_month is not None:
        quoted = by_month.get(realization_ordinal)
        if quoted is not None and not pd.isna(quoted):
            if spec.risk_free_is_monthly_decimal:
                return float(quoted), "panel_monthly_decimal"
            return float((1.0 + float(quoted) / 100.0) ** (1.0 / 12.0) - 1.0), "panel"
    if spec.require_risk_free:
        raise ValueError(f"missing required risk-free rate for month ordinal {realization_ordinal}")
    return spec.monthly_risk_free(), "scalar_fallback"


# ------------------------------------------------------------------ main entry
def build_ch3_factors(panel: PointInTimePanel, spec: Ch3Spec) -> Ch3Result:
    """Build MKT, SMB, VMG and the B/M (FF-3) control factors from a PIT panel."""

    spec.validate()
    notes: list[str] = []

    market = _prepare_market(panel.market, spec.market_cap_lag_months)
    if spec.valuation_cap_field not in market:
        raise ValueError(f"missing valuation capitalisation field: {spec.valuation_cap_field}")
    if spec.earnings_field not in panel.fundamentals:
        raise ValueError(f"missing earnings field: {spec.earnings_field}")
    market["valuation_me"] = pd.to_numeric(market[spec.valuation_cap_field], errors="raise")
    notes.append(f"earnings input: {spec.earnings_field}; valuation denominator: {spec.valuation_cap_field}")
    notes.append("missing next-month returns do not affect formation sorts; affected portfolio returns are NaN")
    has_listing = "listing_months" in market.columns
    has_trading_days = {"trading_days_12m", "trading_days_1m"}.issubset(market.columns)
    has_st = "is_st" in market.columns
    has_financial = "is_financial" in market.columns
    has_book = "book_equity" in panel.fundamentals.columns

    if spec.book_to_market_control and not has_book:
        notes.append("book_equity absent from fundamentals: HML_BM and SMB_FF reported as missing")
    if not has_listing:
        notes.append("listing_months absent: no listing-history screen applied")
    if not has_trading_days:
        notes.append("trading-day counts absent: no trading-activity screen applied")
    notes.append(
        "size and the smallest-"
        f"{spec.exclude_smallest_fraction:.0%} cut use market cap lagged by "
        f"{spec.market_cap_lag_months} month(s) relative to the sort month; lag 0 reads the "
        "sort month end, which is the previous month relative to the return month "
        "[OPEN reading of the paper]"
    )
    notes.append(
        "screen order: the smallest-"
        f"{spec.exclude_smallest_fraction:.0%} cut is applied "
        + ("FIRST, to the raw tradable universe [OPEN]" if spec.size_cut_first else "LAST, to the investable universe [OPEN]")
    )

    # Attach the latest *announced* accounting row to every stock-month.  The
    # row id keeps the result aligned with `market` because merge_asof is a
    # left join that sorts its input.
    request = market.loc[:, ["security_id", "observation_date"]].rename(
        columns={"observation_date": "formation_date"}
    )
    request["_row_id"] = np.arange(len(market))
    joined = panel.available_fundamentals(request).sort_values("_row_id")
    if len(joined) != len(market):
        raise ValueError(
            f"point-in-time fundamental join returned {len(joined)} rows for "
            f"{len(market)} market rows; formation dates must be complete"
        )

    def _payload(column: str, default: Any) -> Any:
        if column in joined.columns:
            return joined[column].to_numpy()
        return default

    market = market.assign(
        earnings_ttm=_payload("earnings_ttm", np.nan),
        earnings_signal=_payload(spec.earnings_field, np.nan),
        book_equity=_payload("book_equity", np.nan),
        fundamental_period_end=_payload("fundamental_period_end", pd.NaT),
        fundamental_available_at=_payload("fundamental_available_at", pd.NaT),
    )
    # E/P uses the earnings most recently *announced* before the sort month end
    # (no fixed accounting lag) divided by the sort month end market cap.
    market["ep"] = (market["earnings_signal"] / market["valuation_me"]).replace([np.inf, -np.inf], np.nan)

    calendar = market.groupby("month_ordinal")["observation_date"].max()
    sample_start = pd.Timestamp(spec.sample_start)
    sample_end = pd.Timestamp(spec.sample_end)

    # ``sample_start``/``sample_end`` bound the *factor series*, i.e. the months
    # in which returns are earned — the convention the published CH-3 files
    # use.  The panel must therefore reach one month further back, to the
    # formation month of the first factor observation.
    rf_monthly = spec.monthly_risk_free()
    risk_free_column = (
        spec.risk_free_field
        if spec.risk_free_field and spec.risk_free_field in market.columns
        else None
    )
    risk_free_by_month: pd.Series | None = None
    if risk_free_column is None:
        notes.append(
            f"risk-free field {spec.risk_free_field!r} absent from the market table; "
            f"using the frozen scalar {spec.risk_free_annual:.4f} per year. MKT is "
            "therefore a total return unless that scalar is set deliberately."
        )
    else:
        risk_free_by_month = market.groupby("month_ordinal")[risk_free_column].median()
        spread = market.groupby("month_ordinal")[risk_free_column].nunique()
        if int(spread.max()) > 1:
            raise ValueError(f"risk-free field {risk_free_column!r} has conflicting values within a month")

    factor_rows: list[dict[str, Any]] = []
    portfolio_rows: list[dict[str, Any]] = []
    assignment_frames: list[pd.DataFrame] = []
    log_frames: list[pd.DataFrame] = []
    diagnostics_rows: list[dict[str, Any]] = []

    for ordinal in sorted(calendar.index):
        realization_date = (
            pd.Timestamp(calendar.loc[ordinal + 1]).to_period("M").to_timestamp("M")
            if (ordinal + 1) in calendar.index else pd.NaT
        )
        if pd.isna(realization_date) or not (sample_start <= realization_date <= sample_end):
            continue

        formation_date = calendar.loc[ordinal]
        cross = market.loc[market["month_ordinal"] == ordinal]
        if cross.empty:
            continue
        screened = _screen_cross_section(
            cross,
            spec,
            has_listing=has_listing,
            has_trading_days=has_trading_days,
            has_st=has_st,
            has_financial=has_financial,
        )

        eligible = screened.loc[screened["in_universe"]].copy()
        log_frames.append(
            screened.loc[
                :,
                [
                    "security_id",
                    "observation_date",
                    "me",
                    "size_me",
                    "ep",
                    "earnings_ttm",
                    "earnings_signal",
                    "valuation_me",
                    "fundamental_period_end",
                    "fundamental_available_at",
                    "in_universe",
                    "exclusion_reason",
                ],
            ].assign(formation_month=formation_date, realization_month=realization_date)
        )

        if eligible.empty:
            notes.append(f"empty eligible universe in {formation_date.date()}; month skipped")
            continue

        assigned, breakpoints, counts = _assign_groups(eligible, spec, has_book=has_book)
        assignments = assigned.loc[
            :,
            [
                "security_id",
                "size_group",
                "value_group",
                "ep_group",
                "portfolio",
                "me",
                "valuation_me",
                "size_me",
                "ep",
                "bm",
                "bm_group",
                "bm_portfolio",
                "ret_next_1m",
            ],
        ].assign(formation_month=formation_date, realization_month=realization_date)
        assignment_frames.append(assignments)

        portfolio_return: dict[str, float] = {}
        portfolio_size: dict[str, int] = {}
        for portfolio, group in assigned.groupby("portfolio", sort=True):
            value, total_me = _portfolio_return(group, spec.weighting)
            portfolio_return[str(portfolio)] = value
            portfolio_size[str(portfolio)] = int(len(group))
            portfolio_rows.append(
                {
                    "formation_month": formation_date,
                    "realization_month": realization_date,
                    "portfolio": str(portfolio),
                    "return": value,
                    "n_stocks": int(len(group)),
                    "total_me": total_me,
                }
            )

        # CH-3: SMB averages all three E/P legs; VMG is value minus growth.
        smb = _mean_of(portfolio_return, ("SV", "SN", "SG")) - _mean_of(
            portfolio_return, ("BV", "BN", "BG")
        )
        vmg = _mean_of(portfolio_return, ("SV", "BV")) - _mean_of(
            portfolio_return, ("SG", "BG")
        )

        hml_bm = float("nan")
        smb_ff = float("nan")
        bm_available = 0
        if spec.book_to_market_control and "bm_portfolio" in assigned.columns:
            bm_groups = assigned.loc[assigned["bm_portfolio"].notna()]
            bm_available = int(len(bm_groups))
            bm_returns: dict[str, float] = {}
            for portfolio, group in bm_groups.groupby("bm_portfolio", sort=True):
                value, _ = _portfolio_return(group, spec.weighting)
                bm_returns[str(portfolio)] = value
                portfolio_size[str(portfolio)] = int(len(group))
                portfolio_rows.append(
                    {
                        "formation_month": formation_date,
                        "realization_month": realization_date,
                        "portfolio": str(portfolio),
                        "return": value,
                        "n_stocks": int(len(group)),
                        "total_me": float(group["me"].sum()),
                    }
                )
            hml_bm = _mean_of(bm_returns, ("FF_SH", "FF_BH")) - _mean_of(
                bm_returns, ("FF_SL", "FF_BL")
            )
            smb_ff = _mean_of(bm_returns, ("FF_SH", "FF_SM", "FF_SL")) - _mean_of(
                bm_returns, ("FF_BH", "FF_BM", "FF_BL")
            )

        # MKT: value-weighted return of the surviving universe (the paper's
        # "top 70% of stocks"), in excess of the one-year deposit rate.
        if spec.market_universe == "all":
            market_cross = market.loc[
                (market["month_ordinal"] == ordinal)
                & (market["me"] > 0)
            ]
        else:
            market_cross = assigned
        mkt_value, mkt_me = _portfolio_return(market_cross, spec.weighting)

        # The factor is a return earned over the realisation month, so the
        # risk-free rate must be that month's rate, not the formation month's.
        # Using the formation month silently mis-states every January, when the
        # deposit rate is repriced.
        rf_monthly, risk_free_source = _resolve_risk_free(
            risk_free_by_month, ordinal + 1, spec
        )

        factor_rows.append(
            {
                "date": realization_date,
                "formation_month": formation_date,
                "MKT": mkt_value - rf_monthly,
                "SMB": smb,
                "VMG": vmg,
                "HML_BM": hml_bm,
                "SMB_FF": smb_ff,
            }
        )

        tracked = [*CH3_PORTFOLIOS, *FF_PORTFOLIOS]
        smallest_portfolio = min(
            (value for key, value in portfolio_size.items() if key in tracked),
            default=0,
        )
        diagnostics_rows.append(
            {
                "date": realization_date,
                "formation_month": formation_date,
                "n_all": int(len(cross)),
                "n_eligible": int(len(eligible)),
                "n_missing_realized_returns": int(assigned["ret_next_1m"].isna().sum()),
                "n_dropped_smallest": int(
                    (screened["exclusion_reason"] == "smallest_30pct").sum()
                ),
                "n_portfolio_missing": int(
                    sum(1 for key in CH3_PORTFOLIOS if key not in portfolio_return)
                ),
                "min_portfolio_n": int(smallest_portfolio),
                "below_min_width": bool(smallest_portfolio < spec.min_stocks_per_portfolio),
                "mkt_total_me": mkt_me,
                "risk_free_monthly": rf_monthly,
                "risk_free_source": risk_free_source,
                # The B/M control silently drops non-positive book equity (the
                # FF convention), so its breakpoints rest on a smaller
                # cross-section than the size sort.  Counting it keeps the
                # "every screen is logged" promise true.
                "n_bm_unavailable": int(len(eligible) - int(bm_available)),
                "breakpoints": breakpoints,
                "portfolio_counts": counts,
            }
        )

    if not factor_rows:
        raise ValueError(
            "no CH-3 factor observations were produced; check the sample window, "
            "the panel coverage and the universe screens"
        )

    factor_returns = (
        pd.DataFrame(factor_rows).drop_duplicates(subset="date").set_index("date").sort_index()
    )
    portfolio_returns = pd.DataFrame(portfolio_rows)
    assignments_frame = (
        pd.concat(assignment_frames, ignore_index=True) if assignment_frames else pd.DataFrame()
    )
    universe_log = pd.concat(log_frames, ignore_index=True) if log_frames else pd.DataFrame()
    diagnostics = pd.DataFrame(diagnostics_rows).set_index("date").sort_index()

    if universe_log.empty:
        raise ValueError("universe log is empty; traceability requires at least one month")

    return Ch3Result(
        factor_returns=factor_returns,
        portfolio_returns=portfolio_returns,
        assignments=assignments_frame,
        universe_log=universe_log,
        diagnostics=diagnostics,
        spec=spec,
        notes=sorted(set(notes)),
    )


def validate_ch3_result(result: Ch3Result) -> list[str]:
    """Post-build quality checks that W06 requires to pass.

    Returns a list of complaints; an empty list means the factor series is
    internally consistent and traceable.  Portfolio-width shortfalls are *not*
    complaints here — the NBER working paper requires at least 50 stocks per
    portfolio, and ``diagnostics['below_min_width']`` carries that flag so a
    thin month is visible rather than fatal.
    """

    problems: list[str] = []
    factors = result.factor_returns
    if result.spec.require_full_sample:
        expected = pd.period_range(result.spec.sample_start, result.spec.sample_end, freq="M")
        actual = pd.DatetimeIndex(factors.index).to_period("M")
        if not expected.equals(actual):
            problems.append("factor months do not cover the complete configured sample")

    for column in FACTOR_COLUMNS:
        if column not in factors.columns:
            problems.append(f"factor {column} missing from the output")
            continue
        if factors[column].isna().all():
            problems.append(f"factor {column} is entirely missing")
            continue
        if result.spec.require_full_sample and factors[column].isna().any():
            problems.append(f"factor {column} has missing values in the required sample")
        span = factors[column].loc[
            factors[column].first_valid_index() : factors[column].last_valid_index()
        ]
        if span.isna().any():
            problems.append(f"factor {column} has interior missing months")

    if len(factors) != result.diagnostics.shape[0]:
        problems.append(
            f"factor rows ({len(factors)}) disagree with diagnostics rows "
            f"({result.diagnostics.shape[0]})"
        )

    unknown_reasons = sorted(
        set(result.universe_log["exclusion_reason"].dropna().unique()) - set(EXCLUSION_ORDER) - {""}
    )
    if unknown_reasons:
        problems.append(f"universe log contains undocumented exclusion reasons: {unknown_reasons}")

    missing_portfolios = int(result.diagnostics["n_portfolio_missing"].sum())
    if missing_portfolios:
        problems.append(f"{missing_portfolios} month-portfolio cells were empty")

    logged = int(result.universe_log["in_universe"].sum())
    if logged != len(result.assignments):
        problems.append(
            f"universe log marks {logged} stock-months as in-universe but "
            f"{len(result.assignments)} assignments were produced"
        )

    dropped = result.universe_log.loc[result.universe_log["exclusion_reason"] == "smallest_30pct"]
    kept = result.assignments
    if not dropped.empty and not kept.empty:
        # The cut is inside each cross-section, so it must be monotone in the
        # size variable *within a month* — never across months, where the whole
        # market cap level has drifted.
        worst_dropped = dropped.groupby("formation_month")["size_me"].max()
        best_kept = kept.groupby("formation_month")["size_me"].min()
        overlap = (
            worst_dropped.to_frame("dropped")
            .join(best_kept.to_frame("kept"), how="inner")
            .dropna()
        )
        offending = overlap.loc[overlap["dropped"] > overlap["kept"]]
        if not offending.empty:
            problems.append(
                "the smallest-30% cut keeps a stock smaller than a dropped one in "
                f"{len(offending)} month(s), first: {offending.index[0].date()}"
            )

    if result.assignments["ret_next_1m"].isna().any():
        problems.append("assignments contain missing next-month returns")

    # Every factor row must be dated by the month it earns, which is the month
    # after the month it was formed in.
    formation = pd.to_datetime(factors["formation_month"])
    realized = pd.DatetimeIndex(factors.index)
    lag = (realized.year * 12 + realized.month) - (formation.dt.year * 12 + formation.dt.month)
    if (lag != 1).any():
        bad = factors.loc[(lag != 1).to_numpy()]
        problems.append(
            f"{len(bad)} factor row(s) are not dated by the month following their "
            f"formation month, first: {bad.index[0].date()}"
        )
    return problems
