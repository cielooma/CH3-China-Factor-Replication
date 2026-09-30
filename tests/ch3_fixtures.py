"""Shared fixtures for the CH-3 tests.

The hand-built panels here are deliberately tiny and readable: a test that
asserts "the smallest 30% were dropped" should be checkable by eye, not by
trusting another 200 lines of pandas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MARKET_COLUMNS = [
    "security_id",
    "observation_date",
    "available_at",
    "trade_date",
    "source",
    "data_version",
    "ret_1m",
    "me",
    "listing_months",
    "trading_days_12m",
    "trading_days_1m",
    "is_st",
    "is_financial",
    "rf_1y_deposit",
]


def month_ends(periods: list[str]) -> pd.DatetimeIndex:
    index = pd.PeriodIndex(periods, freq="M").to_timestamp(how="end")
    return pd.DatetimeIndex(index).normalize().astype("datetime64[ns]")


#: Accounting periods that end before the first market month, so that a
#: trailing-twelve-month figure is already announced when the sample starts.
PRIOR_PERIODS = ["1999-08", "1999-09", "1999-10", "1999-11", "1999-12"]


def build_market(
    securities: list[str],
    months: list[str],
    *,
    market_caps: dict[str, float] | None = None,
    monthly_returns: dict[str, float] | None = None,
    listing_months: int = 24,
    trading_days_12m: int = 200,
    trading_days_1m: int = 20,
    source: str = "test",
    data_version: str = "test_v1",
    risk_free_percent: float = 0.0,
) -> pd.DataFrame:
    """A rectangular market table: every security trades in every month."""

    caps = market_caps or {}
    returns = monthly_returns or {}
    rows = []
    for stamp in month_ends(months):
        for security in securities:
            rows.append(
                {
                    "security_id": security,
                    "observation_date": stamp,
                    "available_at": stamp,
                    "trade_date": stamp,
                    "source": source,
                    "data_version": data_version,
                    "ret_1m": returns.get(security, 0.01),
                    "me": caps.get(security, 1000.0),
                    "listing_months": listing_months,
                    "trading_days_12m": trading_days_12m,
                    "trading_days_1m": trading_days_1m,
                    "is_st": False,
                    "is_financial": False,
                    "rf_1y_deposit": risk_free_percent,
                }
            )
    return pd.DataFrame(rows, columns=MARKET_COLUMNS)


def build_fundamentals(
    securities: list[str],
    periods: list[str],
    *,
    earnings: dict[str, float] | None = None,
    book_equity: dict[str, float] | None = None,
    announcement_lag_days: int = 60,
    source: str = "test",
    data_version: str = "test_v1",
) -> pd.DataFrame:
    """One trailing-twelve-month accounting row per security and quarter."""

    values = earnings or {}
    books = book_equity or {}
    rows = []
    for stamp in month_ends(periods):
        available = stamp + pd.Timedelta(days=announcement_lag_days)
        for security in securities:
            rows.append(
                {
                    "security_id": security,
                    "observation_date": stamp,
                    "available_at": available,
                    "trade_date": available,
                    "source": source,
                    "data_version": data_version,
                    "earnings_ttm": values.get(security, 50.0),
                    "book_equity": books.get(security, 500.0),
                }
            )
    return pd.DataFrame(rows)


def ranked_panel(n_securities: int = 60, months: int = 4) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A panel where EP falls as market cap rises, and returns are flat.

    Deterministic and easy to reason about: the size ranking is the identity,
    so which stocks should be dropped, and which corner of the 2x3 sort they
    land in, can be worked out by hand.
    """

    securities = [f"{index:04d}.SZ" for index in range(n_securities)]
    caps = {security: float(10 + 10 * rank) for rank, security in enumerate(securities)}
    earnings = {security: float(n_securities - rank) * 10.0 for rank, security in enumerate(securities)}
    periods = [f"2000-{month:02d}" for month in range(1, months + 1)]
    market = build_market(securities, periods, market_caps=caps)
    fundamentals = build_fundamentals(
        securities,
        PRIOR_PERIODS + periods,
        earnings=earnings,
        book_equity={s: 500.0 for s in securities},
    )
    return market, fundamentals


def next_month_return(market: pd.DataFrame, security: str, month: int) -> float:
    """The realised return in the month after the given formation month index."""

    frame = market.loc[market["security_id"] == security].sort_values("observation_date")
    return float(frame["ret_1m"].iloc[month + 1])


def exposure_matrix(n_securities: int = 60, months: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A panel where size, E/P and B/M vary independently.

    The characteristics are built as ``me * target`` so that each ratio is a
    permutation that is orthogonal to size.  That matters: with a constant book
    value, B/M is just inverse size and three of the six B/M portfolios come out
    empty, which would make the test pass for the wrong reason.
    """

    securities = [f"{index:04d}.SZ" for index in range(n_securities)]
    caps = {security: 10.0 * (rank + 1) for rank, security in enumerate(securities)}
    ep_target = {
        security: ((rank * 7) % n_securities + 1) / (n_securities * 8.0)
        for rank, security in enumerate(securities)
    }
    bm_target = {
        security: ((rank * 11) % n_securities + 1) / (n_securities / 3.0)
        for rank, security in enumerate(securities)
    }
    earnings = {security: caps[security] * ep_target[security] for security in securities}
    book_equity = {security: caps[security] * bm_target[security] for security in securities}

    periods = [f"2001-{month:02d}" for month in range(1, months + 1)]
    market = build_market(securities, periods, market_caps=caps)
    fundamentals = build_fundamentals(
        securities, PRIOR_PERIODS + periods, earnings=earnings, book_equity=book_equity
    )
    return market, fundamentals


def flat_index(index) -> list[str]:
    return [str(pd.Timestamp(item).date()) for item in index]


def correlation(left: pd.Series, right: pd.Series) -> float:
    joined = pd.concat([left.rename("l"), right.rename("r")], axis=1).dropna()
    return float(np.corrcoef(joined["l"], joined["r"])[0, 1])
