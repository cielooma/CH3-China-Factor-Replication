"""Point-in-time panels and the temporal rules they must satisfy.

The project keeps two tables apart because their clocks are different:

``market``
    One row per security and month end.  Prices, returns, market
    capitalisation and trading status are knowable at that month end, so
    ``observation_date == available_at == trade_date``.

``fundamentals``
    One row per security and fiscal period.  ``observation_date`` is the end
    of the fiscal period, while ``available_at`` is the announcement date,
    which is strictly later.  A signal formed on date ``t`` may only use the
    most recent row with ``available_at <= t``.  Selecting on
    ``observation_date`` instead is the single most common way a Chinese
    A-share backtest silently buys on future earnings.

Both tables carry ``source`` and ``data_version`` so any published number can
be traced back to exactly one extract.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from quant_research.contracts import (
    REQUIRED_POINT_IN_TIME_FIELDS,
    PointInTimeValidationError,
    validate_point_in_time,
)

PROVENANCE_FIELDS: tuple[str, ...] = REQUIRED_POINT_IN_TIME_FIELDS

MARKET_KEY: tuple[str, ...] = ("security_id", "observation_date")
FUNDAMENTAL_KEY: tuple[str, ...] = ("security_id", "observation_date", "available_at")

_DATE_FIELDS = ("observation_date", "available_at", "trade_date")


def availability_violations(table: pd.DataFrame) -> pd.DataFrame:
    """Return the rows that break ``observation <= available <= trade``.

    Vectorised so it can run over a full panel on every load; the returned
    frame gains a ``violation`` column naming the rule that failed.
    """

    observation = pd.to_datetime(table["observation_date"])
    available = pd.to_datetime(table["available_at"])
    trade = pd.to_datetime(table["trade_date"])

    late = observation > available
    early = available > trade
    mask = late | early
    if not mask.any():
        return table.iloc[0:0].assign(violation=pd.Series(dtype="object"))

    offenders = table.loc[mask].copy()
    labels = pd.Series("", index=offenders.index, dtype="object")
    labels.loc[late[mask]] = "observation_date > available_at (value known before it existed)"
    labels.loc[early[mask]] = "available_at > trade_date (traded before it was knowable)"
    offenders["violation"] = labels
    return offenders


def _require_columns(table: pd.DataFrame, required: Sequence[str], name: str) -> None:
    missing = [column for column in required if column not in table.columns]
    if missing:
        raise PointInTimeValidationError(
            f"{name} is missing required columns: {', '.join(missing)}"
        )


def _coerce_date_column(series: pd.Series, name: str, column: str) -> pd.Series:
    """Parse one date column, refusing the formats that fail silently.

    Two traps are closed here.  First, a numeric column is ambiguous: a
    ``99999999`` sentinel parses happily as 1970-01-01 and then quietly
    back-dates an announcement, so the only accepted numeric form is YYYYMMDD
    inside a plausible year range.  Second, the result is always nanosecond
    resolution, because pandas 3 may otherwise infer microseconds and
    ``merge_asof`` refuses to join across resolutions.
    """

    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        as_float = series.to_numpy(dtype="float64", copy=False)
        as_int = np.where(np.isnan(as_float), -1, as_float).astype("int64")
        valid = (as_float == as_int) & (as_int >= 19900101) & (as_int <= 21001231)
        if not valid.all():
            offenders = series.loc[~valid].head(3).tolist()
            raise PointInTimeValidationError(
                f"{name}.{column} is numeric; YYYYMMDD is the only accepted numeric "
                f"date format. Offending values: {offenders}"
            )
        return pd.to_datetime(
            pd.Series(as_int, index=series.index).astype(str), format="%Y%m%d", errors="raise"
        ).astype("datetime64[ns]")
    try:
        return pd.to_datetime(series, errors="raise").astype("datetime64[ns]")
    except (ValueError, TypeError) as exc:
        raise PointInTimeValidationError(
            f"{name}.{column} could not be parsed as a date: {exc}"
        ) from exc


def _coerce_dates(table: pd.DataFrame, name: str) -> pd.DataFrame:
    coerced = table.copy()
    for column in _DATE_FIELDS:
        if column in coerced.columns:
            coerced[column] = _coerce_date_column(coerced[column], name, column)
    return coerced


def _validate_table(
    table: pd.DataFrame,
    name: str,
    key: tuple[str, ...],
    provenance: Mapping[str, str],
) -> pd.DataFrame:
    _require_columns(table, PROVENANCE_FIELDS, name)
    if table.empty:
        return _coerce_dates(table, name)

    table = _coerce_dates(table, name)

    for column in ("security_id", "source", "data_version"):
        blanks = table[column].astype("string").str.strip().eq("").fillna(True)
        if blanks.any():
            raise PointInTimeValidationError(
                f"{name}.{column} contains {int(blanks.sum())} empty value(s)"
            )
    for column in _DATE_FIELDS:
        if table[column].isna().any():
            raise PointInTimeValidationError(
                f"{name}.{column} contains {int(table[column].isna().sum())} null date(s)"
            )

    duplicated = table.duplicated(subset=list(key))
    if duplicated.any():
        example = table.loc[duplicated, list(key)].iloc[0]
        raise PointInTimeValidationError(
            f"{name} has duplicate rows for {tuple(example)}; "
            "one security may appear at most once per period"
        )

    for column, expected in provenance.items():
        observed = table[column].astype("string").unique()
        unexpected = sorted({value for value in observed if value != expected})
        if unexpected:
            raise PointInTimeValidationError(
                f"{name}.{column} declares {unexpected} but the panel says {expected!r}"
            )

    violations = availability_violations(table)
    if not violations.empty:
        preview = violations[list(key) + ["violation"]].head(3).to_dict("records")
        raise PointInTimeValidationError(
            f"{name} violates the time contract in {len(violations)} row(s): {preview}"
        )
    return table


@dataclass
class PointInTimePanel:
    """A validated, provenance-carrying pair of research tables."""

    market: pd.DataFrame
    fundamentals: pd.DataFrame
    source: str
    data_version: str

    def __post_init__(self) -> None:
        if not str(self.source).strip():
            raise PointInTimeValidationError("source cannot be empty")
        if not str(self.data_version).strip():
            raise PointInTimeValidationError("data_version cannot be empty")
        # Dates are parsed on construction, never inside validate(): a panel
        # built with validate=False is still used to *diagnose* a bad extract,
        # and a diagnostic path that crashes on string dates is useless.
        self.market = _coerce_dates(self.market, "market")
        self.fundamentals = _coerce_dates(self.fundamentals, "fundamentals")

    # ------------------------------------------------------------------ build
    @classmethod
    def from_frames(
        cls,
        market: pd.DataFrame,
        fundamentals: pd.DataFrame,
        *,
        source: str,
        data_version: str,
        validate: bool = True,
    ) -> "PointInTimePanel":
        panel = cls(
            market=market.copy(),
            fundamentals=fundamentals.copy(),
            source=source,
            data_version=data_version,
        )
        if validate:
            panel.validate()
        return panel

    @classmethod
    def from_csv(
        cls,
        market_path: str | Path,
        fundamentals_path: str | Path,
        *,
        validate: bool = True,
    ) -> "PointInTimePanel":
        market = pd.read_csv(market_path)
        fundamentals = pd.read_csv(fundamentals_path)
        if market.empty or fundamentals.empty:
            raise PointInTimeValidationError("both market and fundamentals tables must be non-empty")
        source = str(market["source"].iloc[0]) if "source" in market else "unknown"
        data_version = (
            str(market["data_version"].iloc[0]) if "data_version" in market else "unknown"
        )
        return cls.from_frames(
            market,
            fundamentals,
            source=source,
            data_version=data_version,
            validate=validate,
        )

    # --------------------------------------------------------------- validate
    def validate(self) -> None:
        provenance = {"source": self.source, "data_version": self.data_version}
        self.market = _validate_table(self.market, "market", MARKET_KEY, provenance)
        self.fundamentals = _validate_table(
            self.fundamentals, "fundamentals", FUNDAMENTAL_KEY, provenance
        )

    # ----------------------------------------------------------------- query
    def as_of(self, as_of: str | pd.Timestamp) -> "PointInTimePanel":
        """Truncate the panel to everything knowable and actionable at ``as_of``.

        Three filters, because a row can be unusable for three different
        reasons: the value may not exist yet (``observation_date``), may not be
        public yet (``available_at``), or may not be tradable yet
        (``trade_date``).  Checking only the first two lets a row whose value is
        current but whose ``trade_date`` sits years in the future slip into a
        factor.
        """

        cutoff = pd.Timestamp(as_of)
        market = self.market.loc[
            (self.market["observation_date"] <= cutoff)
            & (self.market["available_at"] <= cutoff)
            & (self.market["trade_date"] <= cutoff)
        ]
        fundamentals = self.fundamentals.loc[
            (self.fundamentals["available_at"] <= cutoff)
            & (self.fundamentals["trade_date"] <= cutoff)
        ]
        return PointInTimePanel(
            market=market.reset_index(drop=True),
            fundamentals=fundamentals.reset_index(drop=True),
            source=self.source,
            data_version=self.data_version,
        )

    def available_fundamentals(self, requests: pd.DataFrame) -> pd.DataFrame:
        """Attach the latest *announced* fundamental row to each request.

        ``requests`` must carry ``security_id`` and ``formation_date``.  The
        join is a backward ``merge_asof`` on ``available_at``, never on
        ``observation_date``: at a formation date the researcher may only see
        the most recent announcement, not the most recent fiscal period.

        Ties are resolved deterministically.  Two fiscal periods can legally be
        announced on the same day (a quarterly report and a restated annual
        one), and ``merge_asof`` keeps the last tied row — which would
        otherwise be decided by the order the vendor happened to serialise its
        file in.  The right frame is pre-sorted by fiscal period, then stably
        sorted on ``available_at``, so the later period wins and the result does
        not depend on row order.
        """

        _require_columns(requests, ("security_id", "formation_date"), "requests")
        if self.fundamentals.empty:
            return requests.copy()

        left = requests.copy()
        left["formation_date"] = pd.to_datetime(
            left["formation_date"], errors="raise"
        ).astype("datetime64[ns]")
        left = left.loc[left["formation_date"].notna()]

        # Only accounting payload travels across the join; the market table
        # keeps its own provenance.  The right-hand availability column is
        # renamed so the two clocks never collide in one frame.
        payload = [
            column
            for column in self.fundamentals.columns
            if column
            not in {
                "security_id",
                "source",
                "data_version",
                "trade_date",
                "available_at",
                "observation_date",
            }
        ]
        right = self.fundamentals.loc[:, ["security_id", "available_at", "observation_date", *payload]]
        right = right.rename(
            columns={
                "observation_date": "fundamental_period_end",
                "available_at": "fundamental_available_at",
            }
        )
        right["_join_available_at"] = right["fundamental_available_at"].astype(
            "datetime64[ns]"
        )
        # Sort by period first, then stably by the join key: within a tie the
        # most recent fiscal period ends up last, which is the row merge_asof
        # keeps.
        right = right.sort_values(
            ["fundamental_period_end", "_join_available_at"], kind="mergesort"
        ).sort_values("_join_available_at", kind="mergesort")

        left = left.sort_values("formation_date", kind="mergesort")
        merged = pd.merge_asof(
            left,
            right,
            left_on="formation_date",
            right_on="_join_available_at",
            by="security_id",
            direction="backward",
        )
        return merged.reset_index(drop=True)

    def to_records(self, table: str = "market", limit: int | None = None) -> list[dict[str, Any]]:
        """Materialise records so ``contracts.validate_point_in_time`` can check them."""

        frame = self.market if table == "market" else self.fundamentals
        if limit is not None:
            frame = frame.head(limit)
        return frame.to_dict("records")

    def assert_contracts(self, limit: int = 200) -> None:
        """Second, independent check through the frozen contract validator."""

        validate_point_in_time(self.to_records("market", limit))
        validate_point_in_time(self.to_records("fundamentals", limit))

    # ------------------------------------------------------------ provenance
    def fingerprint(self) -> str:
        from quant_research.repro.manifest import fingerprint

        return fingerprint(
            {
                "source": self.source,
                "data_version": self.data_version,
                "market_rows": int(len(self.market)),
                "fundamental_rows": int(len(self.fundamentals)),
                "market_span": [
                    str(self.market["observation_date"].min()),
                    str(self.market["observation_date"].max()),
                ]
                if not self.market.empty
                else [],
            }
        )
