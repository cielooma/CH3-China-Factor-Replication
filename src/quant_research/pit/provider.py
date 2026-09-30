"""A file-backed :class:`~quant_research.interfaces.DataProvider`.

The provider is deliberately dumb: it reads two CSV files, refuses to hand
back anything announced after ``as_of``, and validates the time contract
before returning.  Swapping in Wind, CSMAR or tushare later means writing one
more class with the same ``load`` signature; no factor or backtest code moves.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from quant_research.pit.panel import PointInTimePanel, PROVENANCE_FIELDS


def describe_panel(
    panel: PointInTimePanel,
    *,
    market_path: str | Path | None = None,
    fundamentals_path: str | Path | None = None,
) -> dict[str, object]:
    """Summarise a panel that has already been loaded.

    Kept separate from ``CSVDataProvider.describe`` so a caller holding an
    unvalidated panel — the diagnostic path — can describe it without
    triggering a second, strict read that would raise.
    """

    description: dict[str, object] = {
        "source": panel.source,
        "data_version": panel.data_version,
        "market_rows": int(len(panel.market)),
        "fundamental_rows": int(len(panel.fundamentals)),
    }
    if market_path is not None:
        description["market_path"] = str(market_path)
    if fundamentals_path is not None:
        description["fundamentals_path"] = str(fundamentals_path)
    if not panel.market.empty:
        description["securities"] = int(panel.market["security_id"].nunique())
        description["market_start"] = str(panel.market["observation_date"].min().date())
        description["market_end"] = str(panel.market["observation_date"].max().date())
    return description


class CSVDataProvider:
    """Load a point-in-time panel from two CSV extracts."""

    def __init__(
        self,
        market_path: str | Path,
        fundamentals_path: str | Path,
        *,
        validate: bool = True,
    ) -> None:
        self.market_path = Path(market_path)
        self.fundamentals_path = Path(fundamentals_path)
        self.validate = validate
        self._cache: PointInTimePanel | None = None

    def _panel(self) -> PointInTimePanel:
        if self._cache is None:
            for path in (self.market_path, self.fundamentals_path):
                if not path.exists():
                    raise FileNotFoundError(f"missing point-in-time extract: {path}")
            self._cache = PointInTimePanel.from_csv(
                self.market_path,
                self.fundamentals_path,
                validate=self.validate,
            )
        return self._cache

    def load(
        self,
        as_of: str,
        universe: Sequence[str] | None = None,
        fields: Sequence[str] | None = None,
    ) -> PointInTimePanel:
        """Return everything knowable at ``as_of``, optionally narrowed.

        ``universe`` filters on ``security_id``; ``fields`` selects value
        columns.  Provenance columns are always retained — a panel without
        them cannot pass the contract check.
        """

        panel = self._panel().as_of(as_of)

        market = panel.market
        if universe is not None:
            requested = list(universe)
            market = market.loc[market["security_id"].isin(requested)]
        if fields is not None:
            keep = list(dict.fromkeys([*PROVENANCE_FIELDS, *fields]))
            missing = [column for column in fields if column not in market.columns]
            if missing:
                raise KeyError(f"requested fields absent from the market extract: {missing}")
            market = market.loc[:, [column for column in keep if column in market.columns]]

        fundamentals = panel.fundamentals
        if universe is not None:
            fundamentals = fundamentals.loc[fundamentals["security_id"].isin(list(universe))]

        narrowed = PointInTimePanel(
            market=market.reset_index(drop=True),
            fundamentals=fundamentals.reset_index(drop=True),
            source=panel.source,
            data_version=panel.data_version,
        )
        if self.validate:
            narrowed.validate()
        return narrowed

    def describe(self) -> dict[str, object]:
        return describe_panel(
            self._panel(),
            market_path=self.market_path,
            fundamentals_path=self.fundamentals_path,
        )


def assert_provider_protocol(provider: object) -> None:
    """Fail loudly if a provider drifts from the frozen ``DataProvider`` shape."""

    load = getattr(provider, "load", None)
    if not callable(load):
        raise TypeError(f"{type(provider).__name__} has no callable load()")
    signature = getattr(load, "__code__", None)
    if signature is not None:
        params = set(signature.co_varnames[: signature.co_argcount])
        expected = {"as_of", "universe", "fields"}
        missing = expected - params
        if missing:
            raise TypeError(f"{type(provider).__name__}.load is missing parameters: {sorted(missing)}")
