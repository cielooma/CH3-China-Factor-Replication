"""Public research interfaces.

Concrete Bloomberg, Wind, CSMAR, model, optimizer, and backtest implementations
are added only in their scheduled weeks.  The protocols keep their inputs and
outputs stable across reproductions.
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence


PointInTimePanel = Sequence[Mapping[str, Any]]
FeaturePanel = Sequence[Mapping[str, Any]]
ForecastPanel = Sequence[Mapping[str, Any]]
CostBreakdown = Mapping[str, float]
TargetWeights = Mapping[str, float]
BacktestResult = Mapping[str, Any]


class DataProvider(Protocol):
    def load(
        self,
        as_of: str,
        universe: Sequence[str],
        fields: Sequence[str],
    ) -> PointInTimePanel:
        """Load only observations available no later than ``as_of``."""


class FactorEngine(Protocol):
    def transform(self, panel: PointInTimePanel) -> FeaturePanel:
        """Create point-in-time features from validated observations."""


class AlphaModel(Protocol):
    def fit(self, train: FeaturePanel, as_of: str) -> "AlphaModel":
        """Fit using observations available no later than ``as_of``."""

    def predict(self, panel: FeaturePanel, as_of: str) -> ForecastPanel:
        """Return forecasts with model, seed, and training-cutoff metadata."""


class CostModel(Protocol):
    def estimate(
        self,
        orders: Mapping[str, float],
        market_state: Mapping[str, Mapping[str, float]],
    ) -> CostBreakdown:
        """Estimate dated fees, spread, and market impact."""


class Optimizer(Protocol):
    def solve(
        self,
        forecast: ForecastPanel,
        risk: Mapping[str, Any],
        cost: CostBreakdown,
        constraints: Mapping[str, Any],
    ) -> TargetWeights:
        """Produce target weights or a clear infeasibility diagnosis."""


class BacktestEngine(Protocol):
    def run(self, config: Mapping[str, Any]) -> BacktestResult:
        """Run the event-timed, cost-aware, reproducible backtest."""

