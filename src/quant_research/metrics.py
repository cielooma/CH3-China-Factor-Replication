"""Small dependency-free performance metrics for W01 smoke tests."""

from __future__ import annotations

from math import prod
from typing import Iterable, Sequence


def _to_float_list(values: Iterable[float], name: str) -> list[float]:
    result = [float(value) for value in values]
    if not result:
        raise ValueError(f"{name} cannot be empty")
    return result


def simple_returns(prices: Sequence[float]) -> list[float]:
    """Return consecutive simple returns from strictly positive prices."""

    values = _to_float_list(prices, "prices")
    if len(values) < 2:
        raise ValueError("prices must contain at least two observations")
    if any(price <= 0 for price in values):
        raise ValueError("prices must be strictly positive")
    return [current / previous - 1.0 for previous, current in zip(values, values[1:])]


def cumulative_wealth(
    returns: Iterable[float],
    initial_wealth: float = 1.0,
) -> list[float]:
    """Return the wealth path including the initial value."""

    values = _to_float_list(returns, "returns")
    if initial_wealth <= 0:
        raise ValueError("initial_wealth must be positive")
    wealth = [float(initial_wealth)]
    for value in values:
        if value <= -1:
            raise ValueError("a simple return cannot be less than or equal to -100%")
        wealth.append(wealth[-1] * (1.0 + value))
    return wealth


def annualized_return(returns: Iterable[float], periods_per_year: int = 12) -> float:
    """Geometrically annualize a non-empty simple-return series."""

    values = _to_float_list(returns, "returns")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    gross = prod(1.0 + value for value in values)
    if gross < 0:
        raise ValueError("compounded wealth cannot be negative")
    return gross ** (periods_per_year / len(values)) - 1.0


def max_drawdown(returns: Iterable[float]) -> float:
    """Return maximum drawdown as a non-positive fraction."""

    wealth = cumulative_wealth(returns)
    peak = wealth[0]
    worst = 0.0
    for value in wealth:
        peak = max(peak, value)
        drawdown = value / peak - 1.0
        worst = min(worst, drawdown)
    return worst

