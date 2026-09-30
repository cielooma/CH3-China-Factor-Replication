"""Reusable foundations for the six-month quantitative research plan.

Subpackages, added as each research phase needs them:

``quant_research.pit``
    Point-in-time panels, the as-of fundamental join, and the five leakage
    checks.  Everything else is built on top of this layer.

``quant_research.factors``
    Factor constructions.  CH-3 (Liu–Stambaugh–Yuan 2019) is the first.

``quant_research.repro``
    Config fingerprints, run manifests and the experiment registry, so any
    published number can be traced back to the code and data that made it.
"""

from .contracts import PointInTimeRecord, validate_point_in_time
from .metrics import annualized_return, cumulative_wealth, max_drawdown, simple_returns

__all__ = [
    "PointInTimeRecord",
    "validate_point_in_time",
    "simple_returns",
    "cumulative_wealth",
    "annualized_return",
    "max_drawdown",
]
