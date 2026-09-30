"""Frozen CH-3 recipe.

    from quant_research.factors import Ch3Spec, build_ch3_factors
"""

from quant_research.factors.ch3 import (
    FACTOR_COLUMNS,
    SIZE_GROUPS,
    VALUE_GROUPS,
    Ch3Result,
    Ch3Spec,
    build_ch3_factors,
    validate_ch3_result,
)

__all__ = [
    "FACTOR_COLUMNS",
    "SIZE_GROUPS",
    "VALUE_GROUPS",
    "Ch3Result",
    "Ch3Spec",
    "build_ch3_factors",
    "validate_ch3_result",
]
