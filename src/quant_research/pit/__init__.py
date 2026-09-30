"""Point-in-time data layer.

Import from here rather than from the submodules:

    from quant_research.pit import PointInTimePanel, CSVDataProvider, run_all
"""

from quant_research.pit.leakage import (
    ERROR,
    WARNING,
    LeakageError,
    LeakageFinding,
    LeakageReport,
    check_availability,
    check_chronological_split,
    check_execution_timing,
    check_factor_series_completeness,
    check_scaling_window,
    check_universe_is_dated,
    describe,
    run_all,
)
from quant_research.pit.panel import (
    FUNDAMENTAL_KEY,
    MARKET_KEY,
    PROVENANCE_FIELDS,
    PointInTimePanel,
    availability_violations,
)
from quant_research.pit.provider import (
    CSVDataProvider,
    assert_provider_protocol,
    describe_panel,
)

__all__ = [
    "CSVDataProvider",
    "ERROR",
    "FUNDAMENTAL_KEY",
    "LeakageError",
    "LeakageFinding",
    "LeakageReport",
    "MARKET_KEY",
    "PROVENANCE_FIELDS",
    "PointInTimePanel",
    "WARNING",
    "assert_provider_protocol",
    "availability_violations",
    "check_availability",
    "check_chronological_split",
    "check_execution_timing",
    "check_factor_series_completeness",
    "check_scaling_window",
    "check_universe_is_dated",
    "describe",
    "describe_panel",
    "run_all",
]
