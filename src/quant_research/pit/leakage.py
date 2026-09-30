"""Five checks that catch the leakage patterns this project forbids.

Each check returns findings instead of raising, so a report can be written
even when the run is rejected.  ``LeakageReport.raise_if_errors`` is the
gate the pipeline calls before any factor number is produced.

The five red lines from the project README map one-to-one onto the checks:

===================================  ==========================  ==================
README red line                      check                       wired into a run?
===================================  ==========================  ==================
same-close / future execution        ``check_execution_timing``  yes — factor timing
current-index membership backfill    ``check_universe_is_dated`` only if a dated
                                                                 membership frame is
                                                                 supplied
full-sample standardisation          ``check_scaling_window``    yes — declarations
random K-fold splits                 ``check_chronological_split`` yes — declared
                                                                 periods
using data before it existed         ``check_availability``      yes — on every load
===================================  ==========================  ==================

Two of these answer structural questions ("is membership dated at all", "do the
declared windows overlap"), not empirical ones.  They cannot detect a vendor
that backfills announcement dates, and they cannot see inside a transform.  The
stronger enforcement for CH-3 is structural and lives in the factor code: every
breakpoint and quantile is recomputed inside its own cross-section, and the
tests pin that down.  Read a PASS as "these declarations and these timestamps
are consistent", never as "this run is leak-free".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Iterable, Mapping, Sequence

import pandas as pd

from quant_research.pit.panel import PointInTimePanel, availability_violations

ERROR = "error"
WARNING = "warning"


def _plain(value: object) -> object:
    """Recursively convert a value into something ``json.dumps`` accepts.

    Evidence carries ``Timestamp`` objects and numpy scalars straight out of
    pandas.  A report that cannot be written is worse than no report, and this
    is the last place the conversion can be guaranteed.
    """

    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return str(value.date()) if hasattr(value, "date") else str(value)
    if not isinstance(value, (str, bytes)) and hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):  # pragma: no cover - defensive
            return str(value)
    return value


@dataclass(frozen=True)
class LeakageFinding:
    code: str
    severity: str
    message: str
    evidence: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "evidence": _plain(dict(self.evidence)),
        }


@dataclass
class LeakageReport:
    findings: list[LeakageFinding] = field(default_factory=list)

    def add(self, finding: LeakageFinding) -> None:
        self.findings.append(finding)

    @property
    def errors(self) -> list[LeakageFinding]:
        return [item for item in self.findings if item.severity == ERROR]

    @property
    def warnings(self) -> list[LeakageFinding]:
        return [item for item in self.findings if item.severity == WARNING]

    @property
    def passed(self) -> bool:
        return not self.errors

    def extend(self, other: "LeakageReport") -> None:
        self.findings.extend(other.findings)

    def raise_if_errors(self, context: str = "pipeline", allow_override: bool = False) -> None:
        if self.passed or allow_override:
            return
        detail = "; ".join(f"{item.code}: {item.message}" for item in self.errors)
        raise LeakageError(f"{context} blocked by leakage checks -> {detail}")

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "n_errors": len(self.errors),
            "n_warnings": len(self.warnings),
            "findings": [item.to_dict() for item in self.findings],
        }


class LeakageError(RuntimeError):
    """Raised when a run is blocked by a leakage check."""


# --------------------------------------------------------------------- checks
def check_availability(panel: PointInTimePanel) -> LeakageReport:
    """Nothing may be used before its ``available_at``."""

    report = LeakageReport()
    for name, table in (("market", panel.market), ("fundamentals", panel.fundamentals)):
        violations = availability_violations(table)
        if violations.empty:
            continue
        examples = violations.head(5)[
            ["security_id", "observation_date", "available_at", "trade_date", "violation"]
        ].to_dict("records")
        report.add(
            LeakageFinding(
                code="availability",
                severity=ERROR,
                message=(
                    f"{name} has {len(violations)} row(s) that break "
                    "observation_date <= available_at <= trade_date"
                ),
                evidence={"table": name, "n_violations": int(len(violations)), "examples": examples},
            )
        )
    return report


def check_universe_is_dated(
    membership: pd.DataFrame,
    *,
    security_column: str = "security_id",
    date_column: str = "in_universe_date",
) -> LeakageReport:
    """Universe membership must be dated, not a snapshot of today's index."""

    report = LeakageReport()
    if security_column not in membership.columns:
        report.add(
            LeakageFinding(
                code="survivorship_universe",
                severity=ERROR,
                message=f"universe is missing {security_column!r}",
            )
        )
        return report
    if date_column not in membership.columns:
        report.add(
            LeakageFinding(
                code="survivorship_universe",
                severity=ERROR,
                message=(
                    f"universe has no dated membership column {date_column!r}; "
                    "a static universe is a current-constituent backfill"
                ),
                evidence={"columns": sorted(map(str, membership.columns))},
            )
        )
        return report
    if pd.to_datetime(membership[date_column], errors="coerce").isna().any():
        report.add(
            LeakageFinding(
                code="survivorship_universe",
                severity=ERROR,
                message=f"{date_column} contains unparseable or missing dates",
            )
        )
    return report


_FULL_SAMPLE_TOKENS = frozenset(
    {"full", "fullsample", "all", "global", "entire", "whole", "sample", "static", "insample"}
)


def _declares_full_sample(window: str) -> bool:
    """True when a declared scaling window names the whole sample.

    Matching is on tokens, so ``full``, ``in_sample``, ``entire_sample``,
    ``whole_sample``, ``static_full`` and ``full_sample`` are all caught, while
    ``cross_section_at_formation`` and ``latest_announcement_before_formation``
    are not.
    """

    normalised = window.strip().lower().replace("-", "_").replace(".", "_")
    tokens = {token for token in re.split(r"[^a-z0-9]+", normalised) if token}
    tokens.add(normalised)
    return bool(tokens & _FULL_SAMPLE_TOKENS)


def check_scaling_window(feature_specs: Sequence[Mapping[str, object]]) -> LeakageReport:
    """Standardisation must be fit on an expanding or rolling window.

    Each feature spec declares ``fit_window``.  Anything that names the whole
    sample is a look-ahead: the scaling constants embed the test period.

    **This is a declaration gate, not a proof.**  It cannot look inside a
    transform and see that a mean was computed over all months; it can only
    refuse a declaration that admits to it and refuse a missing declaration.
    The structural enforcement lives in the factor code, where every breakpoint
    and quantile is recomputed inside its own cross-section, and in the tests
    that pin that behaviour down.
    """

    report = LeakageReport()
    offenders: list[dict[str, object]] = []
    for spec in feature_specs:
        name = str(spec.get("name", "<unnamed>"))
        window = spec.get("fit_window")
        if window is None:
            offenders.append({"feature": name, "fit_window": None, "reason": "undeclared"})
            continue
        if _declares_full_sample(str(window)):
            offenders.append(
                {"feature": name, "fit_window": window, "reason": "declares the full sample"}
            )
    if offenders:
        report.add(
            LeakageFinding(
                code="full_sample_scaling",
                severity=ERROR,
                message=(
                    f"{len(offenders)} feature(s) declare a full-sample or undeclared "
                    "scaling window; use expanding or trailing windows instead"
                ),
                evidence={"features": offenders},
            )
        )
    return report


def check_chronological_split(
    periods: Mapping[str, Sequence[str]],
    *,
    require_convention: bool = True,
) -> LeakageReport:
    """Declared periods must be ordered and disjoint, whatever they are called.

    Disjointness is enforced on **every** pair, sorted by start date — not only
    on periods named train/validation/test.  A design that names its windows
    ``fold_a`` and ``fold_b`` (or ``sub_period_1`` and ``sub_period_2``) gets
    exactly the same overlap check.  ``require_convention=False`` only waives
    the naming warning, never the overlap rule.
    """

    report = LeakageReport()
    parsed: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
    for name, bounds in periods.items():
        if len(bounds) != 2:
            report.add(
                LeakageFinding(
                    code="random_split",
                    severity=ERROR,
                    message=f"period {name!r} must be given as (start, end)",
                    evidence={"period": name, "bounds": list(bounds)},
                )
            )
            continue
        start, end = pd.Timestamp(bounds[0]), pd.Timestamp(bounds[1])
        if start > end:
            report.add(
                LeakageFinding(
                    code="random_split",
                    severity=ERROR,
                    message=f"period {name!r} starts after it ends",
                    evidence={"period": name, "start": str(start), "end": str(end)},
                )
            )
            continue
        parsed[name] = (start, end)

    ordered = ["train", "validation", "test"]
    # Check every declared pair in date order, regardless of how it is named.
    by_start = sorted(parsed.items(), key=lambda item: item[1][0])
    for (left_name, left_bounds), (right_name, right_bounds) in zip(by_start, by_start[1:]):
        if left_bounds[1] >= right_bounds[0]:
            report.add(
                LeakageFinding(
                    code="random_split",
                    severity=ERROR,
                    message=(
                        f"{left_name} (ends {left_bounds[1].date()}) overlaps or touches "
                        f"{right_name} (starts {right_bounds[0].date()}); splits must be "
                        "chronological"
                    ),
                    evidence={"left": left_name, "right": right_name},
                )
            )
    known = set(parsed)
    unexpected = sorted(known - set(ordered))
    if unexpected and require_convention:
        report.add(
            LeakageFinding(
                code="random_split",
                severity=WARNING,
                message=f"periods outside the train/validation/test convention: {unexpected}",
            )
        )
    return report


def check_execution_timing(
    signal_date: str | date | datetime,
    execution_date: str | date | datetime,
    *,
    label: str = "portfolio",
    allow_same_close: bool = False,
) -> LeakageReport:
    """A signal observed at a close may not be filled at that same close."""

    report = LeakageReport()
    signal = pd.Timestamp(signal_date)
    execution = pd.Timestamp(execution_date)
    if execution < signal:
        report.add(
            LeakageFinding(
                code="execution_timing",
                severity=ERROR,
                message=(
                    f"{label} executes on {execution.date()} before its signal on "
                    f"{signal.date()}"
                ),
            )
        )
    elif execution == signal and not allow_same_close:
        report.add(
            LeakageFinding(
                code="execution_timing",
                severity=ERROR,
                message=(
                    f"{label} fills at the same close that produced the signal "
                    f"({signal.date()}); trade at a later observation instead"
                ),
            )
        )
    return report


def check_factor_series_completeness(series: pd.Series, *, label: str) -> LeakageReport:
    """A silent gap in a monthly factor series is a defect, not a detail."""

    report = LeakageReport()
    if series.empty:
        report.add(
            LeakageFinding(
                code="series_completeness",
                severity=ERROR,
                message=f"{label} is empty",
            )
        )
        return report
    index = pd.DatetimeIndex(series.index)
    duplicated = index[index.duplicated()]
    if len(duplicated):
        report.add(
            LeakageFinding(
                code="series_completeness",
                severity=ERROR,
                message=f"{label} has {len(duplicated)} duplicated month(s)",
                evidence={"duplicates": [str(item.date()) for item in duplicated[:5]]},
            )
        )
    expected = pd.period_range(index.min().to_period("M"), index.max().to_period("M"), freq="M")
    observed = pd.PeriodIndex(index.to_period("M"), freq="M")
    missing = expected.difference(observed)
    if len(missing):
        report.add(
            LeakageFinding(
                code="series_completeness",
                severity=ERROR,
                message=f"{label} is missing {len(missing)} month(s) inside its own span",
                evidence={"missing": [str(item) for item in missing[:5]]},
            )
        )
    return report


def run_all(
    panel: PointInTimePanel,
    *,
    membership: pd.DataFrame | None = None,
    feature_specs: Sequence[Mapping[str, object]] | None = None,
    periods: Mapping[str, Sequence[str]] | None = None,
) -> LeakageReport:
    """Run every applicable check in one pass."""

    report = check_availability(panel)
    if membership is not None:
        report.extend(check_universe_is_dated(membership))
    if feature_specs is not None:
        report.extend(check_scaling_window(feature_specs))
    if periods is not None:
        report.extend(check_chronological_split(periods))
    return report


def describe(report: LeakageReport) -> str:
    lines = [f"leakage checks: {'PASS' if report.passed else 'FAIL'}"]
    for item in report.findings:
        lines.append(f"  [{item.severity.upper()}] {item.code}: {item.message}")
    if not report.findings:
        lines.append("  no findings")
    return "\n".join(lines)


def iter_findings(report: LeakageReport) -> Iterable[LeakageFinding]:
    return iter(report.findings)
