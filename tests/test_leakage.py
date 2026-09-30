"""Every leakage check must catch the mistake it exists for.

A detector that has never been shown to fire is decoration.  Each test below
builds the violation deliberately and asserts the finding.
"""

from __future__ import annotations

import unittest

import pandas as pd

from ch3_fixtures import PRIOR_PERIODS, build_fundamentals, build_market
from quant_research.pit import (
    PointInTimePanel,
    availability_violations,
    check_availability,
    check_chronological_split,
    check_execution_timing,
    check_factor_series_completeness,
    check_scaling_window,
    check_universe_is_dated,
)


class AvailabilityCheckTests(unittest.TestCase):
    def panel(self, market, *, validate: bool = True) -> PointInTimePanel:
        return PointInTimePanel.from_frames(
            market,
            build_fundamentals(["000001.SZ", "000002.SZ"], PRIOR_PERIODS),
            source="test",
            data_version="test_v1",
            validate=validate,
        )

    def test_passes_a_clean_panel(self) -> None:
        report = check_availability(self.panel(build_market(["000001.SZ", "000002.SZ"], ["2000-01"])))
        self.assertTrue(report.passed)
        self.assertEqual(report.findings, [])

    def test_catches_future_information(self) -> None:
        market = build_market(["000001.SZ", "000002.SZ"], ["2000-01"])
        market.loc[0, "available_at"] = pd.Timestamp("1999-12-01")
        report = check_availability(self.panel(market, validate=False))
        self.assertFalse(report.passed)
        self.assertEqual(report.errors[0].code, "availability")
        self.assertEqual(report.errors[0].evidence["n_violations"], 1)

    def test_the_strict_constructor_refuses_the_same_panel(self) -> None:
        """Defence in depth: a leaky panel should not even come into existence."""

        from quant_research.contracts import PointInTimeValidationError

        market = build_market(["000001.SZ", "000002.SZ"], ["2000-01"])
        market.loc[0, "available_at"] = pd.Timestamp("1999-12-01")
        with self.assertRaises(PointInTimeValidationError):
            self.panel(market)


class UniverseIsDatedTests(unittest.TestCase):
    def test_a_static_universe_is_rejected(self) -> None:
        membership = pd.DataFrame({"security_id": ["000001.SZ", "000002.SZ"]})
        report = check_universe_is_dated(membership)
        self.assertFalse(report.passed)
        self.assertIn("current-constituent backfill", report.errors[0].message)

    def test_a_dated_universe_passes(self) -> None:
        membership = pd.DataFrame(
            {
                "security_id": ["000001.SZ"],
                "in_universe_date": ["2005-06-30"],
                "out_universe_date": ["2011-03-31"],
            }
        )
        self.assertTrue(check_universe_is_dated(membership).passed)


class ScalingWindowTests(unittest.TestCase):
    def test_full_sample_scaling_is_rejected(self) -> None:
        report = check_scaling_window(
            [{"name": "size_zscore", "fit_window": "full_sample"}]
        )
        self.assertFalse(report.passed)
        self.assertEqual(report.errors[0].code, "full_sample_scaling")

    def test_undeclared_window_is_rejected(self) -> None:
        report = check_scaling_window([{"name": "size_zscore"}])
        self.assertFalse(report.passed)
        self.assertEqual(report.errors[0].evidence["features"][0]["reason"], "undeclared")

    def test_rolling_and_expanding_windows_pass(self) -> None:
        report = check_scaling_window(
            [
                {"name": "a", "fit_window": "expanding"},
                {"name": "b", "fit_window": "trailing_36m"},
                {"name": "c", "fit_window": "cross_section_at_formation"},
            ]
        )
        self.assertTrue(report.passed)


class ChronologicalSplitTests(unittest.TestCase):
    def test_overlapping_periods_are_rejected(self) -> None:
        report = check_chronological_split(
            {
                "train": ["2000-01-31", "2010-12-31"],
                "validation": ["2010-06-30", "2013-12-31"],
                "test": ["2014-01-31", "2016-12-31"],
            }
        )
        self.assertFalse(report.passed)
        self.assertEqual(report.errors[0].code, "random_split")

    def test_ordered_periods_pass(self) -> None:
        report = check_chronological_split(
            {
                "train": ["2000-01-31", "2010-12-31"],
                "validation": ["2011-01-31", "2013-12-31"],
                "test": ["2014-01-31", "2016-12-31"],
            }
        )
        self.assertTrue(report.passed)

    def test_reversed_period_is_rejected(self) -> None:
        report = check_chronological_split({"train": ["2011-01-31", "2000-01-31"]})
        self.assertFalse(report.passed)

    def test_sub_periods_warn_only_when_the_convention_is_required(self) -> None:
        periods = {
            "sub_period_1": ["2000-01-31", "2008-12-31"],
            "sub_period_2": ["2009-01-31", "2016-12-31"],
        }
        strict = check_chronological_split(periods)
        relaxed = check_chronological_split(periods, require_convention=False)
        self.assertTrue(strict.passed and relaxed.passed)
        self.assertEqual(len(strict.warnings), 1)
        self.assertEqual(len(relaxed.warnings), 0)


class ExecutionTimingTests(unittest.TestCase):
    def test_same_close_fill_is_rejected(self) -> None:
        report = check_execution_timing("2024-06-28", "2024-06-28")
        self.assertFalse(report.passed)
        self.assertIn("same close", report.errors[0].message)

    def test_execution_before_the_signal_is_rejected(self) -> None:
        report = check_execution_timing("2024-06-28", "2024-06-27")
        self.assertFalse(report.passed)

    def test_later_execution_passes(self) -> None:
        self.assertTrue(check_execution_timing("2024-06-28", "2024-07-01").passed)

    def test_same_close_can_be_declared_a_research_convention(self) -> None:
        self.assertTrue(
            check_execution_timing("2024-06-28", "2024-06-28", allow_same_close=True).passed
        )


class FactorSeriesCompletenessTests(unittest.TestCase):
    def test_interior_gap_is_rejected(self) -> None:
        index = pd.to_datetime(["2000-01-31", "2000-02-29", "2000-04-30"])
        report = check_factor_series_completeness(pd.Series([0.1, 0.2, 0.3], index=index), label="MKT")
        self.assertFalse(report.passed)
        self.assertIn("missing 1 month", report.errors[0].message)

    def test_duplicate_month_is_rejected(self) -> None:
        index = pd.to_datetime(["2000-01-31", "2000-01-31"])
        report = check_factor_series_completeness(pd.Series([0.1, 0.2], index=index), label="MKT")
        self.assertFalse(report.passed)
        self.assertIn("duplicated month", report.errors[0].message)

    def test_contiguous_months_pass(self) -> None:
        index = pd.to_datetime(["2000-01-31", "2000-02-29", "2000-03-31"])
        report = check_factor_series_completeness(pd.Series([0.1, 0.2, 0.3], index=index), label="MKT")
        self.assertTrue(report.passed)


class ReportTests(unittest.TestCase):
    def test_raise_if_errors_respects_the_override(self) -> None:
        from quant_research.pit import LeakageError

        report = check_universe_is_dated(pd.DataFrame({"security_id": ["000001.SZ"]}))
        with self.assertRaises(LeakageError):
            report.raise_if_errors("test")
        report.raise_if_errors("test", allow_override=True)  # must not raise

    def test_to_dict_is_json_shaped(self) -> None:
        import json

        report = check_universe_is_dated(pd.DataFrame({"security_id": ["000001.SZ"]}))
        json.dumps(report.to_dict())

    def test_evidence_with_timestamps_still_serialises(self) -> None:
        """A report that cannot be written is worse than no report."""

        import json

        market = build_market(["000001.SZ", "000002.SZ"], ["2000-01"])
        market.loc[0, "available_at"] = pd.Timestamp("1999-12-01")
        panel = PointInTimePanel.from_frames(
            market,
            build_fundamentals(["000001.SZ", "000002.SZ"], PRIOR_PERIODS),
            source="test",
            data_version="test_v1",
            validate=False,
        )
        payload = check_availability(panel).to_dict()
        text = json.dumps(payload)
        self.assertIn("availability", text)
        self.assertIn("1999-12-01", text)


class DeliberateLeakFixtureTests(unittest.TestCase):
    """The generator ships a leaky twin; the detectors must catch it.

    This is the acceptance test for "I can prove I do not leak": a panel that
    is wrong in a known way has to be rejected, not merely frowned upon.
    """

    @classmethod
    def setUpClass(cls) -> None:
        from quant_research.synthetic import SyntheticPanelSpec, simulate_ch3_panel

        cls.bundle = simulate_ch3_panel(
            SyntheticPanelSpec(n_stocks=60, start="2005-01", end="2007-12")
        )

    def panel(self, table) -> PointInTimePanel:
        return PointInTimePanel.from_frames(
            table,
            self.bundle.fundamentals,
            source="synthetic_generator",
            data_version="synthetic_ch3_v1",
            validate=False,
        )

    def test_the_clean_twin_passes(self) -> None:
        self.assertTrue(check_availability(self.panel(self.bundle.market)).passed)

    def test_the_leaky_twin_is_caught(self) -> None:
        report = check_availability(self.panel(self.bundle.market_leaky))
        self.assertFalse(report.passed)
        finding = report.errors[0]
        self.assertEqual(finding.code, "availability")
        self.assertGreater(finding.evidence["n_violations"], 0)
        self.assertIn("examples", finding.evidence)

    def test_both_leak_kinds_are_reported(self) -> None:
        leaked = self.bundle.market_leaky
        violations = availability_violations(leaked)
        messages = " ".join(violations["violation"].unique())
        self.assertIn("observation_date > available_at", messages)
        self.assertIn("available_at > trade_date", messages)

    def test_the_strict_constructor_refuses_the_leaky_twin(self) -> None:
        from quant_research.contracts import PointInTimeValidationError

        with self.assertRaises(PointInTimeValidationError):
            PointInTimePanel.from_frames(
                self.bundle.market_leaky,
                self.bundle.fundamentals,
                source="synthetic_generator",
                data_version="synthetic_ch3_v1",
            )


class SplitNamingTests(unittest.TestCase):
    """Disjointness must not depend on what the windows are called."""

    def test_non_conventional_overlapping_periods_are_rejected(self) -> None:
        report = check_chronological_split(
            {
                "fold_a": ["2000-01-31", "2010-12-31"],
                "fold_b": ["2005-01-31", "2016-12-31"],
            },
            require_convention=False,
        )
        self.assertFalse(report.passed)
        self.assertEqual(report.errors[0].code, "random_split")

    def test_sub_periods_in_date_order_are_accepted(self) -> None:
        report = check_chronological_split(
            {
                "sub_period_2": ["2009-01-31", "2016-12-31"],
                "sub_period_1": ["2000-01-31", "2008-12-31"],
            },
            require_convention=False,
        )
        self.assertTrue(report.passed)

    def test_disjointness_is_checked_across_three_named_folds(self) -> None:
        report = check_chronological_split(
            {
                "a": ["2000-01-31", "2005-12-31"],
                "b": ["2005-06-30", "2010-12-31"],
                "c": ["2011-01-31", "2016-12-31"],
            },
            require_convention=False,
        )
        self.assertFalse(report.passed)


class ScalingDeclarationTests(unittest.TestCase):
    def test_aliases_of_full_sample_are_rejected(self) -> None:
        for window in ("full", "in_sample", "entire_sample", "whole_sample", "static_full", "ALL"):
            with self.subTest(window=window):
                self.assertFalse(
                    check_scaling_window([{"name": "x", "fit_window": window}]).passed
                )

    def test_formation_time_windows_are_accepted(self) -> None:
        for window in (
            "cross_section_at_formation",
            "latest_announcement_before_formation",
            "expanding",
            "trailing_36m",
        ):
            with self.subTest(window=window):
                self.assertTrue(
                    check_scaling_window([{"name": "x", "fit_window": window}]).passed
                )


class UniverseCheckScopeTests(unittest.TestCase):
    def test_a_backfilled_membership_table_with_dates_passes_the_structural_check(self) -> None:
        """A documented false negative, pinned down so nobody over-trusts it.

        The check answers "is membership dated at all", not "is this the real
        historical index".  A current-constituent list stamped with plausible
        dates satisfies it.  Only comparing against an independent historical
        source can catch that, which is a data-provenance problem, not a code
        problem.
        """

        backfilled = pd.DataFrame(
            {
                "security_id": ["000001.SZ", "600519.SH"],
                "in_universe_date": ["2000-01-31", "2000-01-31"],
            }
        )
        self.assertTrue(check_universe_is_dated(backfilled).passed)


if __name__ == "__main__":
    unittest.main()
