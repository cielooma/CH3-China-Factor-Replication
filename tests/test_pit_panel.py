"""The point-in-time layer must refuse anything it cannot date."""

from __future__ import annotations

import unittest

import pandas as pd

from ch3_fixtures import PRIOR_PERIODS, build_fundamentals, build_market, month_ends
from quant_research.contracts import PointInTimeValidationError
from quant_research.pit import CSVDataProvider, PointInTimePanel, availability_violations
from quant_research.pit.provider import assert_provider_protocol


class PointInTimePanelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.securities = ["000001.SZ", "000002.SZ"]
        self.market = build_market(self.securities, ["2000-01", "2000-02"])
        self.fundamentals = build_fundamentals(self.securities, PRIOR_PERIODS)

    def panel(self, market=None, fundamentals=None) -> PointInTimePanel:
        return PointInTimePanel.from_frames(
            self.market if market is None else market,
            self.fundamentals if fundamentals is None else fundamentals,
            source="test",
            data_version="test_v1",
        )

    def test_accepts_a_well_formed_panel(self) -> None:
        panel = self.panel()
        panel.assert_contracts()
        self.assertEqual(len(panel.market), 4)

    def test_rejects_missing_provenance_column(self) -> None:
        with self.assertRaisesRegex(PointInTimeValidationError, "missing required columns"):
            self.panel(market=self.market.drop(columns=["data_version"]))

    def test_rejects_duplicate_security_month(self) -> None:
        duplicated = pd.concat([self.market, self.market.head(1)], ignore_index=True)
        with self.assertRaisesRegex(PointInTimeValidationError, "duplicate rows"):
            self.panel(market=duplicated)

    def test_rejects_provenance_that_contradicts_the_panel(self) -> None:
        with self.assertRaisesRegex(PointInTimeValidationError, "declares"):
            PointInTimePanel.from_frames(
                self.market,
                self.fundamentals,
                source="a_different_source",
                data_version="test_v1",
            )

    def test_rejects_a_value_available_before_it_exists(self) -> None:
        leaky = self.market.copy()
        leaky.loc[0, "available_at"] = pd.Timestamp("1999-12-01")
        with self.assertRaisesRegex(PointInTimeValidationError, "violates the time contract"):
            self.panel(market=leaky)

    def test_rejects_trading_before_availability(self) -> None:
        leaky = self.market.copy()
        leaky.loc[1, "trade_date"] = pd.Timestamp("2000-01-01")
        violations = availability_violations(leaky)
        self.assertEqual(len(violations), 1)
        self.assertIn("trade_date", violations["violation"].iloc[0])

    def test_as_of_filters_on_availability_not_on_period(self) -> None:
        """The June quarter is dated June but only knowable in August."""

        securities = ["000001.SZ", "000002.SZ"]
        market = build_market(securities, ["2000-06", "2000-08"])
        fundamentals = build_fundamentals(securities, ["2000-06"], announcement_lag_days=60)
        panel = self.panel(market=market, fundamentals=fundamentals)

        visible_in_july = panel.as_of("2000-07-15")
        self.assertTrue(visible_in_july.fundamentals.empty)
        self.assertEqual(len(panel.as_of("2000-09-30").fundamentals), 2)

    def test_quarter_is_announced_before_the_period_ends(self) -> None:
        """A negative lag would make the value known before it exists."""

        securities = ["000001.SZ"]
        fundamentals = build_fundamentals(securities, ["2000-06"], announcement_lag_days=-5)
        with self.assertRaisesRegex(PointInTimeValidationError, "observation_date"):
            self.panel(fundamentals=fundamentals)


class AvailableFundamentalsTests(unittest.TestCase):
    """The as-of join is the single most leak-prone line in the codebase."""

    def setUp(self) -> None:
        self.securities = ["000001.SZ"]
        self.market = build_market(self.securities, ["2000-01", "2000-06", "2000-09"])
        self.fundamentals = pd.DataFrame(
            [
                {
                    "security_id": "000001.SZ",
                    "observation_date": "2000-03-31",
                    "available_at": "2000-05-15",
                    "trade_date": "2000-05-15",
                    "source": "test",
                    "data_version": "test_v1",
                    "earnings_ttm": 10.0,
                    "book_equity": 100.0,
                },
                {
                    "security_id": "000001.SZ",
                    "observation_date": "2000-06-30",
                    "available_at": "2000-08-20",
                    "trade_date": "2000-08-20",
                    "source": "test",
                    "data_version": "test_v1",
                    "earnings_ttm": 99.0,
                    "book_equity": 100.0,
                },
            ]
        )
        self.panel = PointInTimePanel.from_frames(
            self.market, self.fundamentals, source="test", data_version="test_v1"
        )

    def test_uses_the_latest_announcement_not_the_latest_period(self) -> None:
        request = pd.DataFrame(
            {
                "security_id": ["000001.SZ", "000001.SZ"],
                "formation_date": ["2000-06-30", "2000-09-30"],
            }
        )
        joined = self.panel.available_fundamentals(request).set_index("formation_date")

        # On 2000-06-30 the June quarter exists on paper but has not been
        # announced, so the March figure is the one the researcher may see.
        june = joined.loc[pd.Timestamp("2000-06-30")]
        self.assertEqual(june["earnings_ttm"], 10.0)
        self.assertEqual(june["fundamental_period_end"], pd.Timestamp("2000-03-31"))

        september = joined.loc[pd.Timestamp("2000-09-30")]
        self.assertEqual(september["earnings_ttm"], 99.0)

    def test_no_announcement_before_the_first_formation_date_is_nan(self) -> None:
        request = pd.DataFrame(
            {"security_id": ["000001.SZ"], "formation_date": ["2000-01-31"]}
        )
        joined = self.panel.available_fundamentals(request)
        self.assertTrue(pd.isna(joined["earnings_ttm"].iloc[0]))


class ProviderTests(unittest.TestCase):
    def test_provider_matches_the_frozen_protocol(self) -> None:
        assert_provider_protocol(CSVDataProvider("a.csv", "b.csv"))

    def test_provider_narrows_fields_but_keeps_provenance(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            market_path = Path(directory) / "market.csv"
            fundamentals_path = Path(directory) / "fundamentals.csv"
            build_market(["000001.SZ", "000002.SZ"], ["2000-01", "2000-02"]).to_csv(
                market_path, index=False
            )
            build_fundamentals(["000001.SZ", "000002.SZ"], PRIOR_PERIODS).to_csv(
                fundamentals_path, index=False
            )

            provider = CSVDataProvider(market_path, fundamentals_path)
            loaded = provider.load(as_of="2000-02-29", universe=["000001.SZ"], fields=["ret_1m"])

            self.assertEqual(set(loaded.market["security_id"]), {"000001.SZ"})
            for column in ("observation_date", "available_at", "trade_date", "data_version"):
                self.assertIn(column, loaded.market.columns)
            self.assertNotIn("me", loaded.market.columns)
            self.assertEqual(len(loaded.market), 2)

    def test_provider_reports_a_missing_field(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            market_path = Path(directory) / "market.csv"
            fundamentals_path = Path(directory) / "fundamentals.csv"
            build_market(["000001.SZ"], ["2000-01"]).to_csv(market_path, index=False)
            build_fundamentals(["000001.SZ"], PRIOR_PERIODS).to_csv(fundamentals_path, index=False)
            provider = CSVDataProvider(market_path, fundamentals_path)
            with self.assertRaisesRegex(KeyError, "absent from the market extract"):
                provider.load(as_of="2000-01-31", fields=["not_a_column"])


class TemporalHoleTests(unittest.TestCase):
    """Regressions for holes that let future information reach a factor."""

    def test_availability_ties_are_resolved_by_fiscal_period_not_row_order(self) -> None:
        """Two periods announced on one day must resolve the same way always."""

        securities = ["000001.SZ"]
        market = build_market(securities, ["2000-06", "2000-07"])
        early = {
            "security_id": "000001.SZ",
            "observation_date": "2000-03-31",
            "available_at": "2000-06-30",
            "trade_date": "2000-06-30",
            "source": "test",
            "data_version": "test_v1",
            "earnings_ttm": 10.0,
            "book_equity": 100.0,
        }
        later = {**early, "observation_date": "2000-05-31", "earnings_ttm": 99.0}
        request = pd.DataFrame(
            {"security_id": ["000001.SZ"], "formation_date": ["2000-06-30"]}
        )

        results = []
        for order in ([early, later], [later, early]):
            panel = PointInTimePanel.from_frames(
                market, pd.DataFrame(order), source="test", data_version="test_v1"
            )
            joined = panel.available_fundamentals(request)
            results.append(float(joined["earnings_ttm"].iloc[0]))

        self.assertEqual(results[0], results[1])
        # The later fiscal period wins; a restated annual report does not
        # shadow the newer quarter.
        self.assertEqual(results[0], 99.0)

    def test_as_of_excludes_rows_that_are_not_tradable_yet(self) -> None:
        market = build_market(["000001.SZ", "000002.SZ"], ["2000-01"])
        market.loc[0, "trade_date"] = pd.Timestamp("2016-12-30")
        panel = PointInTimePanel.from_frames(
            market,
            build_fundamentals(["000001.SZ", "000002.SZ"], PRIOR_PERIODS),
            source="test",
            data_version="test_v1",
        )
        visible = panel.as_of("2000-02-29").market
        self.assertEqual(len(visible), 1)
        self.assertEqual(visible["security_id"].iloc[0], "000002.SZ")

    def test_a_panel_built_without_validation_still_has_datetime_columns(self) -> None:
        """The diagnostic path must not crash on the data it is diagnosing."""

        market = build_market(["000001.SZ"], ["2000-01"])
        market.loc[0, "available_at"] = pd.Timestamp("1999-12-01")
        panel = PointInTimePanel.from_frames(
            market,
            build_fundamentals(["000001.SZ"], PRIOR_PERIODS),
            source="test",
            data_version="test_v1",
            validate=False,
        )
        for column in ("observation_date", "available_at", "trade_date"):
            self.assertTrue(
                pd.api.types.is_datetime64_any_dtype(panel.market[column]),
                f"{column} was left as {panel.market[column].dtype}",
            )
        self.assertTrue(panel.market["observation_date"].dt.year.eq(2000).all())

    def test_numeric_sentinel_dates_are_rejected(self) -> None:
        market = build_market(["000001.SZ"], ["2000-01"])
        market["available_at"] = 99999999
        with self.assertRaisesRegex(PointInTimeValidationError, "YYYYMMDD"):
            PointInTimePanel.from_frames(
                market,
                build_fundamentals(["000001.SZ"], PRIOR_PERIODS),
                source="test",
                data_version="test_v1",
            )

    def test_yyyymmdd_integers_are_accepted(self) -> None:
        market = build_market(["000001.SZ"], ["2000-01"])
        market["observation_date"] = 20000131
        market["available_at"] = 20000131
        market["trade_date"] = 20000131
        panel = PointInTimePanel.from_frames(
            market,
            build_fundamentals(["000001.SZ"], PRIOR_PERIODS),
            source="test",
            data_version="test_v1",
        )
        self.assertEqual(panel.market["observation_date"].iloc[0], pd.Timestamp("2000-01-31"))


if __name__ == "__main__":
    unittest.main()
