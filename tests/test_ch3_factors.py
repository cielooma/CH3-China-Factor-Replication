"""CH-3 construction: the sort, the timing and the recovery of known factors."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ch3_fixtures import (
    PRIOR_PERIODS,
    build_fundamentals,
    build_market,
    exposure_matrix,
    month_ends,
    ranked_panel,
)
from quant_research.factors import Ch3Spec, build_ch3_factors, validate_ch3_result
from quant_research.pit import PointInTimePanel
from quant_research.synthetic import SyntheticPanelSpec, simulate_ch3_panel

ROOT = Path(__file__).resolve().parents[1]
CH3_PORTFOLIOS = ("SV", "SN", "SG", "BV", "BN", "BG")


def panel_from(market: pd.DataFrame, fundamentals: pd.DataFrame) -> PointInTimePanel:
    return PointInTimePanel.from_frames(
        market, fundamentals, source="test", data_version="test_v1"
    )


class UniverseScreenTests(unittest.TestCase):
    def setUp(self) -> None:
        market, fundamentals = ranked_panel(n_securities=60)
        self.market = market
        self.result = build_ch3_factors(panel_from(market, fundamentals), Ch3Spec())
        self.first_month = self.result.universe_log["formation_month"].min()
        self.log = self.result.universe_log.loc[
            self.result.universe_log["formation_month"] == self.first_month
        ]

    def test_every_stock_month_is_logged(self) -> None:
        # Four market months give three formation months: the last month has no
        # following month to earn a return in.
        self.assertEqual(len(self.result.universe_log), 60 * 3)
        self.assertEqual(
            set(self.result.universe_log["exclusion_reason"].unique()),
            {"", "smallest_30pct"},
        )

    def test_drops_exactly_the_smallest_thirty_percent(self) -> None:
        pre_cut = self.log.loc[
            self.log["in_universe"] | (self.log["exclusion_reason"] == "smallest_30pct")
        ]
        dropped = self.log.loc[self.log["exclusion_reason"] == "smallest_30pct"]
        kept = self.log.loc[self.log["in_universe"]]

        expected_size = int(np.floor(0.30 * len(pre_cut)))
        self.assertEqual(len(dropped), expected_size)

        expected = set(pre_cut.nsmallest(expected_size, "size_me")["security_id"])
        self.assertEqual(set(dropped["security_id"]), expected)

    def test_the_cut_is_monotone_in_market_cap_within_the_month(self) -> None:
        dropped = self.log.loc[self.log["exclusion_reason"] == "smallest_30pct"]
        kept = self.log.loc[self.log["in_universe"]]
        self.assertLessEqual(dropped["size_me"].max(), kept["size_me"].min())

    def test_value_is_the_top_thirty_percent_of_ep(self) -> None:
        """Groups are split positionally: 30/40/30 by count, ties included."""

        market, fundamentals = exposure_matrix(n_securities=60)
        result = build_ch3_factors(panel_from(market, fundamentals), Ch3Spec())
        first = result.factor_returns.index[0]
        eligible = result.assignments.loc[result.assignments["realization_month"] == first]
        breakpoints = result.diagnostics.loc[first, "breakpoints"]
        n = len(eligible)
        n_leg = int(np.floor(0.30 * n))

        ascending = eligible.sort_values(
            ["ep", "security_id"], ascending=[True, True], kind="mergesort"
        )
        descending = eligible.sort_values(
            ["ep", "security_id"], ascending=[False, True], kind="mergesort"
        )
        growth = set(ascending["security_id"].head(n_leg))
        value = set(descending["security_id"].head(n_leg))

        by_security = eligible.set_index("security_id")["value_group"]
        self.assertEqual({name for name in growth if by_security[name] != "G"}, set())
        self.assertEqual({name for name in value if by_security[name] != "V"}, set())
        middle = set(eligible["security_id"]) - growth - value
        self.assertEqual({name for name in middle if by_security[name] != "N"}, set())

        self.assertEqual((eligible["value_group"] == "G").sum(), n_leg)
        self.assertEqual((eligible["value_group"] == "V").sum(), n_leg)
        self.assertAlmostEqual(
            breakpoints["ep_cut_low"],
            float(eligible.loc[eligible["security_id"].isin(growth), "ep"].max()),
            places=10,
        )
        self.assertAlmostEqual(
            breakpoints["ep_cut_high"],
            float(eligible.loc[eligible["security_id"].isin(value), "ep"].min()),
            places=10,
        )

    def test_the_middle_value_group_is_labelled_n_never_m(self) -> None:
        """N avoids a semantic collision with the B/M control's middle group."""

        market, fundamentals = exposure_matrix(n_securities=60)
        result = build_ch3_factors(panel_from(market, fundamentals), Ch3Spec())
        self.assertEqual(set(result.assignments["value_group"]), {"V", "N", "G"})
        self.assertEqual(
            set(result.assignments["value_group"]), set(result.assignments["ep_group"])
        )
        self.assertTrue(
            set(result.assignments["bm_group"].dropna()).issubset({"H", "M", "L"})
        )
        self.assertTrue(
            all(
                str(label).startswith("FF_")
                for label in result.assignments["bm_portfolio"].dropna().unique()
            )
        )

    def test_tied_market_caps_do_not_collapse_the_split(self) -> None:
        """A value<=median rule would put a whole tied block on one side."""

        securities = [f"{index:04d}.SZ" for index in range(200)]
        caps = {
            security: (1.0 if rank < 50 else 5.0 if rank < 150 else 9.0)
            for rank, security in enumerate(securities)
        }
        periods = ["2002-01", "2002-02", "2002-03"]
        market = build_market(securities, periods, market_caps=caps)
        fundamentals = build_fundamentals(
            securities,
            PRIOR_PERIODS + periods,
            earnings={security: float(rank + 1) for rank, security in enumerate(securities)},
        )
        result = build_ch3_factors(panel_from(market, fundamentals), Ch3Spec())
        assignments = result.assignments.loc[
            result.assignments["realization_month"] == result.factor_returns.index[0]
        ]
        small = assignments.loc[assignments["size_group"] == "S"]
        big = assignments.loc[assignments["size_group"] == "B"]
        self.assertLessEqual(abs(len(small) - len(big)), 1)
        self.assertEqual(len(small) + len(big), len(assignments))

    def test_an_empty_leg_makes_the_factor_missing_rather_than_plausible(self) -> None:
        """Dropping a missing leg silently reweights the survivors."""

        from quant_research.factors.ch3 import _mean_of

        complete = {"SV": 0.10, "SN": 0.02, "SG": 0.00, "BV": 0.04, "BN": 0.01, "BG": -0.02}
        self.assertAlmostEqual(
            _mean_of(complete, ("SV", "SN", "SG")) - _mean_of(complete, ("BV", "BN", "BG")),
            0.03,
            places=10,
        )
        missing_leg = {key: value for key, value in complete.items() if key != "SG"}
        self.assertTrue(
            np.isnan(
                _mean_of(missing_leg, ("SV", "SN", "SG"))
                - _mean_of(missing_leg, ("BV", "BN", "BG"))
            )
        )
        self.assertTrue(np.isnan(_mean_of(complete, ("SV", "SN", "SX"))))

    def test_size_groups_split_at_the_median_of_the_survivors(self) -> None:
        assignments = self.result.assignments.loc[
            self.result.assignments["formation_month"] == self.first_month
        ]
        small = assignments.loc[assignments["size_group"] == "S"]
        big = assignments.loc[assignments["size_group"] == "B"]
        # A count split at the median: half the survivors, ties included.
        self.assertEqual(len(small), int(np.floor(0.5 * len(assignments))))
        self.assertLessEqual(abs(len(small) - len(big)), 1)
        self.assertLessEqual(small["size_me"].max(), big["size_me"].min())
        breakpoints = self.result.diagnostics.loc[self.result.factor_returns.index[0], "breakpoints"]
        self.assertAlmostEqual(
            breakpoints["size_cut_me"], float(small["size_me"].max()), places=10
        )

    def test_negative_earnings_are_kept_and_sorted_into_growth(self) -> None:
        market, fundamentals = exposure_matrix(n_securities=60)
        biggest = sorted(market["security_id"].unique())[-6:]
        fundamentals = fundamentals.copy()
        fundamentals.loc[fundamentals["security_id"].isin(biggest), "earnings_ttm"] = -500.0

        result = build_ch3_factors(panel_from(market, fundamentals), Ch3Spec())
        negatives = result.assignments.loc[result.assignments["ep"] < 0]
        self.assertGreater(len(negatives), 0)
        self.assertEqual(set(negatives["value_group"]), {"G"})

    def test_financial_firms_are_kept_by_default(self) -> None:
        market, fundamentals = ranked_panel(n_securities=60)
        market.loc[market["security_id"] == "0000.SZ", "is_financial"] = True
        result = build_ch3_factors(panel_from(market, fundamentals), Ch3Spec())
        self.assertIn(
            "0000.SZ",
            set(result.assignments.loc[result.assignments["formation_month"] == result.universe_log["formation_month"].min(), "security_id"])
            | set(result.universe_log.loc[result.universe_log["exclusion_reason"] == "smallest_30pct", "security_id"]),
        )
        self.assertNotIn("financial_firm", set(result.universe_log["exclusion_reason"]))

    def test_short_listing_history_is_excluded_when_required(self) -> None:
        market, fundamentals = ranked_panel(n_securities=60)
        market.loc[market["security_id"] == "0000.SZ", "listing_months"] = 3
        result = build_ch3_factors(panel_from(market, fundamentals), Ch3Spec())
        reasons = result.universe_log.loc[
            result.universe_log["security_id"] == "0000.SZ", "exclusion_reason"
        ]
        self.assertEqual(set(reasons), {"insufficient_listing_history"})

    def test_thin_trading_activity_is_excluded(self) -> None:
        market, fundamentals = ranked_panel(n_securities=60)
        market.loc[market["security_id"] == "0001.SZ", "trading_days_12m"] = 80
        result = build_ch3_factors(panel_from(market, fundamentals), Ch3Spec())
        reasons = result.universe_log.loc[
            result.universe_log["security_id"] == "0001.SZ", "exclusion_reason"
        ]
        self.assertEqual(set(reasons), {"insufficient_trading_history"})


class FactorFormulaTests(unittest.TestCase):
    def setUp(self) -> None:
        market, fundamentals = exposure_matrix(n_securities=60)
        self.result = build_ch3_factors(panel_from(market, fundamentals), Ch3Spec())
        self.portfolios = (
            self.result.portfolio_returns.pivot(
                index="realization_month", columns="portfolio", values="return"
            )
        )

    def test_smb_averages_all_three_ep_legs_on_each_side(self) -> None:
        expected = self.portfolios[["SV", "SN", "SG"]].mean(axis=1) - self.portfolios[
            ["BV", "BN", "BG"]
        ].mean(axis=1)
        pd.testing.assert_series_equal(
            self.result.factor_returns["SMB"], expected, check_names=False
        )

    def test_vmg_is_value_minus_growth(self) -> None:
        expected = self.portfolios[["SV", "BV"]].mean(axis=1) - self.portfolios[
            ["SG", "BG"]
        ].mean(axis=1)
        pd.testing.assert_series_equal(
            self.result.factor_returns["VMG"], expected, check_names=False
        )

    def test_the_bm_control_builds_its_own_size_factor(self) -> None:
        # The B/M portfolios are recorded with an FF_ prefix so that a reader can
        # never confuse the CH-3 SMB with the Fama-French one.
        expected = self.portfolios[["FF_SH", "FF_SM", "FF_SL"]].mean(axis=1) - self.portfolios[
            ["FF_BH", "FF_BM", "FF_BL"]
        ].mean(axis=1)
        pd.testing.assert_series_equal(
            self.result.factor_returns["SMB_FF"], expected, check_names=False
        )
        expected_hml = self.portfolios[["FF_SH", "FF_BH"]].mean(axis=1) - self.portfolios[
            ["FF_SL", "FF_BL"]
        ].mean(axis=1)
        pd.testing.assert_series_equal(
            self.result.factor_returns["HML_BM"], expected_hml, check_names=False
        )

    def test_all_six_ch3_cells_and_all_six_bm_cells_are_populated(self) -> None:
        self.assertTrue(
            set(CH3_PORTFOLIOS).issubset(set(self.result.portfolio_returns["portfolio"]))
        )
        self.assertTrue(
            {f"FF_{name}" for name in ("SH", "SM", "SL", "BH", "BM", "BL")}.issubset(
                set(self.result.portfolio_returns["portfolio"])
            )
        )

    def test_mkt_is_the_excess_return_over_the_deposit_rate(self) -> None:
        market, fundamentals = ranked_panel(n_securities=60)
        market["rf_1y_deposit"] = 3.0
        result = build_ch3_factors(panel_from(market, fundamentals), Ch3Spec())
        monthly_rate = (1.03) ** (1 / 12) - 1
        gross = result.factor_returns["MKT"] + monthly_rate
        # Every stock returns 1% a month, so the market return is 1% a month.
        np.testing.assert_allclose(gross.to_numpy(), 0.01, atol=1e-12)

    def test_market_factor_uses_the_surviving_universe(self) -> None:
        market, fundamentals = ranked_panel(n_securities=60)
        # Give the dropped microcaps a wild return; MKT must not see it.
        dropped = market["security_id"].isin([f"{i:04d}.SZ" for i in range(18)])
        market.loc[dropped, "ret_1m"] = -0.90
        result = build_ch3_factors(panel_from(market, fundamentals), Ch3Spec())
        np.testing.assert_allclose(result.factor_returns["MKT"].to_numpy(), 0.01, atol=1e-12)


class TimingTests(unittest.TestCase):
    def setUp(self) -> None:
        market, fundamentals = exposure_matrix(n_securities=60, months=5)
        market = market.copy()
        stamps = sorted(market["observation_date"].unique())
        self.spike_month = stamps[2]
        # A mid-cap stock, so the smallest-30% screen does not remove it.
        self.spike_security = "0029.SZ"
        market.loc[
            (market["security_id"] == self.spike_security)
            & (market["observation_date"] == self.spike_month),
            "ret_1m",
        ] = 5.0
        self.market = market
        self.result = build_ch3_factors(panel_from(market, fundamentals), Ch3Spec())

    def test_every_factor_row_is_dated_by_the_month_it_earns(self) -> None:
        factors = self.result.factor_returns
        formation = pd.to_datetime(factors["formation_month"])
        realized = pd.DatetimeIndex(factors.index)
        lag = (realized.year * 12 + realized.month) - (formation.dt.year * 12 + formation.dt.month)
        self.assertTrue((lag == 1).all())

    def test_assignments_carry_the_following_months_return(self) -> None:
        merged = self.result.assignments.merge(
            self.market[["security_id", "observation_date", "ret_1m"]],
            left_on=["security_id", "realization_month"],
            right_on=["security_id", "observation_date"],
        )
        np.testing.assert_allclose(
            merged["ret_next_1m"].to_numpy(), merged["ret_1m"].to_numpy()
        )

    def test_a_return_shock_lands_in_the_month_after_its_formation(self) -> None:
        before = self.result.assignments.loc[
            self.result.assignments["realization_month"] == self.spike_month
        ]
        shock = before.loc[before["security_id"] == self.spike_security]
        self.assertAlmostEqual(float(shock["ret_next_1m"].iloc[0]), 5.0)

        after = self.result.assignments.loc[
            self.result.assignments["formation_month"] == self.spike_month
        ]
        carried = after.loc[after["security_id"] == self.spike_security]
        self.assertNotAlmostEqual(float(carried["ret_next_1m"].iloc[0]), 5.0)

    def test_quality_checks_pass(self) -> None:
        self.assertEqual(validate_ch3_result(self.result), [])


class PointInTimeEarningsTests(unittest.TestCase):
    """A fiscal period that has not been announced must not enter the sort."""

    def build(self) -> pd.DataFrame:
        securities = ["000001.SZ", "000002.SZ", "000003.SZ"]
        market = build_market(
            securities,
            [f"2000-{month:02d}" for month in range(1, 10)],
            market_caps={securities[0]: 100.0, securities[1]: 500.0, securities[2]: 900.0},
        )
        fundamentals = pd.DataFrame(
            [
                {
                    "security_id": security,
                    "observation_date": "1999-09-30",
                    "available_at": "1999-11-30",
                    "trade_date": "1999-11-30",
                    "source": "test",
                    "data_version": "test_v1",
                    "earnings_ttm": 10.0,
                    "book_equity": 100.0,
                }
                for security in securities
            ]
            + [
                {
                    "security_id": security,
                    "observation_date": "2000-06-30",
                    "available_at": "2000-08-20",
                    "trade_date": "2000-08-20",
                    "source": "test",
                    "data_version": "test_v1",
                    "earnings_ttm": 9999.0,
                    "book_equity": 100.0,
                }
                for security in securities
            ]
        )
        return build_ch3_factors(panel_from(market, fundamentals), Ch3Spec()).assignments

    def test_the_unannounced_quarter_is_invisible_before_its_release(self) -> None:
        assignments = self.build()
        before = assignments.loc[assignments["formation_month"] < pd.Timestamp("2000-08-20")]
        self.assertFalse(before.empty)
        np.testing.assert_allclose(
            (before["ep"] * before["me"]).to_numpy(), 10.0, rtol=1e-12
        )

    def test_the_announced_quarter_is_used_after_its_release(self) -> None:
        assignments = self.build()
        after = assignments.loc[assignments["formation_month"] > pd.Timestamp("2000-08-20")]
        self.assertFalse(after.empty)
        np.testing.assert_allclose(
            (after["ep"] * after["me"]).to_numpy(), 9999.0, rtol=1e-12
        )


class SyntheticRecoveryTests(unittest.TestCase):
    """End to end: recover factors that were embedded in the data on purpose."""

    @classmethod
    def setUpClass(cls) -> None:
        bundle = simulate_ch3_panel(SyntheticPanelSpec())
        cls.bundle = bundle
        cls.result = build_ch3_factors(
            PointInTimePanel.from_frames(
                bundle.market,
                bundle.fundamentals,
                source="synthetic_generator",
                data_version="synthetic_ch3_v1",
            ),
            Ch3Spec(),
        )

    def test_the_sample_is_the_published_two_hundred_and_four_months(self) -> None:
        self.assertEqual(len(self.result.factor_returns), 204)
        self.assertEqual(
            str(self.result.factor_returns.index.min().date()), "2000-01-31"
        )
        self.assertEqual(str(self.result.factor_returns.index.max().date()), "2016-12-31")

    def test_quality_checks_pass(self) -> None:
        self.assertEqual(validate_ch3_result(self.result), [])

    def test_portfolios_are_wide_enough(self) -> None:
        self.assertFalse(bool(self.result.diagnostics["below_min_width"].any()))

    def test_every_factor_tracks_its_latent_counterpart(self) -> None:
        reference = self.bundle.author_factors.loc[self.result.factor_returns.index]
        for factor in ("MKT", "SMB", "VMG", "HML_BM", "SMB_FF"):
            with self.subTest(factor=factor):
                joined = pd.concat(
                    [self.result.factor_returns[factor], reference[factor]], axis=1
                ).dropna()
                self.assertGreater(float(joined.corr().iloc[0, 1]), 0.90)


class FrozenConfigTests(unittest.TestCase):
    def test_shipped_configs_load_and_fingerprint_stably(self) -> None:
        for name in ("ch3_reproduction_v1.json", "ch3_reference_v1.json"):
            with self.subTest(config=name):
                payload = json.loads((ROOT / "configs" / name).read_text(encoding="utf-8"))
                spec = Ch3Spec.from_dict(payload["ch3"])
                spec.validate()
                self.assertEqual(spec.fingerprint(), spec.fingerprint())
                self.assertEqual(spec.sample_start, "2000-01-31")
                self.assertEqual(spec.sample_end, "2016-12-31")
                self.assertFalse(spec.exclude_financials)
                self.assertEqual(spec.market_universe, "eligible")

    def test_a_typo_in_the_config_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown CH-3 config keys"):
            Ch3Spec.from_dict({"sample_start": "2000-01-31", "exclude_smallest_fractoin": 0.3})

    def test_invalid_breakpoints_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Ch3Spec(value_breakpoints=(0.7, 0.3)).validate()


class RiskFreeTests(unittest.TestCase):
    """MKT is an excess return over the rate of the month it earns in."""

    def build(self) -> tuple:
        securities = [f"{index:04d}.SZ" for index in range(60)]
        periods = ["2003-01", "2003-02", "2003-03", "2003-04"]
        market = build_market(securities, periods, market_caps={s: float(100 + i) for i, s in enumerate(securities)})
        stamps = month_ends(periods)
        # A deliberately huge step in February so the two conventions cannot be
        # confused with each other.
        for stamp, rate in zip(stamps, [1.98, 12.0, 1.98, 1.98]):
            market.loc[market["observation_date"] == stamp, "rf_1y_deposit"] = rate
        fundamentals = build_fundamentals(
            securities,
            PRIOR_PERIODS + periods,
            earnings={s: float(i + 1) for i, s in enumerate(securities)},
        )
        return build_ch3_factors(panel_from(market, fundamentals), Ch3Spec()), stamps

    def test_the_realisation_months_rate_is_used(self) -> None:
        result, stamps = self.build()
        february = pd.Timestamp(stamps[1])
        row = result.factor_returns.loc[february]
        self.assertEqual(pd.Timestamp(row["formation_month"]), pd.Timestamp(stamps[0]))
        expected = 0.01 - ((1.12) ** (1 / 12) - 1)
        self.assertAlmostEqual(float(row["MKT"]), expected, places=12)
        # The formation month's rate would have given a visibly different answer.
        wrong = 0.01 - ((1.0198) ** (1 / 12) - 1)
        self.assertGreater(abs(float(row["MKT"]) - wrong), 0.005)

    def test_the_source_of_the_rate_is_recorded(self) -> None:
        result, _ = self.build()
        self.assertEqual(set(result.diagnostics["risk_free_source"]), {"panel"})

    def test_a_missing_rate_column_is_reported_rather_than_hidden(self) -> None:
        market, fundamentals = ranked_panel(n_securities=60)
        market = market.drop(columns=["rf_1y_deposit"])
        spec = Ch3Spec(risk_free_field="rf_1y_deposit")
        result = build_ch3_factors(panel_from(market, fundamentals), spec)
        self.assertEqual(set(result.diagnostics["risk_free_source"]), {"scalar_fallback"})
        self.assertTrue(any("risk-free field" in note for note in result.notes))


class ScreenOrderTests(unittest.TestCase):
    def build(self, size_cut_first: bool) -> pd.DataFrame:
        securities = [f"{index:04d}.SZ" for index in range(60)]
        caps = {s: float(10 * (i + 1)) for i, s in enumerate(securities)}
        earnings = {s: 500.0 for s in securities}
        # Ten names have no announced earnings, spread across the size range.
        for index in range(0, 60, 6):
            earnings[securities[index]] = float("nan")
        periods = ["2004-01", "2004-02"]
        market = build_market(securities, periods, market_caps=caps)
        fundamentals = build_fundamentals(securities, PRIOR_PERIODS + periods, earnings=earnings)
        result = build_ch3_factors(
            panel_from(market, fundamentals), Ch3Spec(size_cut_first=size_cut_first)
        )
        return result.diagnostics

    def test_cutting_first_drops_more_names_than_cutting_last(self) -> None:
        last = self.build(size_cut_first=False)
        first = self.build(size_cut_first=True)
        self.assertLess(
            int(last["n_dropped_smallest"].iloc[0]), int(first["n_dropped_smallest"].iloc[0])
        )

    def test_the_choice_is_announced_in_the_notes(self) -> None:
        market, fundamentals = ranked_panel(n_securities=60)
        for flag, expected in ((False, "LAST"), (True, "FIRST")):
            with self.subTest(size_cut_first=flag):
                result = build_ch3_factors(
                    panel_from(market, fundamentals), Ch3Spec(size_cut_first=flag)
                )
                self.assertTrue(any(expected in note for note in result.notes))


class LaggedMarketCapTests(unittest.TestCase):
    def test_a_missing_month_does_not_supply_a_stale_market_cap(self) -> None:
        securities = [f"{index:04d}.SZ" for index in range(40)]
        periods = ["2003-01", "2003-02", "2003-03", "2003-06", "2003-07"]
        market = build_market(securities, periods, market_caps={s: float(100 + i) for i, s in enumerate(securities)})
        fundamentals = build_fundamentals(
            securities,
            PRIOR_PERIODS + periods,
            earnings={s: float(i + 1) for i, s in enumerate(securities)},
        )
        result = build_ch3_factors(
            panel_from(market, fundamentals), Ch3Spec(market_cap_lag_months=1)
        )
        # 2003-06 sits three months after the previous observation.  A naive
        # shift would hand the sort a stale April cap; the gap must be treated
        # as a missing market cap instead.
        june = result.universe_log.loc[
            result.universe_log["formation_month"] == pd.Timestamp("2003-06-30")
        ]
        self.assertFalse(june.empty)
        self.assertEqual(set(june["exclusion_reason"]), {"invalid_market_cap"})

    def test_a_contiguous_history_supplies_the_lagged_cap(self) -> None:
        securities = [f"{index:04d}.SZ" for index in range(40)]
        periods = ["2003-01", "2003-02", "2003-03"]
        market = build_market(securities, periods)
        for step, stamp in enumerate(month_ends(periods)):
            market.loc[market["observation_date"] == stamp, "me"] = [
                100.0 * (step + 1) + index for index in range(40)
            ]
        fundamentals = build_fundamentals(
            securities,
            PRIOR_PERIODS + periods,
            earnings={s: float(i + 1) for i, s in enumerate(securities)},
        )
        result = build_ch3_factors(
            panel_from(market, fundamentals), Ch3Spec(market_cap_lag_months=1)
        )
        # The first factor observation is formed at 2003-02 and must therefore
        # sort on the January market cap.
        formed = result.assignments.loc[
            result.assignments["realization_month"] == result.factor_returns.index[0]
        ]
        expected = {security: 100.0 + index for index, security in enumerate(securities)}
        for row in formed.itertuples():
            self.assertAlmostEqual(row.size_me, expected[row.security_id], places=10)
            self.assertAlmostEqual(row.me, expected[row.security_id] + 100.0, places=10)


class BookEquityScreenTests(unittest.TestCase):
    def test_non_positive_book_equity_is_counted_not_hidden(self) -> None:
        market, fundamentals = exposure_matrix(n_securities=60)
        fundamentals = fundamentals.copy()
        # Mid-cap names, so the smallest-30% cut does not remove them first.
        negative = sorted(market["security_id"].unique())[30:36]
        fundamentals.loc[fundamentals["security_id"].isin(negative), "book_equity"] = -10.0
        result = build_ch3_factors(panel_from(market, fundamentals), Ch3Spec())
        self.assertGreater(int(result.diagnostics["n_bm_unavailable"].sum()), 0)


if __name__ == "__main__":
    unittest.main()
