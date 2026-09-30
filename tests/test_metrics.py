import unittest

from quant_research.metrics import (
    annualized_return,
    cumulative_wealth,
    max_drawdown,
    simple_returns,
)


class MetricsTests(unittest.TestCase):
    def test_simple_returns(self) -> None:
        result = simple_returns([100, 110, 99])
        self.assertAlmostEqual(result[0], 0.1)
        self.assertAlmostEqual(result[1], -0.1)

    def test_cumulative_wealth(self) -> None:
        wealth = cumulative_wealth([0.1, -0.1])
        self.assertAlmostEqual(wealth[-1], 0.99)

    def test_annualized_monthly_return(self) -> None:
        result = annualized_return([0.01] * 12, periods_per_year=12)
        self.assertAlmostEqual(result, 1.01**12 - 1)

    def test_max_drawdown(self) -> None:
        self.assertAlmostEqual(max_drawdown([0.1, -0.2, 0.05]), -0.2)

    def test_rejects_nonpositive_price(self) -> None:
        with self.assertRaises(ValueError):
            simple_returns([100, 0])


if __name__ == "__main__":
    unittest.main()
