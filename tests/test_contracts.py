import unittest

from quant_research.contracts import (
    PointInTimeValidationError,
    validate_point_in_time,
)


class PointInTimeContractTests(unittest.TestCase):
    def test_valid_record(self) -> None:
        validate_point_in_time(
            [
                {
                    "security_id": "000001.SZ",
                    "observation_date": "2025-12-31",
                    "available_at": "2026-03-20",
                    "trade_date": "2026-03-23",
                    "source": "synthetic",
                    "data_version": "v1",
                }
            ]
        )

    def test_rejects_future_information(self) -> None:
        with self.assertRaisesRegex(
            PointInTimeValidationError,
            "available_at cannot be later than trade_date",
        ):
            validate_point_in_time(
                [
                    {
                        "security_id": "000001.SZ",
                        "observation_date": "2025-12-31",
                        "available_at": "2026-03-24",
                        "trade_date": "2026-03-23",
                        "source": "synthetic",
                        "data_version": "v1",
                    }
                ]
            )

    def test_rejects_missing_temporal_field(self) -> None:
        with self.assertRaisesRegex(PointInTimeValidationError, "available_at"):
            validate_point_in_time(
                [
                    {
                        "security_id": "000001.SZ",
                        "observation_date": "2025-12-31",
                        "trade_date": "2026-03-23",
                        "source": "synthetic",
                        "data_version": "v1",
                    }
                ]
            )


if __name__ == "__main__":
    unittest.main()

