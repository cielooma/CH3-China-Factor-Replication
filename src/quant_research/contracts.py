"""Point-in-time data contracts used by every research phase."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterable, Mapping, Any


REQUIRED_POINT_IN_TIME_FIELDS = (
    "security_id",
    "observation_date",
    "available_at",
    "trade_date",
    "source",
    "data_version",
)


class PointInTimeValidationError(ValueError):
    """Raised when a record violates the research time contract."""


def _as_datetime(value: date | datetime | str, field: str) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError as exc:
            raise PointInTimeValidationError(
                f"{field} must be an ISO date or datetime: {value!r}"
            ) from exc
    raise PointInTimeValidationError(
        f"{field} must be date, datetime, or ISO string; got {type(value).__name__}"
    )


@dataclass(frozen=True)
class PointInTimeRecord:
    """Minimum temporal metadata required for a research observation."""

    security_id: str
    observation_date: date | datetime | str
    available_at: date | datetime | str
    trade_date: date | datetime | str
    source: str
    data_version: str

    def validate(self) -> None:
        if not self.security_id.strip():
            raise PointInTimeValidationError("security_id cannot be empty")
        if not self.source.strip():
            raise PointInTimeValidationError("source cannot be empty")
        if not self.data_version.strip():
            raise PointInTimeValidationError("data_version cannot be empty")

        observation = _as_datetime(self.observation_date, "observation_date")
        available = _as_datetime(self.available_at, "available_at")
        trade = _as_datetime(self.trade_date, "trade_date")

        if observation > available:
            raise PointInTimeValidationError(
                "observation_date cannot be later than available_at"
            )
        if available > trade:
            raise PointInTimeValidationError(
                "available_at cannot be later than trade_date"
            )


def validate_point_in_time(records: Iterable[Mapping[str, Any]]) -> None:
    """Validate a collection before it enters a feature or backtest pipeline."""

    for index, record in enumerate(records):
        missing = [field for field in REQUIRED_POINT_IN_TIME_FIELDS if field not in record]
        if missing:
            raise PointInTimeValidationError(
                f"record {index} is missing required fields: {', '.join(missing)}"
            )
        try:
            PointInTimeRecord(
                security_id=str(record["security_id"]),
                observation_date=record["observation_date"],
                available_at=record["available_at"],
                trade_date=record["trade_date"],
                source=str(record["source"]),
                data_version=str(record["data_version"]),
            ).validate()
        except PointInTimeValidationError as exc:
            raise PointInTimeValidationError(f"record {index}: {exc}") from exc

