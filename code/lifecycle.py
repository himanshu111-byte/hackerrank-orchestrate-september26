from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

import pandas as pd

from money import ExchangeRateBook, money


@dataclass
class CashFlow:
    date: pd.Timestamp
    amount: Decimal
    source: str
    event_id: Optional[str] = None
    category: Optional[str] = None
    flexibility: Optional[str] = None

    # Positive = cash coming in
    # Negative = cash leaving
    @property
    def signed_amount(self) -> Decimal:
        return self.amount


IGNORED_STATUSES = {
    "cancelled",
    "failed",
    "unrealized",
}


def is_future_event_relevant(
    row: pd.Series,
    request_date: pd.Timestamp,
) -> bool:

    status = str(row["status"]).lower()
    direction = str(row["direction"]).lower()

    if status in IGNORED_STATUSES:
        return False

    event_date = pd.Timestamp(
        row["event_date"]
    )

    if event_date < request_date:
        return False

    # Problem statement:
    # ignore pending credits.
    if (
        direction == "credit"
        and status == "pending"
    ):
        return False

    # Non-cash investment valuations must
    # never alter available cash.
    if direction == "non_cash":
        return False

    return True


def future_explicit_cashflows(
    events: pd.DataFrame,
    request_date: pd.Timestamp,
    forecast_end: pd.Timestamp,
    home_currency: str,
    fx: ExchangeRateBook,
) -> list[CashFlow]:

    flows: list[CashFlow] = []

    for _, row in events.iterrows():

        if not is_future_event_relevant(
            row,
            request_date,
        ):
            continue

        event_date = pd.Timestamp(
            row["event_date"]
        )

        if event_date > forecast_end:
            continue

        if pd.isna(row["amount"]):
            # Image-derived amounts will be
            # integrated in a later evidence stage.
            #
            # Do NOT interpret blank as zero.
            continue

        converted = fx.convert(
            amount=row["amount"],
            rate_date=event_date,
            from_currency=row["currency"],
            to_currency=home_currency,
        )

        direction = str(
            row["direction"]
        ).lower()

        if direction == "debit":
            converted = -abs(converted)

        elif direction == "credit":
            converted = abs(converted)

        else:
            continue

        flows.append(
            CashFlow(
                date=event_date,
                amount=converted,
                source="explicit_event",
                event_id=row["event_id"],
                category=row["category"],
                flexibility=row["flexibility"],
            )
        )

    return flows