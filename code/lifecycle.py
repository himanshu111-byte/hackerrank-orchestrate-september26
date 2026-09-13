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

    # Optional lower bound supplied by the source financial event.
    #
    # This is carried forward into recurring projections so that
    # spending-change logic can produce:
    #
    #   reduce_to:<event_id>:<minimum_allowed_amount>
    #
    # when the dataset provides such a constraint.
    minimum_allowed_amount: Optional[Decimal] = None

    @property
    def signed_amount(self) -> Decimal:
        """
        Positive = cash coming in.
        Negative = cash leaving.
        """
        return self.amount


IGNORED_STATUSES = {
    "cancelled",
    "failed",
    "unrealized",
}


def get_cash_date(
    row: pd.Series,
) -> pd.Timestamp:
    """
    Return the date on which cash actually affects the balance.

    settlement_date takes precedence over event_date when present.
    """

    event_date = pd.Timestamp(
        row["event_date"]
    )

    if (
        "settlement_date" in row.index
        and pd.notna(
            row["settlement_date"]
        )
    ):
        return pd.Timestamp(
            row["settlement_date"]
        )

    return event_date


def is_future_event_relevant(
    row: pd.Series,
    request_date: pd.Timestamp,
) -> bool:
    """
    Determine whether an explicit financial event belongs in the
    forward-looking cash forecast.
    """

    status = str(
        row["status"]
    ).strip().lower()

    direction = str(
        row["direction"]
    ).strip().lower()

    if status in IGNORED_STATUSES:
        return False

    cash_date = get_cash_date(
        row
    )

    # Anything whose cash effect happened earlier should already be
    # reflected in current_available_balance.
    if cash_date < request_date:
        return False

    # Pending credits cannot safely be relied on.
    if (
        direction == "credit"
        and status == "pending"
    ):
        return False

    # Valuations are not spendable cash.
    if direction == "non_cash":
        return False

    return True


def _converted_minimum_allowed_amount(
    row: pd.Series,
    cash_date: pd.Timestamp,
    home_currency: str,
    fx: ExchangeRateBook,
) -> Decimal | None:
    """
    Read minimum_allowed_amount when the dataset provides it.

    Convert it into home_currency using the same dated FX convention
    as the underlying financial event.

    If the column is absent or blank, return None.
    """

    if (
        "minimum_allowed_amount"
        not in row.index
    ):
        return None

    raw_value = row[
        "minimum_allowed_amount"
    ]

    if pd.isna(
        raw_value
    ):
        return None

    try:
        converted = fx.convert(
            amount=raw_value,
            rate_date=cash_date,
            from_currency=row["currency"],
            to_currency=home_currency,
        )
    except Exception:
        return None

    return abs(
        money(
            converted
        )
    )


def future_explicit_cashflows(
    events: pd.DataFrame,
    request_date: pd.Timestamp,
    forecast_end: pd.Timestamp,
    home_currency: str,
    fx: ExchangeRateBook,
) -> list[CashFlow]:
    """
    Convert relevant explicit financial events into forecast cashflows.
    """

    flows: list[CashFlow] = []

    request_date = (
        pd.Timestamp(
            request_date
        )
        .normalize()
    )

    forecast_end = (
        pd.Timestamp(
            forecast_end
        )
        .normalize()
    )

    for _, row in events.iterrows():

        if not is_future_event_relevant(
            row,
            request_date,
        ):
            continue

        cash_date = (
            get_cash_date(
                row
            )
            .normalize()
        )

        if cash_date > forecast_end:
            continue

        if pd.isna(
            row["amount"]
        ):
            # Image-backed values should already have been patched by
            # image_evidence.py. Never silently interpret a remaining
            # blank amount as zero.
            continue

        converted = fx.convert(
            amount=row["amount"],
            rate_date=cash_date,
            from_currency=row["currency"],
            to_currency=home_currency,
        )

        direction = str(
            row["direction"]
        ).strip().lower()

        if direction == "debit":
            converted = -abs(
                converted
            )

        elif direction == "credit":
            converted = abs(
                converted
            )

        else:
            continue

        minimum_allowed_amount = (
            _converted_minimum_allowed_amount(
                row=row,
                cash_date=cash_date,
                home_currency=home_currency,
                fx=fx,
            )
        )

        flows.append(
            CashFlow(
                date=cash_date,
                amount=money(
                    converted
                ),
                source="explicit_event",
                event_id=str(
                    row["event_id"]
                ),
                category=row["category"],
                flexibility=row["flexibility"],
                minimum_allowed_amount=(
                    minimum_allowed_amount
                ),
            )
        )

    return flows