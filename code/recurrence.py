from dataclasses import dataclass
from decimal import Decimal

import numpy as np
import pandas as pd

from lifecycle import CashFlow
from money import ExchangeRateBook, money


MIN_SERIES_EVENTS = 4

SUPPORTED_FIXED_CADENCES = [
    5,
    7,
    10,
    14,
    21,
]


@dataclass
class RecurringSeries:
    event_type: str
    category: str
    direction: str
    flexibility: str

    last_date: pd.Timestamp
    cadence_days: int | None

    monthly: bool

    amount: Decimal
    sample_count: int


def _median_interval_days(
    dates: list[pd.Timestamp],
) -> float | None:

    if len(dates) < 2:
        return None

    sorted_dates = sorted(dates)

    intervals = [
        (
            sorted_dates[i]
            - sorted_dates[i - 1]
        ).days
        for i in range(
            1,
            len(sorted_dates),
        )
    ]

    if not intervals:
        return None

    return float(
        np.median(intervals)
    )


def _classify_cadence(
    median_interval: float | None,
):
    if median_interval is None:
        return None, False

    # Calendar-month behaviour
    if 27 <= median_interval <= 32:
        return None, True

    best = min(
        SUPPORTED_FIXED_CADENCES,
        key=lambda x: abs(
            x - median_interval
        ),
    )

    if abs(
        best - median_interval
    ) <= 2:
        return best, False

    return None, False


def infer_recurring_series(
    events: pd.DataFrame,
    request_date: pd.Timestamp,
    home_currency: str,
    fx: ExchangeRateBook,
) -> list[RecurringSeries]:

    historical = events[
        (
            events["event_date"]
            < request_date
        )
        & (
            events["status"]
            == "settled"
        )
        & (
            events["direction"]
            != "non_cash"
        )
    ].copy()

    # Refunds, investment trades and unusual
    # lifecycle rows should not generate
    # future recurring patterns.
    historical = historical[
        historical["event_type"].isin(
            [
                "expense",
                "subscription",
                "debt_payment",
                "income",
            ]
        )
    ]

    # Keep recent history. Five occurrences are
    # usually enough to infer the generated
    # recurring pattern in this dataset.
    history_start = (
        request_date
        - pd.Timedelta(days=400)
    )

    historical = historical[
        historical["event_date"]
        >= history_start
    ]

    group_columns = [
        "event_type",
        "category",
        "direction",
        "flexibility",
    ]

    series_list: list[
        RecurringSeries
    ] = []

    for keys, group in historical.groupby(
        group_columns,
        dropna=False,
    ):

        group = (
            group
            .sort_values("event_date")
            .copy()
        )

        if len(group) < MIN_SERIES_EVENTS:
            continue

        # Use recent events only for cadence
        # estimation so old pattern changes
        # do not dominate.
        recent = group.tail(8)

        dates = list(
            recent["event_date"]
        )

        median_interval = (
            _median_interval_days(dates)
        )

        cadence_days, monthly = (
            _classify_cadence(
                median_interval
            )
        )

        if (
            cadence_days is None
            and not monthly
        ):
            continue

        event_type, category, direction, flexibility = keys

        # Use the recent median.
        #
        # This is intentionally robust to
        # one unusually expensive transaction.
        recent_amounts = []

        for _, row in group.tail(5).iterrows():

            if pd.isna(row["amount"]):
                continue

            converted = fx.convert(
                amount=row["amount"],
                rate_date=row["event_date"],
                from_currency=row["currency"],
                to_currency=home_currency,
            )

            recent_amounts.append(
                float(converted)
            )

        if not recent_amounts:
            continue

        forecast_amount = money(
            np.median(
                recent_amounts
            )
        )

        # Income requires a stricter test.
        #
        # We should not assume irregular bonuses/
        # windfalls continue forever.
        if direction == "credit":

            if category != "salary":
                continue

            recent_income = (
                np.array(
                    recent_amounts,
                    dtype=float,
                )
            )

            mean_income = (
                recent_income.mean()
            )

            if mean_income <= 0:
                continue

            cv = (
                recent_income.std()
                / mean_income
            )

            # High variation often indicates
            # bonuses or commissions rather than
            # confirmed regular salary.
            if cv > 0.20:
                continue

        series_list.append(
            RecurringSeries(
                event_type=event_type,
                category=category,
                direction=direction,
                flexibility=flexibility,
                last_date=pd.Timestamp(
                    group.iloc[-1][
                        "event_date"
                    ]
                ),
                cadence_days=cadence_days,
                monthly=monthly,
                amount=forecast_amount,
                sample_count=len(group),
            )
        )

    return series_list


def _next_month(
    date: pd.Timestamp,
) -> pd.Timestamp:
    return (
        date
        + pd.DateOffset(months=1)
    )


def generate_recurring_cashflows(
    series_list: list[RecurringSeries],
    request_date: pd.Timestamp,
    forecast_end: pd.Timestamp,
    explicit_flows: list[CashFlow],
) -> list[CashFlow]:

    generated: list[CashFlow] = []

    for series in series_list:

        if series.monthly:
            next_date = _next_month(
                series.last_date
            )

        else:
            next_date = (
                series.last_date
                + pd.Timedelta(
                    days=series.cadence_days
                )
            )

        while next_date <= forecast_end:

            if next_date >= request_date:

                # Prevent double-counting when an
                # explicit pending/scheduled event
                # already represents this occurrence.
                duplicate = any(
                    flow.category
                    == series.category
                    and abs(
                        (
                            flow.date
                            - next_date
                        ).days
                    )
                    <= 2
                    for flow
                    in explicit_flows
                )

                if not duplicate:

                    amount = (
                        abs(series.amount)
                    )

                    if (
                        series.direction
                        == "debit"
                    ):
                        amount = -amount

                    generated.append(
                        CashFlow(
                            date=next_date,
                            amount=amount,
                            source=(
                                "recurring_projection"
                            ),
                            category=(
                                series.category
                            ),
                            flexibility=(
                                series.flexibility
                            ),
                        )
                    )

            if series.monthly:
                next_date = _next_month(
                    next_date
                )

            else:
                next_date = (
                    next_date
                    + pd.Timedelta(
                        days=series.cadence_days
                    )
                )

    return generated