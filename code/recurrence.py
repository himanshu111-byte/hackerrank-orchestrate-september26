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

    # ---------------------------------------------------------
    # SOURCE EVENT
    # ---------------------------------------------------------
    #
    # A recurring projection must retain the historical event that
    # represents the current recurring series.
    #
    # This allows the decision engine to produce valid challenge
    # instructions such as:
    #
    #     stop:event_476
    #     reduce_to:event_989:665950
    #
    # We use the most recent settled event from the inferred series
    # as its authoritative source event.
    # ---------------------------------------------------------
    source_event_id: str

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

    sorted_dates = sorted(
        dates
    )

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
        np.median(
            intervals
        )
    )


def _classify_cadence(
    median_interval: float | None,
):
    if median_interval is None:
        return None, False

    # Calendar-month behaviour.
    if (
        27
        <= median_interval
        <= 32
    ):
        return None, True

    best = min(
        SUPPORTED_FIXED_CADENCES,
        key=lambda x: abs(
            x
            - median_interval
        ),
    )

    if (
        abs(
            best
            - median_interval
        )
        <= 2
    ):
        return best, False

    return None, False


def infer_recurring_series(
    events: pd.DataFrame,
    request_date: pd.Timestamp,
    home_currency: str,
    fx: ExchangeRateBook,
) -> list[RecurringSeries]:

    request_date = (
        pd.Timestamp(
            request_date
        )
        .normalize()
    )

    historical = events[
        (
            events[
                "event_date"
            ]
            < request_date
        )
        &
        (
            events[
                "status"
            ]
            == "settled"
        )
        &
        (
            events[
                "direction"
            ]
            != "non_cash"
        )
    ].copy()

    # ---------------------------------------------------------
    # Only event types that can reasonably form recurring cash
    # obligations belong in the generic recurrence engine.
    # ---------------------------------------------------------

    historical = historical[
        historical[
            "event_type"
        ].isin(
            [
                "expense",
                "subscription",
                "debt_payment",
                "income",
            ]
        )
    ].copy()

    # ---------------------------------------------------------
    # Salary is handled separately by salary_resolver.py.
    #
    # Bonuses, commissions, arrears and base salary must not be
    # merged into one generic recurring salary stream.
    # ---------------------------------------------------------

    historical = historical[
        (
            historical[
                "category"
            ]
            .astype(str)
            .str.lower()
            != "salary"
        )
    ].copy()

    # ---------------------------------------------------------
    # Keep recent history only.
    # ---------------------------------------------------------

    history_start = (
        request_date
        - pd.Timedelta(
            days=400
        )
    )

    historical = historical[
        (
            historical[
                "event_date"
            ]
            >= history_start
        )
    ].copy()

    group_columns = [
        "event_type",
        "category",
        "direction",
        "flexibility",
    ]

    series_list: list[
        RecurringSeries
    ] = []

    for (
        keys,
        group,
    ) in historical.groupby(
        group_columns,
        dropna=False,
    ):

        group = (
            group
            .sort_values(
                "event_date"
            )
            .copy()
        )

        if (
            len(group)
            < MIN_SERIES_EVENTS
        ):
            continue

        # -----------------------------------------------------
        # Use recent observations for cadence detection so an old
        # historic cadence does not dominate a changed pattern.
        # -----------------------------------------------------

        recent = (
            group.tail(
                8
            )
        )

        dates = list(
            recent[
                "event_date"
            ]
        )

        median_interval = (
            _median_interval_days(
                dates
            )
        )

        (
            cadence_days,
            monthly,
        ) = (
            _classify_cadence(
                median_interval
            )
        )

        if (
            cadence_days
            is None
            and not monthly
        ):
            continue

        (
            event_type,
            category,
            direction,
            flexibility,
        ) = keys

        # -----------------------------------------------------
        # Estimate recurring amount from the latest five events.
        #
        # Median is intentionally resistant to one unusual purchase.
        # -----------------------------------------------------

        recent_amounts = []

        for (
            _,
            row,
        ) in (
            group
            .tail(5)
            .iterrows()
        ):

            if pd.isna(
                row[
                    "amount"
                ]
            ):
                continue

            converted = fx.convert(
                amount=(
                    row[
                        "amount"
                    ]
                ),
                rate_date=(
                    row[
                        "event_date"
                    ]
                ),
                from_currency=(
                    row[
                        "currency"
                    ]
                ),
                to_currency=(
                    home_currency
                ),
            )

            recent_amounts.append(
                float(
                    converted
                )
            )

        if not recent_amounts:
            continue

        forecast_amount = money(
            np.median(
                recent_amounts
            )
        )

        # -----------------------------------------------------
        # Generic credits are deliberately not projected.
        #
        # Guaranteed salary is handled separately. Other credits
        # such as refunds, bonuses and windfalls cannot safely be
        # assumed to recur.
        # -----------------------------------------------------

        if (
            str(
                direction
            ).lower()
            == "credit"
        ):
            continue

        latest_row = (
            group.iloc[-1]
        )

        source_event_id = str(
            latest_row[
                "event_id"
            ]
        )

        series_list.append(
            RecurringSeries(
                event_type=(
                    str(
                        event_type
                    )
                ),
                category=(
                    str(
                        category
                    )
                ),
                direction=(
                    str(
                        direction
                    )
                ),
                flexibility=(
                    str(
                        flexibility
                    )
                ),
                source_event_id=(
                    source_event_id
                ),
                last_date=(
                    pd.Timestamp(
                        latest_row[
                            "event_date"
                        ]
                    )
                    .normalize()
                ),
                cadence_days=(
                    cadence_days
                ),
                monthly=(
                    monthly
                ),
                amount=(
                    forecast_amount
                ),
                sample_count=(
                    len(
                        group
                    )
                ),
            )
        )

    return series_list


def _next_month(
    date: pd.Timestamp,
) -> pd.Timestamp:

    return (
        pd.Timestamp(
            date
        )
        + pd.DateOffset(
            months=1
        )
    )


def generate_recurring_cashflows(
    series_list: list[RecurringSeries],
    request_date: pd.Timestamp,
    forecast_end: pd.Timestamp,
    explicit_flows: list[CashFlow],
) -> list[CashFlow]:

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

    generated: list[
        CashFlow
    ] = []

    for series in series_list:

        if series.monthly:

            next_date = (
                _next_month(
                    series.last_date
                )
            )

        else:

            next_date = (
                series.last_date
                + pd.Timedelta(
                    days=(
                        series.cadence_days
                    )
                )
            )

        while (
            next_date
            <= forecast_end
        ):

            next_date = (
                pd.Timestamp(
                    next_date
                )
                .normalize()
            )

            if (
                next_date
                >= request_date
            ):

                # -------------------------------------------------
                # Avoid double counting when an explicit future
                # event already represents this occurrence.
                # -------------------------------------------------

                duplicate = any(
                    (
                        flow.category
                        == series.category
                    )
                    and
                    (
                        abs(
                            (
                                pd.Timestamp(
                                    flow.date
                                ).normalize()
                                -
                                next_date
                            ).days
                        )
                        <= 2
                    )
                    for flow
                    in explicit_flows
                )

                if not duplicate:

                    amount = abs(
                        series.amount
                    )

                    if (
                        str(
                            series.direction
                        ).lower()
                        == "debit"
                    ):
                        amount = -amount

                    generated.append(
                        CashFlow(
                            date=(
                                next_date
                            ),
                            amount=(
                                amount
                            ),
                            source=(
                                "recurring_projection"
                            ),

                            # -------------------------------------
                            # Critical addition:
                            #
                            # Preserve the historical source event
                            # for stop/reduce instructions.
                            # -------------------------------------
                            event_id=(
                                series.source_event_id
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

                next_date = (
                    _next_month(
                        next_date
                    )
                )

            else:

                next_date = (
                    next_date
                    + pd.Timedelta(
                        days=(
                            series.cadence_days
                        )
                    )
                )

    return generated