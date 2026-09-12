from dataclasses import dataclass
from decimal import Decimal

import pandas as pd

from lifecycle import (
    CashFlow,
    future_explicit_cashflows,
)
from money import ExchangeRateBook, money
from recurrence import (
    generate_recurring_cashflows,
    infer_recurring_series,
)


FORECAST_DAYS = 90


@dataclass
class ForecastResult:
    request_date: pd.Timestamp
    forecast_end: pd.Timestamp
    opening_balance: Decimal

    minimum_balance_required: Decimal
    minimum_forecast_balance: Decimal
    minimum_forecast_date: pd.Timestamp

    flows: list[CashFlow]
    daily: pd.DataFrame


def build_baseline_cashflows(
    events: pd.DataFrame,
    request_date: pd.Timestamp,
    home_currency: str,
    fx: ExchangeRateBook,
) -> list[CashFlow]:

    forecast_end = (
        request_date
        + pd.Timedelta(
            days=FORECAST_DAYS
        )
    )

    explicit = (
        future_explicit_cashflows(
            events=events,
            request_date=request_date,
            forecast_end=forecast_end,
            home_currency=home_currency,
            fx=fx,
        )
    )

    recurring_series = (
        infer_recurring_series(
            events=events,
            request_date=request_date,
            home_currency=home_currency,
            fx=fx,
        )
    )

    recurring = (
        generate_recurring_cashflows(
            series_list=recurring_series,
            request_date=request_date,
            forecast_end=forecast_end,
            explicit_flows=explicit,
        )
    )

    flows = explicit + recurring

    return sorted(
        flows,
        key=lambda flow: (
            flow.date,
            flow.source,
        ),
    )


def run_forecast(
    opening_balance,
    minimum_balance,
    request_date: pd.Timestamp,
    flows: list[CashFlow],
    additional_flows: list[
        CashFlow
    ] | None = None,
) -> ForecastResult:

    request_date = pd.Timestamp(
        request_date
    )

    forecast_end = (
        request_date
        + pd.Timedelta(
            days=FORECAST_DAYS
        )
    )

    opening_balance = money(
        opening_balance
    )

    minimum_balance = money(
        minimum_balance
    )

    all_flows = list(flows)

    if additional_flows:
        all_flows.extend(
            additional_flows
        )

    # Group cash movement by date.
    by_date: dict[
        pd.Timestamp,
        Decimal,
    ] = {}

    for flow in all_flows:

        flow_date = pd.Timestamp(
            flow.date
        ).normalize()

        by_date[flow_date] = (
            by_date.get(
                flow_date,
                Decimal("0"),
            )
            + flow.amount
        )

    balance = opening_balance

    records = []

    current = request_date.normalize()

    while current <= forecast_end.normalize():

        net_flow = by_date.get(
            current,
            Decimal("0"),
        )

        balance += net_flow

        records.append(
            {
                "date": current,
                "net_flow": float(
                    net_flow
                ),
                "balance": float(
                    balance
                ),
            }
        )

        current += pd.Timedelta(
            days=1
        )

    daily = pd.DataFrame(
        records
    )

    minimum_index = (
        daily["balance"].idxmin()
    )

    minimum_forecast_balance = (
        money(
            daily.loc[
                minimum_index,
                "balance",
            ]
        )
    )

    minimum_date = pd.Timestamp(
        daily.loc[
            minimum_index,
            "date",
        ]
    )

    return ForecastResult(
        request_date=request_date,
        forecast_end=forecast_end,
        opening_balance=opening_balance,
        minimum_balance_required=minimum_balance,
        minimum_forecast_balance=(
            minimum_forecast_balance
        ),
        minimum_forecast_date=minimum_date,
        flows=all_flows,
        daily=daily,
    )


def is_safe(
    result: ForecastResult,
) -> bool:
    return (
        result.minimum_forecast_balance
        >=
        result.minimum_balance_required
    )