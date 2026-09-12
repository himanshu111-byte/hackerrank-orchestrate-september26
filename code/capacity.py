from decimal import Decimal

import pandas as pd

from forecast import (
    ForecastResult,
    is_safe,
    run_forecast,
)
from lifecycle import CashFlow
from money import money, round_money


def amount_safe_to_pay(
    baseline: ForecastResult,
    requested_amount,
) -> Decimal:

    requested = money(
        requested_amount
    )

    headroom = (
        baseline.minimum_forecast_balance
        -
        baseline.minimum_balance_required
    )

    safe = max(
        Decimal("0"),
        headroom,
    )

    safe = min(
        requested,
        safe,
    )

    return round_money(
        safe
    )


def test_single_payment(
    opening_balance,
    minimum_balance,
    request_date,
    baseline_flows,
    payment_date,
    payment_amount,
) -> ForecastResult:

    payment = CashFlow(
        date=pd.Timestamp(
            payment_date
        ),
        amount=-abs(
            money(
                payment_amount
            )
        ),
        source="candidate_payment",
        category="requested_expense",
    )

    return run_forecast(
        opening_balance=opening_balance,
        minimum_balance=minimum_balance,
        request_date=request_date,
        flows=baseline_flows,
        additional_flows=[
            payment
        ],
    )


def earliest_full_payment_date(
    opening_balance,
    minimum_balance,
    request_date,
    requested_amount,
    baseline_flows,
) -> pd.Timestamp | None:

    request_date = pd.Timestamp(
        request_date
    ).normalize()

    forecast_end = (
        request_date
        + pd.Timedelta(days=90)
    )

    candidate = request_date

    while candidate <= forecast_end:

        result = test_single_payment(
            opening_balance=(
                opening_balance
            ),
            minimum_balance=(
                minimum_balance
            ),
            request_date=request_date,
            baseline_flows=(
                baseline_flows
            ),
            payment_date=candidate,
            payment_amount=(
                requested_amount
            ),
        )

        if is_safe(result):
            return candidate

        candidate += pd.Timedelta(
            days=1
        )

    return None