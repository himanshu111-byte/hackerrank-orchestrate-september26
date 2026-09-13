from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path

import pandas as pd


CODE_DIR = Path(__file__).resolve().parents[1]

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


from data_loader import (
    get_profile,
    get_user_events,
    load_datasets,
)

from forecast import (
    build_baseline_cashflows,
    run_forecast,
)

from lifecycle import CashFlow

from money import (
    ExchangeRateBook,
    money,
)


def plan_safe_through_deadline(
    opening_balance,
    minimum_balance,
    request_date,
    desired_date,
    baseline_flows,
    payment_date,
    payment_amount,
):
    request_date = pd.Timestamp(request_date).normalize()
    desired_date = pd.Timestamp(desired_date).normalize()
    payment_date = pd.Timestamp(payment_date).normalize()

    result = run_forecast(
        opening_balance=opening_balance,
        minimum_balance=minimum_balance,
        request_date=request_date,
        flows=baseline_flows,
        additional_flows=[
            CashFlow(
                date=payment_date,
                amount=-abs(money(payment_amount)),
                source="deadline_candidate_test",
                category="requested_expense",
                flexibility="fixed",
            )
        ],
    )

    relevant = result.daily[
        (result.daily["date"] >= request_date)
        & (result.daily["date"] <= desired_date)
    ]

    if relevant.empty:
        return False

    minimum_seen = money(
        str(
            relevant["balance"].min()
        )
    )

    return minimum_seen >= money(minimum_balance)


def main():

    bundle = load_datasets()
    fx = ExchangeRateBook(bundle.exchange_rates)

    for request_id in [
        "request_08",
        "request_09",
        "request_13",
    ]:

        request = bundle.sample_requests[
            bundle.sample_requests["request_id"] == request_id
        ].iloc[0]

        user_id = str(
            request["user_id"]
        )

        profile = get_profile(
            bundle.profiles,
            user_id,
        )

        events = get_user_events(
            bundle.events,
            user_id,
        )

        request_date = (
            pd.Timestamp(
                request["request_date"]
            )
            .normalize()
        )

        desired_date = (
            pd.Timestamp(
                request["desired_completion_date"]
            )
            .normalize()
        )

        requested_amount = money(
            request["requested_amount"]
        )

        baseline_flows = build_baseline_cashflows(
            events=events,
            request_date=request_date,
            home_currency=profile["home_currency"],
            fx=fx,
            messages=bundle.messages,
            user_id=user_id,
            request_id=request_id,
        )

        first_safe = None

        candidate_date = request_date

        while candidate_date <= desired_date:

            if plan_safe_through_deadline(
                opening_balance=profile[
                    "current_available_balance"
                ],
                minimum_balance=profile[
                    "minimum_balance_to_keep"
                ],
                request_date=request_date,
                desired_date=desired_date,
                baseline_flows=baseline_flows,
                payment_date=candidate_date,
                payment_amount=requested_amount,
            ):
                first_safe = candidate_date
                break

            candidate_date += pd.Timedelta(
                days=1
            )

        print(
            request_id,
            "expected_method=",
            request[
                "recommended_payment_method"
            ],
            "expected_plan=",
            request[
                "payment_plan"
            ],
            "first_deadline_safe_full=",
            (
                first_safe.strftime(
                    "%Y-%m-%d"
                )
                if first_safe is not None
                else None
            ),
        )


if __name__ == "__main__":
    main()