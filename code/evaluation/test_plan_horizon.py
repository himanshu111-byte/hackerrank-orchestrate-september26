from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path

import pandas as pd


CODE_DIR = (
    Path(__file__)
    .resolve()
    .parents[1]
)

if str(CODE_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(CODE_DIR),
    )


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


def parse_plan(
    value,
) -> list[
    tuple[
        pd.Timestamp,
        Decimal,
    ]
]:

    if (
        value is None
        or pd.isna(value)
    ):
        return []

    text = str(
        value
    ).strip()

    if (
        not text
        or text.lower()
        in {
            "none",
            "nan",
        }
    ):
        return []

    payments = []

    for part in text.split("|"):

        date_text, amount_text = (
            part.split(
                ":",
                1,
            )
        )

        payments.append(
            (
                pd.Timestamp(
                    date_text
                ).normalize(),
                money(
                    amount_text
                ),
            )
        )

    return payments


def main():

    bundle = load_datasets()

    fx = ExchangeRateBook(
        bundle.exchange_rates
    )

    print(
        "=" * 135
    )
    print(
        "EXPECTED SAMPLE PLANS — FULL 90-DAY VS DESIRED-DATE SAFETY"
    )
    print(
        "=" * 135
    )

    full_90_safe_count = 0
    deadline_safe_count = 0
    expected_plan_count = 0

    for (
        _,
        request,
    ) in bundle.sample_requests.iterrows():

        request_id = str(
            request[
                "request_id"
            ]
        )

        expected_method = str(
            request[
                "recommended_payment_method"
            ]
        ).strip()

        payments = parse_plan(
            request[
                "payment_plan"
            ]
        )

        if not payments:
            continue

        expected_plan_count += 1

        user_id = str(
            request[
                "user_id"
            ]
        )

        request_date = (
            pd.Timestamp(
                request[
                    "request_date"
                ]
            )
            .normalize()
        )

        desired_date = (
            pd.Timestamp(
                request[
                    "desired_completion_date"
                ]
            )
            .normalize()
        )

        profile = get_profile(
            bundle.profiles,
            user_id,
        )

        events = get_user_events(
            bundle.events,
            user_id,
        )

        baseline_flows = (
            build_baseline_cashflows(
                events=events,
                request_date=(
                    request_date
                ),
                home_currency=(
                    profile[
                        "home_currency"
                    ]
                ),
                fx=fx,
                messages=(
                    bundle.messages
                ),
                user_id=(
                    user_id
                ),
                request_id=(
                    request_id
                ),
            )
        )

        payment_flows = []

        for (
            payment_date,
            payment_amount,
        ) in payments:

            payment_flows.append(
                CashFlow(
                    date=(
                        payment_date
                    ),
                    amount=-abs(
                        money(
                            payment_amount
                        )
                    ),
                    source=(
                        "expected_payment"
                    ),
                    category=(
                        "requested_expense"
                    ),
                    flexibility=(
                        "fixed"
                    ),
                )
            )

        result = run_forecast(
            opening_balance=(
                profile[
                    "current_available_balance"
                ]
            ),
            minimum_balance=(
                profile[
                    "minimum_balance_to_keep"
                ]
            ),
            request_date=(
                request_date
            ),
            flows=(
                baseline_flows
            ),
            additional_flows=(
                payment_flows
            ),
        )

        required = float(
            profile[
                "minimum_balance_to_keep"
            ]
        )

        full_90_min = float(
            result.daily[
                "balance"
            ].min()
        )

        deadline_rows = (
            result.daily[
                result.daily[
                    "date"
                ]
                <= desired_date
            ]
        )

        if deadline_rows.empty:

            deadline_min = (
                full_90_min
            )

        else:

            deadline_min = float(
                deadline_rows[
                    "balance"
                ].min()
            )

        full_90_safe = (
            full_90_min
            >= required
        )

        deadline_safe = (
            deadline_min
            >= required
        )

        if full_90_safe:
            full_90_safe_count += 1

        if deadline_safe:
            deadline_safe_count += 1

        marker = ""

        if (
            not full_90_safe
            and deadline_safe
        ):
            marker = (
                "<-- SAFE THROUGH DEADLINE ONLY"
            )

        print(
            f"{request_id:12s} "
            f"{expected_method:18s} "
            f"desired={desired_date.date()} "
            f"full90_min={full_90_min:14.2f} "
            f"deadline_min={deadline_min:14.2f} "
            f"required={required:12.2f} "
            f"90d={str(full_90_safe):5s} "
            f"deadline={str(deadline_safe):5s} "
            f"{marker}"
        )

    print()
    print(
        "=" * 135
    )

    print(
        "Expected plans:",
        expected_plan_count,
    )

    print(
        "Safe using full 90-day window:",
        full_90_safe_count,
    )

    print(
        "Safe through desired completion date:",
        deadline_safe_count,
    )


if __name__ == "__main__":
    main()