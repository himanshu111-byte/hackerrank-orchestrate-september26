from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


CODE_DIR = Path(
    __file__
).resolve().parents[1]

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


EXPECTED_ACTIONS = {
    "request_06": {
        "stop": [
            "event_476",
        ],
        "reduce": {},
    },

    "request_11": {
        "stop": [],
        "reduce": {
            "event_989": "665950",
        },
    },

    "request_21": {
        "stop": [
            "event_1815",
        ],
        "reduce": {
            "event_1816": "23.50",
        },
    },
}


def apply_actions(
    flows,
    stop_event_ids,
    reduce_targets,
):
    result = []

    for flow in flows:

        event_id = (
            str(flow.event_id)
            if flow.event_id is not None
            else None
        )

        # Only recurring projected spending is modified.
        if (
            flow.source
            != "recurring_projection"
            or event_id is None
        ):
            result.append(
                flow
            )
            continue

        # Stop this recurring expense completely.
        if event_id in stop_event_ids:
            continue

        # Reduce this recurring expense.
        if event_id in reduce_targets:

            target = abs(
                money(
                    reduce_targets[
                        event_id
                    ]
                )
            )

            amount = (
                -target
                if flow.amount < 0
                else target
            )

            result.append(
                CashFlow(
                    date=flow.date,
                    amount=amount,
                    source=flow.source,
                    event_id=flow.event_id,
                    category=flow.category,
                    flexibility=flow.flexibility,
                )
            )

            continue

        result.append(
            flow
        )

    return result


def main():

    bundle = load_datasets()

    fx = ExchangeRateBook(
        bundle.exchange_rates
    )

    for request_id, actions in EXPECTED_ACTIONS.items():

        request = (
            bundle.sample_requests[
                bundle.sample_requests[
                    "request_id"
                ]
                == request_id
            ]
            .iloc[0]
        )

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

        requested_amount = money(
            request[
                "requested_amount"
            ]
        )

        profile = get_profile(
            profiles=bundle.profiles,
            user_id=user_id,
        )

        events = get_user_events(
            events=bundle.events,
            user_id=user_id,
        )

        baseline_flows = (
            build_baseline_cashflows(
                events=events,
                request_date=request_date,
                home_currency=str(
                    profile[
                        "home_currency"
                    ]
                ),
                fx=fx,
                messages=bundle.messages,
                user_id=user_id,
                request_id=request_id,
            )
        )

        print()
        print(
            "=" * 100
        )

        print(
            request_id
        )

        print(
            "=" * 100
        )

        target_ids = set(
            actions[
                "stop"
            ]
        ) | set(
            actions[
                "reduce"
            ].keys()
        )

        print(
            "SOURCE EVENT ROWS"
        )

        source_rows = events[
            events[
                "event_id"
            ]
            .astype(str)
            .isin(
                target_ids
            )
        ]

        columns = [
            column
            for column in [
                "event_id",
                "event_date",
                "event_type",
                "category",
                "amount",
                "currency",
                "flexibility",
                "status",
            ]
            if column
            in source_rows.columns
        ]

        if not source_rows.empty:

            print(
                source_rows[
                    columns
                ].to_string(
                    index=False
                )
            )

        print()
        print(
            "EXPECTED ACTIONS"
        )

        print(
            "stop =",
            actions[
                "stop"
            ],
        )

        print(
            "reduce =",
            actions[
                "reduce"
            ],
        )

        # -----------------------------------------------------
        # Baseline + full payment TODAY
        # -----------------------------------------------------

        payment_flow = CashFlow(
            date=request_date,
            amount=-abs(
                requested_amount
            ),
            source=(
                "candidate_payment"
            ),
            category=(
                "requested_expense"
            ),
            flexibility=(
                "fixed"
            ),
        )

        without_changes = run_forecast(
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
            additional_flows=[
                payment_flow
            ],
        )

        modified_flows = apply_actions(
            flows=(
                baseline_flows
            ),
            stop_event_ids=set(
                actions[
                    "stop"
                ]
            ),
            reduce_targets=(
                actions[
                    "reduce"
                ]
            ),
        )

        with_changes = run_forecast(
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
                modified_flows
            ),
            additional_flows=[
                payment_flow
            ],
        )

        before_window = (
            without_changes.daily[
                (
                    without_changes.daily[
                        "date"
                    ]
                    >= request_date
                )
                &
                (
                    without_changes.daily[
                        "date"
                    ]
                    <= desired_date
                )
            ]
        )

        after_window = (
            with_changes.daily[
                (
                    with_changes.daily[
                        "date"
                    ]
                    >= request_date
                )
                &
                (
                    with_changes.daily[
                        "date"
                    ]
                    <= desired_date
                )
            ]
        )

        minimum_required = money(
            profile[
                "minimum_balance_to_keep"
            ]
        )

        before_min = money(
            str(
                before_window[
                    "balance"
                ].min()
            )
        )

        after_min = money(
            str(
                after_window[
                    "balance"
                ].min()
            )
        )

        print()
        print(
            "FULL PAYMENT TODAY"
        )

        print(
            "minimum required :",
            minimum_required,
        )

        print(
            "without changes  :",
            before_min,
        )

        print(
            "deficit before   :",
            minimum_required
            - before_min,
        )

        print(
            "with changes     :",
            after_min,
        )

        print(
            "deficit after    :",
            minimum_required
            - after_min,
        )


if __name__ == "__main__":
    main()