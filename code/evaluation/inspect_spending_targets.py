from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


CODE_DIR = Path(
    __file__
).resolve().parents[1]

if str(
    CODE_DIR
) not in sys.path:

    sys.path.insert(
        0,
        str(
            CODE_DIR
        ),
    )


from data_loader import (
    get_profile,
    get_user_events,
    load_datasets,
)

from forecast import (
    build_baseline_cashflows,
)

from money import (
    ExchangeRateBook,
)


def value(
    row,
    column,
):

    if (
        column
        not in row.index
    ):
        return ""

    result = row[
        column
    ]

    if pd.isna(
        result
    ):
        return ""

    return str(
        result
    )


def main():

    bundle = (
        load_datasets()
    )

    fx = ExchangeRateBook(
        bundle.exchange_rates
    )

    targets = [
        "request_06",
        "request_11",
        "request_21",
    ]

    for request_id in targets:

        request = (
            bundle.sample_requests[
                (
                    bundle.sample_requests[
                        "request_id"
                    ]
                    == request_id
                )
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

        profile = get_profile(
            profiles=(
                bundle.profiles
            ),
            user_id=(
                user_id
            ),
        )

        events = get_user_events(
            events=(
                bundle.events
            ),
            user_id=(
                user_id
            ),
        )

        flows = (
            build_baseline_cashflows(
                events=(
                    events
                ),
                request_date=(
                    request_date
                ),
                home_currency=(
                    str(
                        profile[
                            "home_currency"
                        ]
                    )
                ),
                fx=(
                    fx
                ),
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

        recurring = [
            flow
            for flow in flows
            if (
                flow.source
                == "recurring_projection"
            )
        ]

        print()
        print(
            "=" * 100
        )

        print(
            request_id,
            "|",
            user_id,
        )

        print(
            "=" * 100
        )

        print(
            "protected:",
            value(
                profile,
                "expense_categories_to_protect",
            ),
        )

        print(
            "reduce:",
            value(
                profile,
                "expense_categories_user_is_willing_to_reduce",
            ),
        )

        print(
            "stop:",
            value(
                profile,
                "expense_categories_user_is_willing_to_stop",
            ),
        )

        print()

        print(
            "Recurring projections:"
        )

        seen = set()

        for flow in recurring:

            key = (
                flow.event_id,
                flow.category,
                flow.flexibility,
                str(
                    flow.amount
                ),
            )

            if key in seen:
                continue

            seen.add(
                key
            )

            print(
                "event_id=",
                flow.event_id,
                "| category=",
                flow.category,
                "| flexibility=",
                flow.flexibility,
                "| amount=",
                flow.amount,
                "| first_date=",
                pd.Timestamp(
                    flow.date
                ).strftime(
                    "%Y-%m-%d"
                ),
            )


if __name__ == "__main__":
    main()