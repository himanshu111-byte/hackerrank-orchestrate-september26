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
    get_user_events,
    load_datasets,
)


TARGETS = {
    "request_06": {
        "event_id": "event_476",
        "category": "streaming",
    },
    "request_11": {
        "event_id": "event_989",
        "category": "dining",
    },
    "request_21": {
        "event_id": "event_1816",
        "category": "streaming",
    },
}


def main():

    bundle = load_datasets()

    for request_id, target in TARGETS.items():

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

        events = get_user_events(
            events=bundle.events,
            user_id=user_id,
        )

        category = target[
            "category"
        ]

        rows = (
            events[
                (
                    events[
                        "category"
                    ]
                    .astype(str)
                    .str.lower()
                    == category.lower()
                )
                &
                (
                    events[
                        "event_date"
                    ]
                    < request_date
                )
            ]
            .copy()
            .sort_values(
                "event_date"
            )
        )

        print()
        print(
            "=" * 120
        )

        print(
            request_id,
            "| user =",
            user_id,
            "| request_date =",
            request_date.strftime(
                "%Y-%m-%d"
            ),
        )

        print(
            "requested_amount =",
            request[
                "requested_amount"
            ],
        )

        print(
            "expected_safe =",
            request[
                "amount_safe_to_pay"
            ],
        )

        print(
            "expected_changes =",
            request[
                "spending_changes_needed"
            ],
        )

        print(
            "target_event =",
            target[
                "event_id"
            ],
        )

        print(
            "=" * 120
        )

        columns = [
            column
            for column in [
                "event_id",
                "event_date",
                "settlement_date",
                "event_type",
                "category",
                "amount",
                "currency",
                "direction",
                "flexibility",
                "status",
                "description",
            ]
            if column
            in rows.columns
        ]

        print(
            rows[
                columns
            ]
            .tail(15)
            .to_string(
                index=False
            )
        )

        if not rows.empty:

            settled = rows[
                rows[
                    "status"
                ]
                == "settled"
            ].copy()

            amounts = (
                pd.to_numeric(
                    settled[
                        "amount"
                    ],
                    errors="coerce",
                )
                .dropna()
            )

            print()

            print(
                "AMOUNT STATS"
            )

            if not amounts.empty:

                print(
                    "latest =",
                    amounts.iloc[-1],
                )

                print(
                    "last 3 median =",
                    amounts.tail(3).median(),
                )

                print(
                    "last 4 median =",
                    amounts.tail(4).median(),
                )

                print(
                    "last 5 median =",
                    amounts.tail(5).median(),
                )

                print(
                    "last 6 median =",
                    amounts.tail(6).median(),
                )

                print(
                    "overall median =",
                    amounts.median(),
                )

                print(
                    "min =",
                    amounts.min(),
                )

                print(
                    "max =",
                    amounts.max(),
                )


if __name__ == "__main__":
    main()