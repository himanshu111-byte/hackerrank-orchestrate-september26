from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(ROOT / "code"),
)

from data_loader import load_datasets


TARGET_REQUESTS = [
    "request_11",
    "request_04",
    "request_03",
    "request_25",
    "request_02",
    "request_01",
    "request_05",
]


def main():
    data = load_datasets()

    print("=" * 120)
    print("SALARY / INCOME STREAM DIAGNOSTIC")
    print("=" * 120)

    for request_id in TARGET_REQUESTS:

        req = data.sample_requests[
            data.sample_requests["request_id"]
            == request_id
        ].iloc[0]

        user_id = req["user_id"]
        request_date = pd.Timestamp(
            req["request_date"]
        )

        events = data.events[
            (data.events["user_id"] == user_id)
            &
            (
                data.events["direction"]
                .astype(str)
                .str.lower()
                == "credit"
            )
        ].copy()

        events["event_date"] = pd.to_datetime(
            events["event_date"],
            errors="coerce",
        )

        if "settlement_date" in events.columns:
            events["settlement_date"] = pd.to_datetime(
                events["settlement_date"],
                errors="coerce",
            )

        historical = events[
            events["event_date"]
            <= request_date
        ].copy()

        print()
        print("=" * 120)

        print(
            request_id,
            "|",
            user_id,
            "| request_date:",
            request_date.date(),
        )

        print(
            "current_balance:",
            req.get(
                "current_balance",
                "from profile",
            ),
        )

        if historical.empty:
            print("NO HISTORICAL CREDIT EVENTS")
            continue

        columns = [
            "event_id",
            "event_date",
            "settlement_date",
            "event_type",
            "category",
            "description",
            "amount",
            "currency",
            "status",
        ]

        columns = [
            c
            for c in columns
            if c in historical.columns
        ]

        print(
            historical[
                columns
            ]
            .sort_values(
                "event_date"
            )
            .tail(30)
            .to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()