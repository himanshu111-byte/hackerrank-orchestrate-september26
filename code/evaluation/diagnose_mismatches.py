from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(ROOT / "code"),
)


from data_loader import (
    load_datasets,
    get_profile,
    get_user_events,
)
from evidence import (
    extract_message_evidence,
)


def main():

    data = load_datasets()

    baseline_path = (
        ROOT
        / "stage3_baseline_results.csv"
    )

    if not baseline_path.exists():
        raise FileNotFoundError(
            "Run stage3_baseline.py first."
        )

    results = pd.read_csv(
        baseline_path
    )

    results["abs_diff"] = (
        results["safe_diff"]
        .abs()
    )

    mismatches = (
        results[
            results["abs_diff"]
            > 0.01
        ]
        .sort_values(
            "abs_diff",
            ascending=False,
        )
    )

    print("=" * 110)
    print("PUBLIC SAMPLE MISMATCH DIAGNOSTIC")
    print("=" * 110)

    for _, result in (
        mismatches.iterrows()
    ):

        request_id = (
            result["request_id"]
        )

        req = (
            data.sample_requests[
                data.sample_requests[
                    "request_id"
                ]
                == request_id
            ].iloc[0]
        )

        user_id = req["user_id"]

        events = get_user_events(
            data.events,
            user_id,
        )

        evidence = (
            extract_message_evidence(
                data.messages,
                user_id,
                request_id,
            )
        )

        blank_events = events[
            events["amount"].isna()
        ]

        linked_events = events[
            events[
                "linked_event_id"
            ].notna()
        ]

        print()
        print("-" * 110)

        print(
            request_id,
            "| expected:",
            result["expected_safe"],
            "| predicted:",
            result["predicted_safe"],
            "| diff:",
            result["safe_diff"],
        )

        print(
            "user:",
            user_id,
        )

        print(
            "messages/evidence:",
            len(evidence),
        )

        for item in evidence:
            print(
                "  MESSAGE:",
                item.evidence_type,
                item.amount,
                item.currency,
                item.effective_date,
                item.related_event_id,
            )

        print(
            "blank image-backed events:",
            len(blank_events),
        )

        if not blank_events.empty:
            print(
                blank_events[
                    [
                        "event_id",
                        "description",
                        "category",
                        "currency",
                    ]
                ].to_string(
                    index=False
                )
            )

        print(
            "linked lifecycle rows:",
            len(linked_events),
        )

        if not linked_events.empty:
            print(
                linked_events[
                    [
                        "event_id",
                        "event_type",
                        "status",
                        "linked_event_id",
                        "amount",
                        "direction",
                    ]
                ].to_string(
                    index=False
                )
            )


if __name__ == "__main__":
    main()