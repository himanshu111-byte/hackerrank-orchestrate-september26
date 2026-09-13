from __future__ import annotations

import sys
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

from main import (
    decide_request,
)

from money import (
    ExchangeRateBook,
)


FIELDS = [
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
]


def normalize_text(
    value,
) -> str:

    if pd.isna(value):
        return ""

    return str(
        value
    ).strip()


def normalize_money(
    value,
) -> str:

    if pd.isna(value):
        return ""

    try:
        number = float(
            value
        )

        return f"{number:.2f}"

    except (
        TypeError,
        ValueError,
    ):
        return normalize_text(
            value
        )


def normalize_date(
    value,
) -> str:

    if (
        value is None
        or pd.isna(value)
    ):
        return ""

    try:
        return (
            pd.Timestamp(
                value
            )
            .strftime(
                "%Y-%m-%d"
            )
        )

    except Exception:
        return normalize_text(
            value
        )


def normalize_field(
    field: str,
    value,
) -> str:

    if field == "amount_safe_to_pay":
        return normalize_money(
            value
        )

    if (
        field
        == "earliest_date_for_full_payment"
    ):
        return normalize_date(
            value
        )

    return normalize_text(
        value
    )


def main():

    bundle = load_datasets()

    fx = ExchangeRateBook(
        bundle.exchange_rates
    )

    comparison_rows = []

    for (
        _,
        request,
    ) in bundle.sample_requests.iterrows():

        request_id = str(
            request[
                "request_id"
            ]
        )

        user_id = str(
            request[
                "user_id"
            ]
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

        predicted = decide_request(
            request=request,
            profile=profile,
            user_events=events,
            payment_options=(
                bundle.payment_options
            ),
            messages=(
                bundle.messages
            ),
            fx=fx,
        )

        row = {
            "request_id": (
                request_id
            )
        }

        exact_count = 0

        for field in FIELDS:

            expected = normalize_field(
                field,
                request.get(
                    field,
                    "",
                ),
            )

            actual = normalize_field(
                field,
                predicted.get(
                    field,
                    "",
                ),
            )

            matched = (
                expected
                == actual
            )

            if matched:
                exact_count += 1

            row[
                f"expected_{field}"
            ] = expected

            row[
                f"predicted_{field}"
            ] = actual

            row[
                f"match_{field}"
            ] = matched

        row[
            "field_matches"
        ] = (
            f"{exact_count}/"
            f"{len(FIELDS)}"
        )

        row[
            "expected_explanation"
        ] = normalize_text(
            request.get(
                "decision_explanation",
                "",
            )
        )

        row[
            "predicted_explanation"
        ] = normalize_text(
            predicted.get(
                "decision_explanation",
                "",
            )
        )

        comparison_rows.append(
            row
        )

    results = pd.DataFrame(
        comparison_rows
    )

    print(
        "=" * 120
    )

    print(
        "FULL DECISION ENGINE — SAMPLE VALIDATION"
    )

    print(
        "=" * 120
    )

    summary_columns = [
        "request_id",
        "expected_affordability_status",
        "predicted_affordability_status",
        "expected_recommended_payment_method",
        "predicted_recommended_payment_method",
        "field_matches",
    ]

    print(
        results[
            summary_columns
        ].to_string(
            index=False
        )
    )

    print()

    print(
        "=" * 120
    )

    print(
        "FIELD ACCURACY"
    )

    print(
        "=" * 120
    )

    for field in FIELDS:

        column = (
            f"match_{field}"
        )

        correct = int(
            results[
                column
            ].sum()
        )

        total = len(
            results
        )

        print(
            f"{field:35s} "
            f"{correct:2d}/{total}"
        )

    print()

    print(
        "=" * 120
    )

    print(
        "METHOD / STATUS MISMATCHES"
    )

    print(
        "=" * 120
    )

    important_mismatch = results[
        ~(
            results[
                "match_affordability_status"
            ]
            &
            results[
                "match_recommended_payment_method"
            ]
        )
    ]

    if important_mismatch.empty:

        print(
            "None"
        )

    else:

        print(
            important_mismatch[
                [
                    "request_id",
                    "expected_affordability_status",
                    "predicted_affordability_status",
                    "expected_recommended_payment_method",
                    "predicted_recommended_payment_method",
                ]
            ].to_string(
                index=False
            )
        )

    output_path = (
        CODE_DIR
        / "evaluation"
        / "decision_sample_results.csv"
    )

    results.to_csv(
        output_path,
        index=False,
    )

    print()

    print(
        "Saved:"
    )

    print(
        output_path
    )


if __name__ == "__main__":
    main()