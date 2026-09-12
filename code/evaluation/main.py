from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "dataset"

sys.path.insert(0, str(ROOT / "code"))


OUTPUT_COLUMNS = [
    "request_id",
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation",
]


def normalize(value):
    if pd.isna(value):
        return ""

    return str(value).strip()


def compare_exact(actual, expected, column):
    a = actual[column].apply(normalize)
    e = expected[column].apply(normalize)

    return a == e


def evaluate_predictions(predictions: pd.DataFrame):
    expected = pd.read_csv(
        DATASET / "sample_requests.csv"
    )[OUTPUT_COLUMNS]

    merged = expected.merge(
        predictions,
        on="request_id",
        how="left",
        suffixes=("_expected", "_actual"),
    )

    fields = [
        "affordability_status",
        "recommended_payment_method",
        "payment_plan",
        "earliest_date_for_full_payment",
        "spending_changes_needed",
    ]

    print("=" * 80)
    print("PUBLIC SAMPLE EVALUATION")
    print("=" * 80)

    # amount_safe_to_pay
    expected_amount = pd.to_numeric(
        merged["amount_safe_to_pay_expected"],
        errors="coerce",
    )

    actual_amount = pd.to_numeric(
        merged["amount_safe_to_pay_actual"],
        errors="coerce",
    )

    amount_match = (
        (expected_amount - actual_amount).abs()
        <= 0.01
    )

    print(
        f"amount_safe_to_pay: "
        f"{amount_match.mean() * 100:.2f}% "
        f"({amount_match.sum()}/{len(merged)})"
    )

    for field in fields:
        expected_col = f"{field}_expected"
        actual_col = f"{field}_actual"

        match = (
            merged[expected_col].apply(normalize)
            ==
            merged[actual_col].apply(normalize)
        )

        print(
            f"{field}: "
            f"{match.mean() * 100:.2f}% "
            f"({match.sum()}/{len(merged)})"
        )

    print("\nMISMATCHES")
    print("=" * 80)

    for _, row in merged.iterrows():
        mismatch_fields = []

        if abs(
            float(row["amount_safe_to_pay_expected"])
            -
            float(row["amount_safe_to_pay_actual"])
        ) > 0.01:
            mismatch_fields.append("amount_safe_to_pay")

        for field in fields:
            if (
                normalize(row[f"{field}_expected"])
                !=
                normalize(row[f"{field}_actual"])
            ):
                mismatch_fields.append(field)

        if mismatch_fields:
            print(
                row["request_id"],
                "=>",
                ", ".join(mismatch_fields),
            )


if __name__ == "__main__":
    print(
        "Evaluation harness ready. "
        "The prediction engine will be connected next."
    )