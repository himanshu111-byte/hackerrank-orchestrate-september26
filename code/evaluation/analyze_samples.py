from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "dataset"


def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(DATASET_DIR / name)


def fmt(value):
    if pd.isna(value):
        return ""
    return str(value)


def main():
    samples = load_csv("sample_requests.csv")
    profiles = load_csv("financial_profiles.csv")
    events = load_csv("financial_events.csv")
    options = load_csv("request_payment_options.csv")
    messages = load_csv("messages.csv")
    images = load_csv("images.csv")

    report_lines = []

    print("=" * 90)
    print("BUY OR WAIT? — PUBLIC SAMPLE ANALYSIS")
    print("=" * 90)
    print(f"Sample requests: {len(samples)}")

    print("\nAffordability status distribution:")
    print(samples["affordability_status"].value_counts())

    print("\nRecommended payment methods:")
    print(samples["recommended_payment_method"].value_counts())

    for _, req in samples.iterrows():
        request_id = req["request_id"]
        user_id = req["user_id"]

        profile_rows = profiles[profiles["user_id"] == user_id]
        user_events = events[events["user_id"] == user_id].copy()

        request_options = options[
            options["request_id"] == request_id
        ].copy()

        request_messages = messages[
            (messages["user_id"] == user_id)
            | (messages["request_id"] == request_id)
        ].copy()

        request_images = images[
            (images["user_id"] == user_id)
            | (images["request_id"] == request_id)
        ].copy()

        lines = []

        lines.append("=" * 100)
        lines.append(f"REQUEST: {request_id}")
        lines.append("=" * 100)

        lines.append("")
        lines.append("REQUEST INPUT")
        lines.append("-" * 100)

        request_fields = [
            "user_id",
            "request_date",
            "request_type",
            "requested_amount",
            "desired_completion_date",
            "allows_partial_payment",
            "request_text",
        ]

        for field in request_fields:
            lines.append(f"{field}: {fmt(req[field])}")

        lines.append("")
        lines.append("EXPECTED OUTPUT")
        lines.append("-" * 100)

        output_fields = [
            "amount_safe_to_pay",
            "affordability_status",
            "recommended_payment_method",
            "payment_plan",
            "earliest_date_for_full_payment",
            "spending_changes_needed",
            "decision_explanation",
        ]

        for field in output_fields:
            lines.append(f"{field}: {fmt(req[field])}")

        lines.append("")
        lines.append("FINANCIAL PROFILE")
        lines.append("-" * 100)

        if profile_rows.empty:
            lines.append("NO PROFILE FOUND")
        else:
            profile = profile_rows.iloc[0]

            profile_fields = [
                "home_currency",
                "current_available_balance",
                "minimum_balance_to_keep",
                "financial_priorities",
                "expense_categories_to_protect",
                "expense_categories_user_is_willing_to_reduce",
                "expense_categories_user_is_willing_to_stop",
                "payment_methods_user_will_consider",
                "max_installment_months",
            ]

            for field in profile_fields:
                lines.append(f"{field}: {fmt(profile[field])}")

        lines.append("")
        lines.append("PAYMENT OPTIONS")
        lines.append("-" * 100)

        if request_options.empty:
            lines.append("NONE")
        else:
            lines.append(request_options.to_string(index=False))

        lines.append("")
        lines.append("MESSAGES")
        lines.append("-" * 100)

        if request_messages.empty:
            lines.append("NONE")
        else:
            lines.append(
                request_messages[
                    [
                        "message_id",
                        "request_id",
                        "related_event_id",
                        "sent_at",
                        "source_type",
                        "message_text",
                    ]
                ].to_string(index=False)
            )

        lines.append("")
        lines.append("IMAGES")
        lines.append("-" * 100)

        if request_images.empty:
            lines.append("NONE")
        else:
            lines.append(request_images.to_string(index=False))

        request_date = pd.to_datetime(req["request_date"])
        forecast_end = request_date + pd.Timedelta(days=90)

        if "event_date" in user_events.columns:
            user_events["event_date_dt"] = pd.to_datetime(
                user_events["event_date"], errors="coerce"
            )

            historical_start = request_date - pd.Timedelta(days=120)

            relevant_events = user_events[
                (
                    (user_events["event_date_dt"] >= historical_start)
                    & (user_events["event_date_dt"] <= forecast_end)
                )
                | (user_events["linked_event_id"].notna())
            ].copy()
        else:
            relevant_events = user_events.copy()

        lines.append("")
        lines.append(
            "FINANCIAL EVENTS "
            "(120 DAYS BEFORE REQUEST THROUGH 90-DAY FORECAST)"
        )
        lines.append("-" * 100)

        event_columns = [
            "event_id",
            "event_type",
            "description",
            "category",
            "direction",
            "amount",
            "currency",
            "event_date",
            "settlement_date",
            "status",
            "linked_event_id",
            "flexibility",
            "minimum_allowed_amount",
        ]

        if relevant_events.empty:
            lines.append("NONE")
        else:
            lines.append(
                relevant_events[event_columns]
                .sort_values(["event_date", "event_id"])
                .to_string(index=False)
            )

        lines.append("")
        lines.append("QUICK DERIVED FACTS")
        lines.append("-" * 100)

        safe = float(req["amount_safe_to_pay"])
        requested = float(req["requested_amount"])

        lines.append(
            f"safe/requested ratio: "
            f"{safe / requested:.4f}"
            if requested
            else "safe/requested ratio: n/a"
        )

        lines.append(
            f"full amount safe today without changes: "
            f"{safe >= requested}"
        )

        lines.append(
            f"has spending changes: "
            f"{req['spending_changes_needed'] != 'none'}"
        )

        lines.append(
            f"recommended method: "
            f"{req['recommended_payment_method']}"
        )

        lines.append(
            f"status: "
            f"{req['affordability_status']}"
        )

        lines.append("")
        lines.append("")

        section = "\n".join(lines)

        print(section)

        report_lines.append(section)

    output_path = ROOT / "sample_analysis.txt"

    output_path.write_text(
        "\n".join(report_lines),
        encoding="utf-8"
    )

    print("\n" + "=" * 90)
    print(f"Saved diagnostic report to: {output_path}")
    print("=" * 90)


if __name__ == "__main__":
    main()