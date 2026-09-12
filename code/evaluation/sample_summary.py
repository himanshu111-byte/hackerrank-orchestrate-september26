from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "dataset"


def main():
    samples = pd.read_csv(DATASET / "sample_requests.csv")
    profiles = pd.read_csv(DATASET / "financial_profiles.csv")
    options = pd.read_csv(DATASET / "request_payment_options.csv")

    rows = []

    for _, request in samples.iterrows():
        profile = profiles[
            profiles["user_id"] == request["user_id"]
        ].iloc[0]

        request_options = options[
            options["request_id"] == request["request_id"]
        ]

        rows.append(
            {
                "request_id": request["request_id"],
                "type": request["request_type"],
                "requested": request["requested_amount"],
                "safe_today": request["amount_safe_to_pay"],
                "safe_ratio": round(
                    request["amount_safe_to_pay"]
                    / request["requested_amount"],
                    4,
                ),
                "status": request["affordability_status"],
                "method": request["recommended_payment_method"],
                "earliest_full": request[
                    "earliest_date_for_full_payment"
                ],
                "allows_partial": request[
                    "allows_partial_payment"
                ],
                "methods_considered": profile[
                    "payment_methods_user_will_consider"
                ],
                "max_installment_months": profile[
                    "max_installment_months"
                ],
                "option_count": len(request_options),
                "spending_changes": request[
                    "spending_changes_needed"
                ],
            }
        )

    summary = pd.DataFrame(rows)

    print(summary.to_string(index=False))

    print("\nSTATUS COUNTS")
    print(summary["status"].value_counts())

    print("\nMETHOD COUNTS")
    print(summary["method"].value_counts())

    print("\nSTATUS × METHOD")
    print(
        pd.crosstab(
            summary["status"],
            summary["method"]
        )
    )


if __name__ == "__main__":
    main()