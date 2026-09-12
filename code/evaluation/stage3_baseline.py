from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(ROOT / "code"),
)


from capacity import (  # noqa: E402
    amount_safe_to_pay,
    earliest_full_payment_date,
)
from data_loader import (  # noqa: E402
    get_profile,
    get_user_events,
    load_datasets,
)
from forecast import (  # noqa: E402
    build_baseline_cashflows,
    run_forecast,
)
from money import ExchangeRateBook  # noqa: E402


def fmt_date(value):
    if value is None:
        return ""

    if pd.isna(value):
        return ""

    return pd.Timestamp(
        value
    ).strftime("%Y-%m-%d")


def main():

    data = load_datasets()

    fx = ExchangeRateBook(
        data.exchange_rates
    )

    results = []

    print(
        "=" * 100
    )

    print(
        "STAGE 3 — BASELINE "
        "FINANCIAL FORECAST"
    )

    print(
        "=" * 100
    )

    for _, request in (
        data.sample_requests.iterrows()
    ):

        request_id = (
            request["request_id"]
        )

        user_id = request["user_id"]

        request_date = pd.Timestamp(
            request["request_date"]
        )

        profile = get_profile(
            data.profiles,
            user_id,
        )

        events = get_user_events(
            data.events,
            user_id,
        )

        flows = (
            build_baseline_cashflows(
                events=events,
                request_date=request_date,
                home_currency=(
                    profile[
                        "home_currency"
                    ]
                ),
                fx=fx,
            )
        )

        baseline = run_forecast(
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
            request_date=request_date,
            flows=flows,
        )

        safe = amount_safe_to_pay(
            baseline=baseline,
            requested_amount=(
                request[
                    "requested_amount"
                ]
            ),
        )

        earliest = (
            earliest_full_payment_date(
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
                requested_amount=(
                    request[
                        "requested_amount"
                    ]
                ),
                baseline_flows=flows,
            )
        )

        expected_safe = float(
            request[
                "amount_safe_to_pay"
            ]
        )

        predicted_safe = float(
            safe
        )

        expected_earliest = (
            fmt_date(
                request[
                    "earliest_date_for_full_payment"
                ]
            )
        )

        predicted_earliest = (
            fmt_date(earliest)
        )

        safe_diff = (
            predicted_safe
            - expected_safe
        )

        results.append(
            {
                "request_id":
                    request_id,

                "expected_safe":
                    expected_safe,

                "predicted_safe":
                    predicted_safe,

                "safe_diff":
                    safe_diff,

                "expected_earliest":
                    expected_earliest,

                "predicted_earliest":
                    predicted_earliest,

                "baseline_min":
                    float(
                        baseline
                        .minimum_forecast_balance
                    ),

                "min_required":
                    float(
                        baseline
                        .minimum_balance_required
                    ),

                "flow_count":
                    len(flows),
            }
        )

    result_df = pd.DataFrame(
        results
    )

    pd.set_option(
        "display.max_columns",
        None,
    )

    pd.set_option(
        "display.width",
        220,
    )

    print(
        result_df.to_string(
            index=False
        )
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "AMOUNT SAFE SUMMARY"
    )

    print(
        "=" * 100
    )

    exact_amount = (
        result_df[
            "safe_diff"
        ].abs()
        <= 0.01
    )

    print(
        "Exact within 0.01:",
        f"{exact_amount.sum()}"
        f"/{len(result_df)}",
    )

    print(
        "Mean absolute difference:",
        result_df[
            "safe_diff"
        ].abs().mean(),
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "EARLIEST DATE SUMMARY"
    )

    print(
        "=" * 100
    )

    date_match = (
        result_df[
            "expected_earliest"
        ]
        ==
        result_df[
            "predicted_earliest"
        ]
    )

    print(
        "Exact dates:",
        f"{date_match.sum()}"
        f"/{len(result_df)}",
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "BIGGEST AMOUNT MISMATCHES"
    )

    print(
        "=" * 100
    )

    biggest = (
        result_df
        .assign(
            abs_diff=lambda x:
                x["safe_diff"].abs()
        )
        .sort_values(
            "abs_diff",
            ascending=False,
        )
        .head(10)
    )

    print(
        biggest[
            [
                "request_id",
                "expected_safe",
                "predicted_safe",
                "safe_diff",
                "expected_earliest",
                "predicted_earliest",
            ]
        ].to_string(
            index=False
        )
    )

    output_path = (
        ROOT
        / "stage3_baseline_results.csv"
    )

    result_df.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nSaved detailed results to "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()