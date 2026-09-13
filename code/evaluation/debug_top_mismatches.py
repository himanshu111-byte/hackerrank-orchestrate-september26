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
    get_user_events,
)

from money import ExchangeRateBook

from forecast import (
    build_baseline_cashflows,
)

from recurrence import (
    infer_recurring_series,
)

from salary_resolver import (
    infer_stable_salary_stream,
    generate_salary_occurrences,
)

from salary_evidence import (
    apply_salary_evidence,
)

from evidence import (
    extract_message_evidence,
)


TARGETS = [
    "request_04",
    "request_25",
    "request_02",
    "request_11",
    "request_03",
]


def get_value(obj, name, default=""):

    try:
        return getattr(
            obj,
            name,
        )
    except Exception:
        return default


def main():

    data = load_datasets()

    fx = ExchangeRateBook(
        data.exchange_rates
    )

    print("=" * 130)
    print("TOP MISMATCH CASH-FLOW DIAGNOSTIC")
    print("=" * 130)

    for request_id in TARGETS:

        req = data.sample_requests[
            data.sample_requests["request_id"]
            == request_id
        ].iloc[0]

        user_id = req["user_id"]

        request_date = pd.Timestamp(
            req["request_date"]
        )

        profile = data.profiles[
            data.profiles["user_id"]
            == user_id
        ].iloc[0]

        home_currency = profile[
            "home_currency"
        ]

        events = get_user_events(
            data.events,
            user_id,
        )

        evidence = (
            extract_message_evidence(
                messages=data.messages,
                user_id=user_id,
                request_id=request_id,
            )
        )

        flows = (
            build_baseline_cashflows(
                events=events,
                request_date=request_date,
                home_currency=home_currency,
                fx=fx,
                messages=data.messages,
                user_id=user_id,
                request_id=request_id,
            )
        )

        recurring_series = (
            infer_recurring_series(
                events=events,
                request_date=request_date,
                home_currency=home_currency,
                fx=fx,
            )
        )

        salary_stream = (
            infer_stable_salary_stream(
                events=events,
                request_date=request_date,
            )
        )

        salary_stream = (
            apply_salary_evidence(
                stream=salary_stream,
                evidence=evidence,
                request_date=request_date,
            )
        )

        print()
        print("=" * 130)

        print(
            request_id,
            "|",
            user_id,
            "|",
            home_currency,
            "| request date:",
            request_date.date(),
        )

        print(
            "expected amount_safe:",
            req["amount_safe_to_pay"],
        )

        print(
            "expected earliest:",
            req[
                "earliest_date_for_full_payment"
            ],
        )

        # =====================================================
        # SALARY
        # =====================================================

        print("\nSALARY")

        if salary_stream is None:

            print("  NONE")

        else:

            print(
                "  amount:",
                salary_stream.amount,
            )

            print(
                "  currency:",
                salary_stream.currency,
            )

            print(
                "  last date:",
                salary_stream.last_date.date(),
            )

            print(
                "  description:",
                salary_stream.description,
            )

            salary_occurrences = (
                generate_salary_occurrences(
                    stream=salary_stream,
                    request_date=request_date,
                    forecast_end=(
                        request_date
                        + pd.Timedelta(days=90)
                    ),
                    home_currency=home_currency,
                    fx=fx,
                )
            )

            print(
                "  generated occurrences:"
            )

            for (
                salary_date,
                salary_amount,
            ) in salary_occurrences:

                print(
                    "   ",
                    pd.Timestamp(
                        salary_date
                    ).date(),
                    salary_amount,
                )

        # =====================================================
        # GENERIC RECURRING SERIES
        # =====================================================

        print("\nINFERRED RECURRING SERIES")

        if not recurring_series:

            print("  NONE")

        else:

            for series in recurring_series:

                print(
                    " ",
                    "type=",
                    get_value(
                        series,
                        "event_type",
                    ),
                    "| category=",
                    get_value(
                        series,
                        "category",
                    ),
                    "| direction=",
                    get_value(
                        series,
                        "direction",
                    ),
                    "| amount=",
                    get_value(
                        series,
                        "amount",
                    ),
                    "| monthly=",
                    get_value(
                        series,
                        "monthly",
                    ),
                    "| cadence=",
                    get_value(
                        series,
                        "cadence_days",
                    ),
                    "| last_date=",
                    get_value(
                        series,
                        "last_date",
                    ),
                    "| samples=",
                    get_value(
                        series,
                        "sample_count",
                    ),
                )

        # =====================================================
        # FINAL FORECAST FLOWS
        # =====================================================

        print("\nFINAL FORECAST FLOWS")

        total_positive = 0
        total_negative = 0

        for flow in flows:

            amount = flow.amount

            if amount > 0:
                total_positive += amount
            else:
                total_negative += amount

            print(
                " ",
                pd.Timestamp(
                    flow.date
                ).date(),
                "|",
                str(amount).rjust(15),
                "| source=",
                get_value(
                    flow,
                    "source",
                ),
                "| category=",
                get_value(
                    flow,
                    "category",
                ),
                "| flexibility=",
                get_value(
                    flow,
                    "flexibility",
                ),
            )

        print(
            "\nTOTAL POSITIVE:",
            total_positive,
        )

        print(
            "TOTAL NEGATIVE:",
            total_negative,
        )

        print(
            "NET FORECAST FLOW:",
            total_positive
            + total_negative,
        )

        # =====================================================
        # HISTORICAL SETTLED EVENTS
        # =====================================================

        print("\nRECENT HISTORICAL SETTLED EVENTS")

        historical = events[
            (
                events["event_date"]
                <= request_date
            )
            &
            (
                events["status"]
                .astype(str)
                .str.lower()
                == "settled"
            )
        ].copy()

        historical = (
            historical.sort_values(
                "event_date"
            )
            .tail(30)
        )

        columns = [
            column
            for column in [
                "event_id",
                "event_date",
                "event_type",
                "description",
                "category",
                "amount",
                "currency",
                "direction",
                "status",
            ]
            if column
            in historical.columns
        ]

        if historical.empty:

            print("  NONE")

        else:

            print(
                historical[
                    columns
                ].to_string(
                    index=False
                )
            )


if __name__ == "__main__":
    main()