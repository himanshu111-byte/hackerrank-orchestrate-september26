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

from evidence import (
    extract_message_evidence,
)

from salary_resolver import (
    infer_stable_salary_stream,
    generate_salary_occurrences,
)

from salary_evidence import (
    apply_salary_evidence,
)

from forecast import (
    build_baseline_cashflows,
)


TARGETS = [
    "request_02",
    "request_11",
    "request_04",
    "request_03",
    "request_25",
]


def main():

    data = load_datasets()

    fx = ExchangeRateBook(
        data.exchange_rates
    )

    print("=" * 120)
    print("SALARY FLOW INTEGRATION DEBUG")
    print("=" * 120)

    for request_id in TARGETS:

        req = data.sample_requests[
            data.sample_requests["request_id"]
            == request_id
        ].iloc[0]

        user_id = req["user_id"]

        request_date = pd.Timestamp(
            req["request_date"]
        )

        # -----------------------------------------------------
        # Home currency comes from profiles.
        # -----------------------------------------------------

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

        stream = (
            infer_stable_salary_stream(
                events=events,
                request_date=request_date,
            )
        )

        stream = (
            apply_salary_evidence(
                stream=stream,
                evidence=evidence,
                request_date=request_date,
            )
        )

        forecast_end = (
            request_date
            + pd.Timedelta(
                days=90
            )
        )

        print()
        print("=" * 120)

        print(
            request_id,
            "|",
            user_id,
            "|",
            home_currency,
            "| request:",
            request_date.date(),
        )

        # -----------------------------------------------------
        # Salary resolver output
        # -----------------------------------------------------

        print("\nSALARY STREAM")

        if stream is None:

            print("  NONE")

        else:

            print(
                "  amount:",
                stream.amount
            )

            print(
                "  currency:",
                stream.currency
            )

            print(
                "  last_date:",
                stream.last_date.date()
            )

            print(
                "  description:",
                stream.description
            )

            occurrences = (
                generate_salary_occurrences(
                    stream=stream,
                    request_date=request_date,
                    forecast_end=forecast_end,
                    home_currency=home_currency,
                    fx=fx,
                )
            )

            print(
                "\nGENERATED SALARY OCCURRENCES:"
            )

            if not occurrences:

                print("  NONE")

            else:

                for (
                    salary_date,
                    salary_amount,
                ) in occurrences:

                    print(
                        " ",
                        pd.Timestamp(
                            salary_date
                        ).date(),
                        salary_amount,
                    )

        # -----------------------------------------------------
        # Final forecast flow output
        # -----------------------------------------------------

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

        print(
            "\nALL SALARY-LIKE FLOWS IN FINAL FORECAST:"
        )

        salary_flows = [
            flow
            for flow in flows
            if (
                str(
                    getattr(
                        flow,
                        "category",
                        "",
                    )
                ).lower()
                == "salary"
            )
        ]

        if not salary_flows:

            print("  NONE")

        else:

            for flow in salary_flows:

                print(
                    " ",
                    pd.Timestamp(
                        flow.date
                    ).date(),
                    flow.amount,
                    "| source:",
                    flow.source,
                    "| category:",
                    flow.category,
                )

        # -----------------------------------------------------
        # Also show all positive credits because salary may
        # exist under a non-salary category/source.
        # -----------------------------------------------------

        print(
            "\nALL POSITIVE FLOWS:"
        )

        positive_flows = [
            flow
            for flow in flows
            if flow.amount > 0
        ]

        if not positive_flows:

            print("  NONE")

        else:

            for flow in positive_flows:

                print(
                    " ",
                    pd.Timestamp(
                        flow.date
                    ).date(),
                    flow.amount,
                    "| source:",
                    flow.source,
                    "| category:",
                    getattr(
                        flow,
                        "category",
                        "",
                    ),
                )

        print(
            "\nTOTAL FINAL FLOW COUNT:",
            len(flows),
        )


if __name__ == "__main__":
    main()