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

from salary_resolver import (
    infer_stable_salary_stream,
)

from salary_evidence import (
    apply_salary_evidence,
)

from evidence import (
    extract_message_evidence,
)


TARGETS = [
    "request_11",
    "request_04",
    "request_03",
    "request_25",
    "request_02",
    "request_05",
]


def main():

    data = load_datasets()

    print("=" * 110)
    print("SALARY RESOLVER TEST")
    print("=" * 110)

    for request_id in TARGETS:

        req = data.sample_requests[
            data.sample_requests[
                "request_id"
            ]
            == request_id
        ].iloc[0]

        user_id = req["user_id"]

        request_date = pd.Timestamp(
            req["request_date"]
        )

        events = get_user_events(
            data.events,
            user_id,
        )

        stream = (
            infer_stable_salary_stream(
                events=events,
                request_date=request_date,
            )
        )

        evidence = (
            extract_message_evidence(
                messages=data.messages,
                user_id=user_id,
                request_id=request_id,
            )
        )

        stream = apply_salary_evidence(
            stream=stream,
            evidence=evidence,
            request_date=request_date,
        )

        print()
        print(
            request_id,
            "|",
            user_id,
        )

        if stream is None:
            print(
                "  salary_stream = NONE"
            )

        else:
            print(
                "  amount =",
                stream.amount,
            )

            print(
                "  currency =",
                stream.currency,
            )

            print(
                "  last_date =",
                stream.last_date.date(),
            )

            print(
                "  description =",
                stream.description,
            )


if __name__ == "__main__":
    main()