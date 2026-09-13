from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(ROOT / "code"),
)


from data_loader import load_datasets
from evidence import extract_message_evidence


def main():

    data = load_datasets()

    print("=" * 100)
    print("MESSAGE EVIDENCE")
    print("=" * 100)

    for _, req in (
        data.sample_requests.iterrows()
    ):

        evidence = (
            extract_message_evidence(
                messages=data.messages,
                user_id=req["user_id"],
                request_id=req["request_id"],
            )
        )

        if not evidence:
            continue

        print()
        print(
            req["request_id"],
            req["user_id"],
        )

        for item in evidence:
            print(
                " ",
                item.evidence_type,
                "amount=",
                item.amount,
                "currency=",
                item.currency,
                "date=",
                item.effective_date,
                "event=",
                item.related_event_id,
                "confirmed=",
                item.confirmed,
            )


if __name__ == "__main__":
    main()