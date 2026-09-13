from pathlib import Path
import sys

import pandas as pd

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "code"))

from data_loader import (
    load_datasets,
    get_profile,
    get_user_events,
)
from forecast import build_baseline_cashflows
from capacity import earliest_full_payment_date
from money import ExchangeRateBook, money


TARGETS = [
    "request_01",
    "request_08",
    "request_11",
    "request_12",
    "request_17",
    "request_21",
]


def fmt(value):
    if value is None or pd.isna(value):
        return ""
    return str(pd.Timestamp(value).date())


data = load_datasets()

fx = ExchangeRateBook(
    data.exchange_rates
)


for rid in TARGETS:

    req = (
        data.sample_requests[
            data.sample_requests["request_id"] == rid
        ]
        .iloc[0]
    )

    user_id = req["user_id"]

    profile = get_profile(
        data.profiles,
        user_id,
    )

    events = get_user_events(
        data.events,
        user_id,
    )

    request_date = pd.Timestamp(
        req["request_date"]
    ).normalize()

    flows = build_baseline_cashflows(
        events=events,
        request_date=request_date,
        home_currency=profile["home_currency"],
        fx=fx,
        messages=data.messages,
        user_id=user_id,
        request_id=rid,
    )

    predicted = earliest_full_payment_date(
        opening_balance=profile[
            "current_available_balance"
        ],
        minimum_balance=profile[
            "minimum_balance_to_keep"
        ],
        request_date=request_date,
        requested_amount=req[
            "requested_amount"
        ],
        baseline_flows=flows,
    )

    print()
    print("=" * 120)
    print(
        f"{rid} | user={user_id} | "
        f"request_date={fmt(request_date)} | "
        f"requested={req['requested_amount']} | "
        f"opening={profile['current_available_balance']} | "
        f"minimum={profile['minimum_balance_to_keep']}"
    )
    print(
        f"EXPECTED EARLIEST: "
        f"{fmt(req['earliest_date_for_full_payment'])}"
    )
    print(
        f"PREDICTED EARLIEST: "
        f"{fmt(predicted)}"
    )
    print("-" * 120)

    running = money(
        profile[
            "current_available_balance"
        ]
    )

    grouped = {}

    for flow in flows:
        date = pd.Timestamp(
            flow.date
        ).normalize()

        grouped.setdefault(
            date,
            [],
        ).append(
            flow
        )

    for date in sorted(grouped):

        items = grouped[date]

        daily_total = sum(
            (
                flow.amount
                for flow in items
            ),
            money("0"),
        )

        running += daily_total

        print(
            f"\n{date.date()} | "
            f"daily={daily_total} | "
            f"balance={running}"
        )

        for flow in items:

            source = getattr(
                flow,
                "source",
                "",
            )

            category = getattr(
                flow,
                "category",
                "",
            )

            print(
                f"    {flow.amount:>15} "
                f"| source={source:<25} "
                f"| category={category}"
            )

    print()
