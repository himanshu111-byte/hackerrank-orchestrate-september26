from pathlib import Path
import sys

import pandas as pd

ROOT = Path.cwd()
sys.path.insert(
    0,
    str(ROOT / "code"),
)

from data_loader import (
    load_datasets,
    get_profile,
    get_user_events,
)

from forecast import (
    build_baseline_cashflows,
)

from money import (
    ExchangeRateBook,
    money,
)


def rolling_candidate_safe(
    opening_balance,
    minimum_balance,
    original_request_date,
    candidate_date,
    payment_amount,
    flows,
):
    """
    Experimental rolling 90-day verifier.

    For a candidate full-payment date:

    1. Carry the baseline balance forward from the original
       request date until the day before the candidate date.

    2. Apply the requested payment on the candidate date.

    3. Simulate the candidate date through candidate + 90 days.

    4. The candidate is safe only if the balance never falls
       below minimum_balance_to_keep.
    """

    original_request_date = (
        pd.Timestamp(
            original_request_date
        )
        .normalize()
    )

    candidate_date = (
        pd.Timestamp(
            candidate_date
        )
        .normalize()
    )

    candidate_end = (
        candidate_date
        + pd.Timedelta(
            days=90
        )
    )

    balance = money(
        opening_balance
    )

    minimum_balance = money(
        minimum_balance
    )

    payment_amount = money(
        payment_amount
    )

    # =========================================================
    # 1. CARRY BALANCE FORWARD TO CANDIDATE DATE
    # =========================================================
    #
    # Include baseline flows from request_date up to, but not
    # including, candidate_date.
    # =========================================================

    for flow in flows:

        flow_date = (
            pd.Timestamp(
                flow.date
            )
            .normalize()
        )

        if (
            original_request_date
            <= flow_date
            < candidate_date
        ):
            balance += flow.amount

    # =========================================================
    # 2. COLLECT BASELINE FLOWS FOR THE CANDIDATE WINDOW
    # =========================================================

    by_date = {}

    for flow in flows:

        flow_date = (
            pd.Timestamp(
                flow.date
            )
            .normalize()
        )

        if (
            candidate_date
            <= flow_date
            <= candidate_end
        ):

            by_date[
                flow_date
            ] = (
                by_date.get(
                    flow_date,
                    money("0"),
                )
                + flow.amount
            )

    # =========================================================
    # 3. ADD FULL REQUEST PAYMENT ON CANDIDATE DATE
    # =========================================================

    by_date[
        candidate_date
    ] = (
        by_date.get(
            candidate_date,
            money("0"),
        )
        - abs(
            payment_amount
        )
    )

    # =========================================================
    # 4. SIMULATE CANDIDATE DATE THROUGH CANDIDATE + 90 DAYS
    # =========================================================

    current = (
        candidate_date
    )

    while (
        current
        <= candidate_end
    ):

        balance += by_date.get(
            current,
            money("0"),
        )

        if (
            balance
            < minimum_balance
        ):
            return False

        current += pd.Timedelta(
            days=1
        )

    return True


def rolling_earliest(
    opening_balance,
    minimum_balance,
    request_date,
    requested_amount,
    flows,
):
    """
    Find the earliest full-payment candidate within the original
    90-day search period that passes its own rolling 90-day
    safety check.
    """

    request_date = (
        pd.Timestamp(
            request_date
        )
        .normalize()
    )

    latest_candidate = (
        request_date
        + pd.Timedelta(
            days=90
        )
    )

    candidate = (
        request_date
    )

    while (
        candidate
        <= latest_candidate
    ):

        if rolling_candidate_safe(
            opening_balance=(
                opening_balance
            ),
            minimum_balance=(
                minimum_balance
            ),
            original_request_date=(
                request_date
            ),
            candidate_date=(
                candidate
            ),
            payment_amount=(
                requested_amount
            ),
            flows=(
                flows
            ),
        ):
            return candidate

        candidate += pd.Timedelta(
            days=1
        )

    return None


def format_date(
    value,
) -> str:
    """
    Format a timestamp/date consistently for comparison.
    """

    if (
        value is None
        or pd.isna(
            value
        )
    ):
        return ""

    return str(
        pd.Timestamp(
            value
        ).date()
    )


# =============================================================
# LOAD DATA
# =============================================================

data = load_datasets()

fx = ExchangeRateBook(
    data.exchange_rates
)


# Focus on samples where earliest-date behavior is especially
# informative.
targets = [
    "request_01",
    "request_03",
    "request_04",
    "request_06",
    "request_08",
    "request_11",
    "request_12",
    "request_17",
    "request_19",
    "request_21",
]


print(
    "=" * 100
)

print(
    "ROLLING 90-DAY EARLIEST PAYMENT TEST"
)

print(
    "=" * 100
)


matches = 0


for rid in targets:

    req = (
        data.sample_requests[
            data.sample_requests[
                "request_id"
            ]
            == rid
        ]
        .iloc[0]
    )

    user_id = (
        req[
            "user_id"
        ]
    )

    profile = get_profile(
        data.profiles,
        user_id,
    )

    events = get_user_events(
        data.events,
        user_id,
    )

    request_date = pd.Timestamp(
        req[
            "request_date"
        ]
    )

    # ---------------------------------------------------------
    # IMPORTANT:
    #
    # build_baseline_cashflows() currently generates only through
    # request_date + 90 days.
    #
    # This script is therefore an A/B experiment only.
    #
    # If rolling verification proves useful, we will later extend
    # baseline generation to request_date + 180 days so candidates
    # near the end of the original 90-day search range also have
    # a complete subsequent 90-day forecast.
    # ---------------------------------------------------------

    flows = build_baseline_cashflows(
        events=events,
        request_date=(
            request_date
        ),
        home_currency=(
            profile[
                "home_currency"
            ]
        ),
        fx=fx,
        messages=(
            data.messages
        ),
        user_id=(
            user_id
        ),
        request_id=(
            rid
        ),
    )

    predicted = rolling_earliest(
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
            req[
                "requested_amount"
            ]
        ),
        flows=(
            flows
        ),
    )

    expected_text = format_date(
        req[
            "earliest_date_for_full_payment"
        ]
    )

    predicted_text = format_date(
        predicted
    )

    exact = (
        expected_text
        == predicted_text
    )

    if exact:
        matches += 1

    marker = (
        "OK"
        if exact
        else "DIFF"
    )

    print(
        f"{rid:12s} "
        f"expected={expected_text:12s} "
        f"rolling={predicted_text:12s} "
        f"{marker}"
    )


print()

print(
    "=" * 100
)

print(
    f"Exact matches: "
    f"{matches}/{len(targets)}"
)

print(
    "=" * 100
)