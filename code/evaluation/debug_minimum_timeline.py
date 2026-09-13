from collections import defaultdict
from decimal import Decimal
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(ROOT / "code"),
)


from capacity import (
    amount_safe_to_pay,
    earliest_full_payment_date,
)
from data_loader import (
    get_profile,
    get_user_events,
    load_datasets,
)
from forecast import (
    build_baseline_cashflows,
    run_forecast,
)
from lifecycle import CashFlow
from money import (
    ExchangeRateBook,
    money,
)


TARGETS = [
    "request_04",
    "request_25",
    "request_02",
    "request_11",
    "request_03",
]


WINDOW_DAYS = 7


def fmt_date(value):
    if value is None:
        return ""

    if pd.isna(value):
        return ""

    return pd.Timestamp(
        value
    ).strftime("%Y-%m-%d")


def build_daily_timeline(
    opening_balance,
    request_date,
    flows,
):

    request_date = pd.Timestamp(
        request_date
    ).normalize()

    forecast_end = (
        request_date
        + pd.Timedelta(days=90)
    )

    daily_flows = defaultdict(
        lambda: Decimal("0")
    )

    flow_map = defaultdict(list)

    for flow in flows:

        flow_date = pd.Timestamp(
            flow.date
        ).normalize()

        if not (
            request_date
            <= flow_date
            <= forecast_end
        ):
            continue

        amount = money(
            flow.amount
        )

        daily_flows[
            flow_date
        ] += amount

        flow_map[
            flow_date
        ].append(flow)

    balance = money(
        opening_balance
    )

    rows = []

    current = request_date

    while current <= forecast_end:

        opening_day_balance = (
            balance
        )

        net_flow = daily_flows[
            current
        ]

        balance += net_flow

        rows.append(
            {
                "date":
                    current,

                "opening_balance":
                    opening_day_balance,

                "net_flow":
                    net_flow,

                "closing_balance":
                    balance,
            }
        )

        current += pd.Timedelta(
            days=1
        )

    return rows, flow_map


def find_minimum_row(
    timeline,
):

    return min(
        timeline,
        key=lambda row:
            row[
                "closing_balance"
            ],
    )


def print_flow(
    flow,
):

    print(
        "      ",
        str(
            money(
                flow.amount
            )
        ).rjust(16),
        "|",
        str(
            getattr(
                flow,
                "source",
                "",
            )
        ).ljust(22),
        "|",
        str(
            getattr(
                flow,
                "category",
                "",
            )
        ).ljust(22),
        "|",
        str(
            getattr(
                flow,
                "flexibility",
                "",
            )
        ),
    )


def print_window(
    timeline,
    flow_map,
    center_date,
    minimum_balance,
):

    center_date = pd.Timestamp(
        center_date
    ).normalize()

    start = (
        center_date
        - pd.Timedelta(
            days=WINDOW_DAYS
        )
    )

    end = (
        center_date
        + pd.Timedelta(
            days=WINDOW_DAYS
        )
    )

    print(
        "\nBALANCE WINDOW AROUND MINIMUM"
    )

    for row in timeline:

        date = row["date"]

        if not (
            start
            <= date
            <= end
        ):
            continue

        marker = ""

        if date == center_date:
            marker = "  <=== MINIMUM"

        below = ""

        if (
            row["closing_balance"]
            < minimum_balance
        ):
            below = " [BELOW REQUIRED]"

        print(
            " ",
            date.date(),
            "| open:",
            str(
                row[
                    "opening_balance"
                ]
            ).rjust(16),
            "| net:",
            str(
                row[
                    "net_flow"
                ]
            ).rjust(16),
            "| close:",
            str(
                row[
                    "closing_balance"
                ]
            ).rjust(16),
            marker,
            below,
        )

        for flow in flow_map.get(
            date,
            [],
        ):
            print_flow(
                flow
            )


def add_candidate_payment(
    flows,
    payment_date,
    payment_amount,
):

    result = list(
        flows
    )

    result.append(
        CashFlow(
            date=pd.Timestamp(
                payment_date
            ),
            amount=(
                -abs(
                    money(
                        payment_amount
                    )
                )
            ),
            source=(
                "candidate_payment"
            ),
            category=(
                "requested_expense"
            ),
            flexibility=(
                "fixed"
            ),
        )
    )

    return result


def diagnose_candidate(
    label,
    payment_date,
    requested_amount,
    opening_balance,
    minimum_balance,
    request_date,
    baseline_flows,
):

    if not payment_date:
        return

    payment_date = pd.Timestamp(
        payment_date
    ).normalize()

    augmented_flows = (
        add_candidate_payment(
            flows=baseline_flows,
            payment_date=(
                payment_date
            ),
            payment_amount=(
                requested_amount
            ),
        )
    )

    timeline, flow_map = (
        build_daily_timeline(
            opening_balance=(
                opening_balance
            ),
            request_date=(
                request_date
            ),
            flows=(
                augmented_flows
            ),
        )
    )

    minimum_row = (
        find_minimum_row(
            timeline
        )
    )

    safe = (
        minimum_row[
            "closing_balance"
        ]
        >= money(
            minimum_balance
        )
    )

    print()
    print(
        f"{label} FULL-PAYMENT TEST"
    )

    print(
        "  payment date:",
        payment_date.date(),
    )

    print(
        "  minimum after payment:",
        minimum_row[
            "closing_balance"
        ],
    )

    print(
        "  minimum occurs:",
        minimum_row[
            "date"
        ].date(),
    )

    print(
        "  minimum required:",
        money(
            minimum_balance
        ),
    )

    print(
        "  margin:",
        (
            minimum_row[
                "closing_balance"
            ]
            -
            money(
                minimum_balance
            )
        ),
    )

    print(
        "  safe:",
        safe,
    )

    center = (
        minimum_row[
            "date"
        ]
    )

    small_start = (
        center
        - pd.Timedelta(
            days=2
        )
    )

    small_end = (
        center
        + pd.Timedelta(
            days=2
        )
    )

    print(
        "  flows near candidate minimum:"
    )

    found = False

    for date in sorted(
        flow_map.keys()
    ):

        if not (
            small_start
            <= date
            <= small_end
        ):
            continue

        for flow in flow_map[
            date
        ]:

            found = True

            print(
                "   ",
                date.date(),
                end=" ",
            )

            print_flow(
                flow
            )

    if not found:
        print(
            "    NONE"
        )


def main():

    data = load_datasets()

    fx = ExchangeRateBook(
        data.exchange_rates
    )

    for request_id in TARGETS:

        request = (
            data.sample_requests[
                data.sample_requests[
                    "request_id"
                ]
                == request_id
            ]
            .iloc[0]
        )

        user_id = (
            request[
                "user_id"
            ]
        )

        request_date = (
            pd.Timestamp(
                request[
                    "request_date"
                ]
            )
            .normalize()
        )

        profile = get_profile(
            data.profiles,
            user_id,
        )

        events = get_user_events(
            data.events,
            user_id,
        )

        opening_balance = money(
            profile[
                "current_available_balance"
            ]
        )

        minimum_balance = money(
            profile[
                "minimum_balance_to_keep"
            ]
        )

        requested_amount = money(
            request[
                "requested_amount"
            ]
        )

        flows = (
            build_baseline_cashflows(
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
                    request_id
                ),
            )
        )

        baseline = run_forecast(
            opening_balance=(
                opening_balance
            ),
            minimum_balance=(
                minimum_balance
            ),
            request_date=(
                request_date
            ),
            flows=flows,
        )

        predicted_safe = (
            amount_safe_to_pay(
                baseline=baseline,
                requested_amount=(
                    requested_amount
                ),
            )
        )

        predicted_earliest = (
            earliest_full_payment_date(
                opening_balance=(
                    opening_balance
                ),
                minimum_balance=(
                    minimum_balance
                ),
                request_date=(
                    request_date
                ),
                requested_amount=(
                    requested_amount
                ),
                baseline_flows=(
                    flows
                ),
            )
        )

        expected_safe = money(
            request[
                "amount_safe_to_pay"
            ]
        )

        expected_earliest = (
            request[
                "earliest_date_for_full_payment"
            ]
        )

        timeline, flow_map = (
            build_daily_timeline(
                opening_balance=(
                    opening_balance
                ),
                request_date=(
                    request_date
                ),
                flows=flows,
            )
        )

        minimum_row = (
            find_minimum_row(
                timeline
            )
        )

        actual_headroom = (
            minimum_row[
                "closing_balance"
            ]
            -
            minimum_balance
        )

        expected_implied_min = (
            minimum_balance
            + expected_safe
        )

        print()
        print(
            "=" * 145
        )

        print(
            request_id,
            "|",
            user_id,
            "|",
            profile[
                "home_currency"
            ],
        )

        print(
            "=" * 145
        )

        print(
            "REQUEST DATE:        ",
            request_date.date(),
        )

        print(
            "OPENING BALANCE:     ",
            opening_balance,
        )

        print(
            "MINIMUM REQUIRED:    ",
            minimum_balance,
        )

        print(
            "REQUESTED AMOUNT:    ",
            requested_amount,
        )

        print()

        print(
            "EXPECTED SAFE:       ",
            expected_safe,
        )

        print(
            "PREDICTED SAFE:      ",
            predicted_safe,
        )

        print(
            "SAFE DIFFERENCE:     ",
            (
                predicted_safe
                - expected_safe
            ),
        )

        print()

        print(
            "EXPECTED EARLIEST:   ",
            fmt_date(
                expected_earliest
            ),
        )

        print(
            "PREDICTED EARLIEST:  ",
            fmt_date(
                predicted_earliest
            ),
        )

        print()

        print(
            "BASELINE MINIMUM:    ",
            minimum_row[
                "closing_balance"
            ],
        )

        print(
            "MINIMUM DATE:        ",
            minimum_row[
                "date"
            ].date(),
        )

        print(
            "ACTUAL HEADROOM:     ",
            actual_headroom,
        )

        print(
            "EXPECTED IMPLIED MIN:",
            expected_implied_min,
        )

        print(
            "MINIMUM GAP:         ",
            (
                minimum_row[
                    "closing_balance"
                ]
                -
                expected_implied_min
            ),
        )

        print(
            "SAFE CAPPED BY REQUEST?:",
            (
                expected_safe
                >= requested_amount
            ),
        )

        print_window(
            timeline=timeline,
            flow_map=flow_map,
            center_date=(
                minimum_row[
                    "date"
                ]
            ),
            minimum_balance=(
                minimum_balance
            ),
        )

        if (
            expected_earliest
            is not None
            and not pd.isna(
                expected_earliest
            )
        ):

            diagnose_candidate(
                label="EXPECTED-DATE",
                payment_date=(
                    expected_earliest
                ),
                requested_amount=(
                    requested_amount
                ),
                opening_balance=(
                    opening_balance
                ),
                minimum_balance=(
                    minimum_balance
                ),
                request_date=(
                    request_date
                ),
                baseline_flows=flows,
            )

        if (
            predicted_earliest
            is not None
        ):

            if (
                pd.isna(
                    expected_earliest
                )
                or
                pd.Timestamp(
                    predicted_earliest
                ).normalize()
                !=
                pd.Timestamp(
                    expected_earliest
                ).normalize()
            ):

                diagnose_candidate(
                    label=(
                        "PREDICTED-DATE"
                    ),
                    payment_date=(
                        predicted_earliest
                    ),
                    requested_amount=(
                        requested_amount
                    ),
                    opening_balance=(
                        opening_balance
                    ),
                    minimum_balance=(
                        minimum_balance
                    ),
                    request_date=(
                        request_date
                    ),
                    baseline_flows=flows,
                )


if __name__ == "__main__":
    main()
