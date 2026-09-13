from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from itertools import combinations
from pathlib import Path

import pandas as pd

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
    round_money,
)

from spending_changes import (
    actions_are_compatible,
    apply_spending_actions,
    format_spending_actions,
    generate_spending_actions,
)


ROOT = Path(
    __file__
).resolve().parents[1]

OUTPUT_PATH = (
    ROOT
    / "output.csv"
)


OUTPUT_COLUMNS = [
    "request_id",
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation",
]


# =============================================================
# CANDIDATE REPRESENTATION
# =============================================================


@dataclass
class PaymentCandidate:
    method: str

    payments: list[
        tuple[
            pd.Timestamp,
            Decimal,
        ]
    ]

    total_paid: Decimal

    spending_changes: str = "none"

    payment_option_id: str | None = None

    status: str = "affordable_with_plan"

    explanation: str = ""

    @property
    def first_payment_date(
        self,
    ) -> pd.Timestamp:

        return min(
            pd.Timestamp(
                date
            ).normalize()
            for (
                date,
                _,
            ) in self.payments
        )

    @property
    def final_payment_date(
        self,
    ) -> pd.Timestamp:

        return max(
            pd.Timestamp(
                date
            ).normalize()
            for (
                date,
                _,
            ) in self.payments
        )

    @property
    def payment_count(
        self,
    ) -> int:

        return len(
            self.payments
        )


# =============================================================
# GENERIC HELPERS
# =============================================================


def _bool_value(
    value,
) -> bool:

    if isinstance(
        value,
        bool,
    ):
        return value

    if pd.isna(
        value
    ):
        return False

    return (
        str(
            value
        )
        .strip()
        .lower()
        in {
            "true",
            "1",
            "yes",
            "y",
        }
    )


def _split_preferences(
    value,
) -> set[str]:

    if pd.isna(
        value
    ):
        return set()

    return {
        item.strip()
        for item in str(
            value
        ).split("|")
        if item.strip()
    }


def _fmt_money(
    value,
) -> str:

    value = round_money(
        money(
            value
        )
    )

    text = format(
        value,
        "f",
    )

    if "." in text:

        text = (
            text
            .rstrip("0")
            .rstrip(".")
        )

    if text in {
        "-0",
        "",
    }:
        return "0"

    return text


def _fmt_date(
    value,
) -> str:

    if (
        value is None
        or pd.isna(
            value
        )
    ):
        return ""

    return (
        pd.Timestamp(
            value
        )
        .normalize()
        .strftime(
            "%Y-%m-%d"
        )
    )


def _format_payment_plan(
    payments: list[
        tuple[
            pd.Timestamp,
            Decimal,
        ]
    ],
) -> str:

    if not payments:
        return "none"

    ordered = sorted(
        payments,
        key=lambda item: (
            pd.Timestamp(
                item[0]
            )
        ),
    )

    return "|".join(
        (
            f"{_fmt_date(date)}:"
            f"{_fmt_money(amount)}"
        )
        for (
            date,
            amount,
        ) in ordered
    )


# =============================================================
# PLAN VERIFICATION
# =============================================================


def _payments_to_cashflows(
    payments: list[
        tuple[
            pd.Timestamp,
            Decimal,
        ]
    ],
) -> list[
    CashFlow
]:

    result = []

    for (
        payment_date,
        payment_amount,
    ) in payments:

        amount = abs(
            money(
                payment_amount
            )
        )

        if amount <= 0:
            continue

        result.append(
            CashFlow(
                date=(
                    pd.Timestamp(
                        payment_date
                    )
                    .normalize()
                ),
                amount=-amount,
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


def _candidate_is_safe(
    candidate: PaymentCandidate,
    opening_balance,
    minimum_balance,
    request_date,
    desired_completion_date,
    baseline_flows,
) -> bool:
    """
    Candidate-plan validation used by the public ground-truth examples.

    The underlying baseline remains a 90-day forecast.

    A candidate must:

      1. finish by desired_completion_date; and
      2. remain above minimum_balance from request_date through the
         desired completion horizon.
    """

    request_date = (
        pd.Timestamp(
            request_date
        )
        .normalize()
    )

    desired_completion_date = (
        pd.Timestamp(
            desired_completion_date
        )
        .normalize()
    )

    if (
        candidate.final_payment_date
        > desired_completion_date
    ):
        return False

    result = run_forecast(
        opening_balance=(
            opening_balance
        ),
        minimum_balance=(
            minimum_balance
        ),
        request_date=(
            request_date
        ),
        flows=(
            baseline_flows
        ),
        additional_flows=(
            _payments_to_cashflows(
                candidate.payments
            )
        ),
    )

    relevant_daily = (
        result.daily[
            (
                result.daily[
                    "date"
                ]
                >= request_date
            )
            &
            (
                result.daily[
                    "date"
                ]
                <= desired_completion_date
            )
        ]
    )

    if relevant_daily.empty:
        return False

    minimum_through_deadline = money(
        str(
            relevant_daily[
                "balance"
            ].min()
        )
    )

    return (
        minimum_through_deadline
        >=
        money(
            minimum_balance
        )
    )


def _find_first_deadline_safe_full_date(
    opening_balance,
    minimum_balance,
    request_date,
    desired_completion_date,
    requested_amount,
    baseline_flows,
) -> pd.Timestamp | None:

    request_date = (
        pd.Timestamp(
            request_date
        )
        .normalize()
    )

    desired_completion_date = (
        pd.Timestamp(
            desired_completion_date
        )
        .normalize()
    )

    candidate_date = (
        request_date
    )

    while (
        candidate_date
        <= desired_completion_date
    ):

        candidate = PaymentCandidate(
            method=(
                "full_payment"
                if (
                    candidate_date
                    == request_date
                )
                else "wait"
            ),
            payments=[
                (
                    candidate_date,
                    requested_amount,
                )
            ],
            total_paid=(
                requested_amount
            ),
        )

        if _candidate_is_safe(
            candidate=(
                candidate
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
            desired_completion_date=(
                desired_completion_date
            ),
            baseline_flows=(
                baseline_flows
            ),
        ):
            return candidate_date

        candidate_date += (
            pd.Timedelta(
                days=1
            )
        )

    return None


# =============================================================
# FULL PAYMENT
# =============================================================


def _full_payment_candidate(
    request_date: pd.Timestamp,
    desired_completion_date: pd.Timestamp,
    requested_amount: Decimal,
    deadline_safe_full_date: pd.Timestamp | None,
    accepted_methods: set[str],
) -> PaymentCandidate | None:

    if (
        "full_payment"
        not in accepted_methods
    ):
        return None

    if (
        deadline_safe_full_date
        is None
    ):
        return None

    request_date = (
        pd.Timestamp(
            request_date
        )
        .normalize()
    )

    desired_completion_date = (
        pd.Timestamp(
            desired_completion_date
        )
        .normalize()
    )

    deadline_safe_full_date = (
        pd.Timestamp(
            deadline_safe_full_date
        )
        .normalize()
    )

    if (
        deadline_safe_full_date
        != request_date
    ):
        return None

    if (
        request_date
        > desired_completion_date
    ):
        return None

    return PaymentCandidate(
        method=(
            "full_payment"
        ),
        payments=[
            (
                request_date,
                requested_amount,
            )
        ],
        total_paid=(
            requested_amount
        ),
        status=(
            "affordable_now"
        ),
        explanation=(
            "The full requested amount can be paid today while "
            "maintaining the required minimum balance."
        ),
    )


# =============================================================
# WAIT
# =============================================================


def _wait_candidate(
    request_date: pd.Timestamp,
    desired_completion_date: pd.Timestamp,
    requested_amount: Decimal,
    deadline_safe_full_date: pd.Timestamp | None,
    accepted_methods: set[str],
) -> PaymentCandidate | None:

    if (
        "full_payment"
        not in accepted_methods
    ):
        return None

    if (
        deadline_safe_full_date
        is None
    ):
        return None

    request_date = (
        pd.Timestamp(
            request_date
        )
        .normalize()
    )

    desired_completion_date = (
        pd.Timestamp(
            desired_completion_date
        )
        .normalize()
    )

    deadline_safe_full_date = (
        pd.Timestamp(
            deadline_safe_full_date
        )
        .normalize()
    )

    if (
        deadline_safe_full_date
        <= request_date
    ):
        return None

    if (
        deadline_safe_full_date
        > desired_completion_date
    ):
        return None

    return PaymentCandidate(
        method=(
            "wait"
        ),
        payments=[
            (
                deadline_safe_full_date,
                requested_amount,
            )
        ],
        total_paid=(
            requested_amount
        ),
        status=(
            "affordable_later"
        ),
        explanation=(
            "The full amount is not safe to pay today, but becomes "
            f"safe on {_fmt_date(deadline_safe_full_date)}."
        ),
    )


# =============================================================
# PARTIAL PAYMENT
# =============================================================


def _partial_payment_candidate(
    request_date: pd.Timestamp,
    desired_completion_date: pd.Timestamp,
    requested_amount: Decimal,
    safe_today: Decimal,
    earliest_full: pd.Timestamp | None,
    allows_partial_payment: bool,
    accepted_methods: set[str],
) -> PaymentCandidate | None:

    if not allows_partial_payment:
        return None

    if (
        "partial_payment"
        not in accepted_methods
    ):
        return None

    if not (
        Decimal("0")
        < safe_today
        < requested_amount
    ):
        return None

    if earliest_full is None:
        return None

    request_date = (
        pd.Timestamp(
            request_date
        )
        .normalize()
    )

    earliest_full = (
        pd.Timestamp(
            earliest_full
        )
        .normalize()
    )

    desired_completion_date = (
        pd.Timestamp(
            desired_completion_date
        )
        .normalize()
    )

    if (
        earliest_full
        > desired_completion_date
    ):
        return None

    remainder = round_money(
        requested_amount
        - safe_today
    )

    if remainder <= 0:
        return None

    return PaymentCandidate(
        method=(
            "partial_payment"
        ),
        payments=[
            (
                request_date,
                safe_today,
            ),
            (
                earliest_full,
                remainder,
            ),
        ],
        total_paid=(
            requested_amount
        ),
        status=(
            "affordable_with_plan"
        ),
        explanation=(
            f"Pay {_fmt_money(safe_today)} today and the remaining "
            f"{_fmt_money(remainder)} on "
            f"{_fmt_date(earliest_full)}."
        ),
    )


# =============================================================
# INSTALLMENTS
# =============================================================


def _installment_months_allowed(
    number_of_payments: int,
    max_installment_months,
) -> bool:

    if pd.isna(
        max_installment_months
    ):
        return False

    try:
        maximum = int(
            float(
                max_installment_months
            )
        )

    except (
        TypeError,
        ValueError,
    ):
        return False

    return (
        number_of_payments
        <= maximum
    )


def _build_installment_payments(
    option: pd.Series,
) -> list[
    tuple[
        pd.Timestamp,
        Decimal,
    ]
]:

    number_of_payments = int(
        option[
            "number_of_payments"
        ]
    )

    first_payment_date = (
        pd.Timestamp(
            option[
                "first_payment_date"
            ]
        )
        .normalize()
    )

    frequency_days = int(
        float(
            option[
                "payment_frequency_days"
            ]
        )
    )

    payment_amount = money(
        option[
            "payment_amount"
        ]
    )

    payments = []

    for index in range(
        number_of_payments
    ):

        payment_date = (
            first_payment_date
            + pd.Timedelta(
                days=(
                    index
                    * frequency_days
                )
            )
        )

        payments.append(
            (
                payment_date,
                payment_amount,
            )
        )

    return payments


def _installment_candidates(
    request_id: str,
    payment_options: pd.DataFrame,
    desired_completion_date: pd.Timestamp,
    accepted_methods: set[str],
    max_installment_months,
) -> list[
    PaymentCandidate
]:

    if (
        "installments"
        not in accepted_methods
    ):
        return []

    if pd.isna(
        max_installment_months
    ):
        return []

    options = (
        payment_options[
            (
                payment_options[
                    "request_id"
                ]
                == request_id
            )
            &
            (
                payment_options[
                    "payment_method"
                ]
                .astype(str)
                .str.lower()
                == "installments"
            )
        ]
        .copy()
        .sort_values(
            "payment_option_id"
        )
    )

    candidates = []

    desired_completion_date = (
        pd.Timestamp(
            desired_completion_date
        )
        .normalize()
    )

    for (
        _,
        option,
    ) in options.iterrows():

        number_of_payments = int(
            option[
                "number_of_payments"
            ]
        )

        if not _installment_months_allowed(
            number_of_payments=(
                number_of_payments
            ),
            max_installment_months=(
                max_installment_months
            ),
        ):
            continue

        if pd.isna(
            option[
                "first_payment_date"
            ]
        ):
            continue

        if pd.isna(
            option[
                "payment_frequency_days"
            ]
        ):
            continue

        payments = (
            _build_installment_payments(
                option
            )
        )

        if not payments:
            continue

        final_payment_date = (
            pd.Timestamp(
                payments[-1][0]
            )
            .normalize()
        )

        if (
            final_payment_date
            > desired_completion_date
        ):
            continue

        total_paid = sum(
            (
                amount
                for (
                    _,
                    amount,
                ) in payments
            ),
            Decimal("0"),
        )

        candidates.append(
            PaymentCandidate(
                method=(
                    "installments"
                ),
                payments=(
                    payments
                ),
                total_paid=(
                    round_money(
                        total_paid
                    )
                ),
                payment_option_id=str(
                    option[
                        "payment_option_id"
                    ]
                ),
                status=(
                    "affordable_with_plan"
                ),
                explanation=(
                    "Use the supplied "
                    f"{number_of_payments}-payment installment "
                    f"option ending "
                    f"{_fmt_date(final_payment_date)}."
                ),
            )
        )

    return candidates


# =============================================================
# SPENDING CHANGES
# =============================================================


def _spending_change_candidates(
    profile: pd.Series,
    request_date: pd.Timestamp,
    requested_amount: Decimal,
    accepted_methods: set[str],
    baseline_flows: list[
        CashFlow
    ],
) -> list[
    tuple[
        PaymentCandidate,
        list[CashFlow],
    ]
]:
    """
    Generate full-payment-today candidates using between one and
    three legal flexible-spending changes.

    Each returned item contains:

        (candidate, modified_baseline_flows)

    because a spending change modifies recurring projections rather
    than the payment itself.
    """

    if (
        "full_payment"
        not in accepted_methods
    ):
        return []

    actions = (
        generate_spending_actions(
            profile=profile,
            baseline_flows=(
                baseline_flows
            ),
        )
    )

    if not actions:
        return []

    request_date = (
        pd.Timestamp(
            request_date
        )
        .normalize()
    )

    results = []

    max_action_count = min(
        3,
        len(
            actions
        ),
    )

    for action_count in range(
        1,
        max_action_count + 1,
    ):

        for combo_tuple in combinations(
            actions,
            action_count,
        ):

            selected_actions = list(
                combo_tuple
            )

            # Prevent cases such as:
            #
            # stop:event_1816
            # reduce_to:event_1816:23.50
            #
            # in one plan.
            if not actions_are_compatible(
                selected_actions
            ):
                continue

            modified_flows = (
                apply_spending_actions(
                    baseline_flows=(
                        baseline_flows
                    ),
                    actions=(
                        selected_actions
                    ),
                )
            )

            changes_text = (
                format_spending_actions(
                    selected_actions
                )
            )

            candidate = PaymentCandidate(
                method=(
                    "full_payment"
                ),
                payments=[
                    (
                        request_date,
                        requested_amount,
                    )
                ],
                total_paid=(
                    requested_amount
                ),
                spending_changes=(
                    changes_text
                ),
                status=(
                    "affordable_with_plan"
                ),
                explanation=(
                    "Make the permitted flexible-spending changes "
                    f"({changes_text}), then pay the full requested "
                    "amount today while maintaining the required "
                    "minimum balance."
                ),
            )

            results.append(
                (
                    candidate,
                    modified_flows,
                )
            )

    return results


# =============================================================
# CANDIDATE RANKING
# =============================================================


def _option_id_rank(
    option_id: str | None,
) -> int:

    if option_id is None:
        return -1

    text = str(
        option_id
    )

    digits = "".join(
        character
        for character in text
        if character.isdigit()
    )

    if not digits:
        return 10**9

    return int(
        digits
    )


def _candidate_rank(
    candidate: PaymentCandidate,
):
    """
    Ranking:

    1. no spending changes
    2. minimum total paid
    3. earlier start
    4. fewer payments
    5. lowest supplied option ID

    For otherwise identical spending-change candidates, fewer textual
    changes and deterministic lexical ordering provide stable output.
    """

    requires_changes = (
        candidate.spending_changes
        != "none"
    )

    change_count = (
        0
        if not requires_changes
        else len(
            candidate.spending_changes.split(
                "|"
            )
        )
    )

    return (
        1 if requires_changes else 0,
        candidate.total_paid,
        candidate.first_payment_date,
        candidate.payment_count,
        _option_id_rank(
            candidate.payment_option_id
        ),
        change_count,
        candidate.spending_changes,
    )


# =============================================================
# REQUEST DECISION
# =============================================================


def decide_request(
    request: pd.Series,
    profile: pd.Series,
    user_events: pd.DataFrame,
    payment_options: pd.DataFrame,
    messages: pd.DataFrame,
    fx: ExchangeRateBook,
) -> dict:

    request_id = str(
        request[
            "request_id"
        ]
    )

    user_id = str(
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

    desired_completion_date = (
        pd.Timestamp(
            request[
                "desired_completion_date"
            ]
        )
        .normalize()
    )

    requested_amount = money(
        request[
            "requested_amount"
        ]
    )

    home_currency = str(
        profile[
            "home_currency"
        ]
    ).strip().upper()

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

    accepted_methods = (
        _split_preferences(
            profile[
                "payment_methods_user_will_consider"
            ]
        )
    )

    allows_partial = (
        _bool_value(
            request[
                "allows_partial_payment"
            ]
        )
    )

    # =========================================================
    # BASELINE
    # =========================================================

    baseline_flows = (
        build_baseline_cashflows(
            events=user_events,
            request_date=request_date,
            home_currency=home_currency,
            fx=fx,
            messages=messages,
            user_id=user_id,
            request_id=request_id,
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
        flows=(
            baseline_flows
        ),
    )

    # =========================================================
    # INDEPENDENT OUTPUT FIELDS
    # =========================================================

    safe_today = (
        amount_safe_to_pay(
            baseline=baseline,
            requested_amount=(
                requested_amount
            ),
        )
    )

    earliest_full = (
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
                baseline_flows
            ),
        )
    )

    # =========================================================
    # DEADLINE-AWARE FULL/W​​AIT DATE
    # =========================================================

    deadline_safe_full = None

    if (
        "full_payment"
        in accepted_methods
    ):

        deadline_safe_full = (
            _find_first_deadline_safe_full_date(
                opening_balance=(
                    opening_balance
                ),
                minimum_balance=(
                    minimum_balance
                ),
                request_date=(
                    request_date
                ),
                desired_completion_date=(
                    desired_completion_date
                ),
                requested_amount=(
                    requested_amount
                ),
                baseline_flows=(
                    baseline_flows
                ),
            )
        )

    # Candidate paired with the baseline it must be validated against.
    candidates: list[
        tuple[
            PaymentCandidate,
            list[CashFlow],
        ]
    ] = []

    # =========================================================
    # FULL
    # =========================================================

    full = (
        _full_payment_candidate(
            request_date=(
                request_date
            ),
            desired_completion_date=(
                desired_completion_date
            ),
            requested_amount=(
                requested_amount
            ),
            deadline_safe_full_date=(
                deadline_safe_full
            ),
            accepted_methods=(
                accepted_methods
            ),
        )
    )

    if full is not None:

        candidates.append(
            (
                full,
                baseline_flows,
            )
        )

    # =========================================================
    # PARTIAL
    # =========================================================

    partial = (
        _partial_payment_candidate(
            request_date=(
                request_date
            ),
            desired_completion_date=(
                desired_completion_date
            ),
            requested_amount=(
                requested_amount
            ),
            safe_today=(
                safe_today
            ),
            earliest_full=(
                earliest_full
            ),
            allows_partial_payment=(
                allows_partial
            ),
            accepted_methods=(
                accepted_methods
            ),
        )
    )

    if partial is not None:

        candidates.append(
            (
                partial,
                baseline_flows,
            )
        )

    # =========================================================
    # INSTALLMENTS
    # =========================================================

    installments = (
        _installment_candidates(
            request_id=(
                request_id
            ),
            payment_options=(
                payment_options
            ),
            desired_completion_date=(
                desired_completion_date
            ),
            accepted_methods=(
                accepted_methods
            ),
            max_installment_months=(
                profile[
                    "max_installment_months"
                ]
            ),
        )
    )

    for candidate in installments:

        candidates.append(
            (
                candidate,
                baseline_flows,
            )
        )

    # =========================================================
    # WAIT
    # =========================================================

    wait = (
        _wait_candidate(
            request_date=(
                request_date
            ),
            desired_completion_date=(
                desired_completion_date
            ),
            requested_amount=(
                requested_amount
            ),
            deadline_safe_full_date=(
                deadline_safe_full
            ),
            accepted_methods=(
                accepted_methods
            ),
        )
    )

    if wait is not None:

        candidates.append(
            (
                wait,
                baseline_flows,
            )
        )

    # =========================================================
    # SPENDING-CHANGE FULL PAYMENT
    # =========================================================

    spending_candidates = (
        _spending_change_candidates(
            profile=(
                profile
            ),
            request_date=(
                request_date
            ),
            requested_amount=(
                requested_amount
            ),
            accepted_methods=(
                accepted_methods
            ),
            baseline_flows=(
                baseline_flows
            ),
        )
    )

    candidates.extend(
        spending_candidates
    )

    # =========================================================
    # VERIFY CANDIDATES
    # =========================================================

    safe_candidates: list[
        PaymentCandidate
    ] = []

    for (
        candidate,
        candidate_baseline,
    ) in candidates:

        if _candidate_is_safe(
            candidate=(
                candidate
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
            desired_completion_date=(
                desired_completion_date
            ),
            baseline_flows=(
                candidate_baseline
            ),
        ):

            safe_candidates.append(
                candidate
            )

    # =========================================================
    # SELECT BEST
    # =========================================================

    if safe_candidates:

        safe_candidates.sort(
            key=(
                _candidate_rank
            )
        )

        selected = (
            safe_candidates[0]
        )

        affordability_status = (
            selected.status
        )

        recommended_method = (
            selected.method
        )

        payment_plan = (
            _format_payment_plan(
                selected.payments
            )
        )

        spending_changes = (
            selected.spending_changes
        )

        explanation = (
            selected.explanation
        )

    else:

        affordability_status = (
            "not_affordable"
        )

        recommended_method = (
            "not_recommended"
        )

        payment_plan = (
            "none"
        )

        spending_changes = (
            "none"
        )

        if (
            earliest_full
            is None
        ):

            explanation = (
                "No eligible payment plan can complete the request "
                "safely while maintaining the minimum balance of "
                f"{home_currency} "
                f"{_fmt_money(minimum_balance)}."
            )

        elif (
            pd.Timestamp(
                earliest_full
            ).normalize()
            > desired_completion_date
        ):

            explanation = (
                "The conservative full-payment forecast becomes safe "
                f"only on {_fmt_date(earliest_full)}, which is after "
                "the desired completion date, and no other eligible "
                "plan is available."
            )

        else:

            explanation = (
                "No payment method accepted by the user provides a "
                "safe plan that completes the request by the desired "
                "date."
            )

    return {
        "request_id": (
            request_id
        ),
        "amount_safe_to_pay": (
            _fmt_money(
                safe_today
            )
        ),
        "affordability_status": (
            affordability_status
        ),
        "recommended_payment_method": (
            recommended_method
        ),
        "payment_plan": (
            payment_plan
        ),
        "earliest_date_for_full_payment": (
            _fmt_date(
                earliest_full
            )
        ),
        "spending_changes_needed": (
            spending_changes
        ),
        "decision_explanation": (
            explanation
        ),
    }


# =============================================================
# MAIN
# =============================================================


def main():

    print(
        "=" * 100
    )

    print(
        "BUY OR WAIT — FINANCIAL AGENT"
    )

    print(
        "=" * 100
    )

    bundle = (
        load_datasets()
    )

    fx = ExchangeRateBook(
        bundle.exchange_rates
    )

    output_rows = []

    total_requests = len(
        bundle.requests
    )

    for (
        index,
        request,
    ) in bundle.requests.iterrows():

        request_id = str(
            request[
                "request_id"
            ]
        )

        user_id = str(
            request[
                "user_id"
            ]
        )

        print(
            f"[{index + 1:03d}/"
            f"{total_requests:03d}] "
            f"{request_id}"
        )

        profile = (
            get_profile(
                profiles=(
                    bundle.profiles
                ),
                user_id=(
                    user_id
                ),
            )
        )

        user_events = (
            get_user_events(
                events=(
                    bundle.events
                ),
                user_id=(
                    user_id
                ),
            )
        )

        result = decide_request(
            request=request,
            profile=profile,
            user_events=(
                user_events
            ),
            payment_options=(
                bundle.payment_options
            ),
            messages=(
                bundle.messages
            ),
            fx=fx,
        )

        output_rows.append(
            result
        )

    output = pd.DataFrame(
        output_rows,
        columns=(
            OUTPUT_COLUMNS
        ),
    )

    output.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()

    print(
        "=" * 100
    )

    print(
        f"Generated {len(output)} decisions."
    )

    print(
        f"Output: {OUTPUT_PATH}"
    )

    print(
        "=" * 100
    )


if __name__ == "__main__":
    main()