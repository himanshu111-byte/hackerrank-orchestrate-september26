from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
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
    """
    Output monetary values without unnecessary trailing zeroes.

    Examples:

        25256.00      -> 25256
        603.30        -> 603.3
        15952906.67   -> 15952906.67
    """

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
) -> list[CashFlow]:

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
                date=pd.Timestamp(
                    payment_date
                ).normalize(),
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
    Verify a concrete candidate payment plan through the user's
    desired completion date.

    Important distinction:

    - The financial baseline itself still spans 90 days.
    - amount_safe_to_pay remains based on the full 90-day baseline.
    - earliest_date_for_full_payment remains independently calculated
      by capacity.py.
    - Candidate-plan feasibility is evaluated from request_date
      through desired_completion_date.

    A candidate is safe only when:

    1. the candidate completes by desired_completion_date, and
    2. the forecast balance never falls below minimum_balance during
       that completion horizon.
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
        >= money(
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
    """
    Find the earliest date from request_date through
    desired_completion_date on which the complete requested amount
    can be paid while remaining above the minimum balance through the
    user's desired completion horizon.

    This is intentionally separate from
    earliest_full_payment_date(), because that function supplies the
    independent conservative 90-day output field.
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
                if candidate_date
                == request_date
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

    # Full payment means the whole requested amount is safe today.
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
            "The full requested amount "
            "can be paid today while "
            "maintaining the required "
            "minimum balance through the "
            "desired completion horizon."
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

    # Waiting is useful only when the amount is not safe today,
    # but becomes safe on a later date.
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
            "The full amount is not safe "
            "to pay today, but becomes safe "
            "on "
            f"{_fmt_date(deadline_safe_full_date)} "
            "while maintaining the required "
            "minimum balance through the "
            "desired completion horizon."
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

    # Exact challenge requirement:
    #
    # payment 1 = amount_safe_to_pay today
    # payment 2 = requested_amount - amount_safe_to_pay
    #
    # Do not optimize or recalculate the first payment.
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
            f"Pay {_fmt_money(safe_today)} "
            "today and the remaining "
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
    """
    Supplied options are approximately monthly schedules.

    max_installment_months therefore acts as the maximum allowed
    supplied installment duration / monthly payment count.
    """

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
) -> list[PaymentCandidate]:

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

        # Preserve the supplied payment schedule exactly.
        #
        # Do not recalculate or modify the final installment.
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
                    f"{number_of_payments}-payment "
                    "installment option ending "
                    f"{_fmt_date(final_payment_date)}."
                ),
            )
        )

    return candidates


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
    Challenge ranking after eligibility and safety validation:

    1. complete request by desired date
       (already enforced before ranking)
    2. no spending changes
    3. minimize total amount paid
    4. start earlier
    5. fewer payments
    6. lowest payment_option_id
    """

    requires_changes = (
        candidate.spending_changes
        != "none"
    )

    return (
        1 if requires_changes else 0,
        candidate.total_paid,
        candidate.first_payment_date,
        candidate.payment_count,
        _option_id_rank(
            candidate.payment_option_id
        ),
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
    #
    # Build the baseline once per request.
    #
    # The baseline itself remains a full 90-day forecast.
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
    #
    # These remain independent from payment-plan selection.
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
    # DEADLINE-AWARE FULL-PAYMENT FEASIBILITY
    # =========================================================
    #
    # This value is used only to choose full-payment / wait
    # recommendations.
    #
    # It deliberately does NOT replace earliest_full, because
    # earliest_full remains the conservative 90-day output field.
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

    # =========================================================
    # GENERATE ELIGIBLE CANDIDATES
    # =========================================================

    candidates: list[
        PaymentCandidate
    ] = []

    # ---------------------------------------------------------
    # Full payment
    # ---------------------------------------------------------

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
            full
        )

    # ---------------------------------------------------------
    # Partial payment
    # ---------------------------------------------------------
    #
    # Partial payment continues to use the independent challenge
    # fields:
    #
    # first payment = amount_safe_to_pay
    # second payment = remainder on earliest full-payment date
    # ---------------------------------------------------------

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
            partial
        )

    # ---------------------------------------------------------
    # Installments
    # ---------------------------------------------------------

    candidates.extend(
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

    # ---------------------------------------------------------
    # Wait
    # ---------------------------------------------------------

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
            wait
        )

    # =========================================================
    # VERIFY EVERY CANDIDATE
    # =========================================================
    #
    # Candidate schedules are inserted into the same baseline.
    #
    # Candidate-plan safety is evaluated through the desired
    # completion date rather than the entire later 90-day period.
    # =========================================================

    safe_candidates = []

    for candidate in candidates:

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

            safe_candidates.append(
                candidate
            )

    # =========================================================
    # RANK SAFE CANDIDATES
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
            selected
            .spending_changes
        )

        explanation = (
            selected
            .explanation
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
            deadline_safe_full
            is not None
            and "full_payment"
            not in accepted_methods
        ):

            explanation = (
                "The requested amount could "
                "be completed safely, but "
                "full payment is not among "
                "the user's accepted payment "
                "methods and no other eligible "
                "plan is safe."
            )

        elif (
            earliest_full
            is None
        ):

            explanation = (
                "No eligible payment plan "
                "can complete the request "
                "safely while maintaining "
                f"the minimum balance of "
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
                "The conservative full-payment "
                "forecast becomes safe only on "
                f"{_fmt_date(earliest_full)}, "
                "which is after the desired "
                "completion date, and no other "
                "eligible plan is available."
            )

        else:

            explanation = (
                "No payment method accepted "
                "by the user provides a safe "
                "plan that completes the "
                "request by the desired date."
            )

    # =========================================================
    # FINAL OUTPUT ROW
    # =========================================================

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

    # Preserve deterministic dataset order.
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
        f"Generated {len(output)} "
        f"decisions."
    )

    print(
        f"Output: {OUTPUT_PATH}"
    )

    print(
        "=" * 100
    )


if __name__ == "__main__":
    main()