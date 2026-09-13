from __future__ import annotations

from dataclasses import replace

import pandas as pd

from evidence import MessageEvidence
from money import money
from salary_resolver import SalaryStream


# =============================================================
# SALARY MESSAGE SEMANTICS
# =============================================================

# These phrases indicate that the amount applies only to the
# next / affected payroll cycle rather than permanently changing
# the employee's normal recurring salary.
TEMPORARY_SALARY_MARKERS = {
    "next salary",
    "next payroll",
    "temporary monthly pay",
    "temporary pay",
    "affected pay cycle",
    "approved unpaid leave",
}

# These phrases explicitly establish a continuing/base salary.
PERSISTENT_SALARY_MARKERS = {
    "base salary",
    "gaji pokok",
    "monthly base salary",
}

# These phrases establish the first salary payment.
FIRST_SALARY_MARKERS = {
    "first salary",
}


def _evidence_text(
    item: MessageEvidence,
) -> str:
    """
    Return normalized raw message text for semantic classification.
    """

    return str(
        item.raw_text or ""
    ).strip().lower()


def _is_temporary_salary(
    item: MessageEvidence,
) -> bool:
    """
    True when the message describes a temporary or next-payroll
    salary amount.
    """

    text = _evidence_text(item)

    return any(
        marker in text
        for marker in TEMPORARY_SALARY_MARKERS
    )


def _is_persistent_salary(
    item: MessageEvidence,
) -> bool:
    """
    True when the message explicitly describes a continuing
    base salary.
    """

    text = _evidence_text(item)

    return any(
        marker in text
        for marker in PERSISTENT_SALARY_MARKERS
    )


def _is_first_salary(
    item: MessageEvidence,
) -> bool:
    """
    True when the message describes the employee's first salary.
    """

    text = _evidence_text(item)

    return any(
        marker in text
        for marker in FIRST_SALARY_MARKERS
    )


def apply_salary_evidence(
    stream: SalaryStream | None,
    evidence: list[MessageEvidence],
    request_date: pd.Timestamp,
) -> SalaryStream | None:
    """
    Apply explicit message evidence to an inferred salary stream.

    Precedence / semantics:

    1. Explicitly ended income
       -> no future salary projection.

    2. Temporary / next-payroll salary
       -> override exactly one future payroll occurrence.

    3. First salary with confirmed date
       -> establish salary amount and payroll-cycle anchor.

    4. Explicit base salary
       -> permanently replace inferred recurring salary amount.

    5. Ambiguous confirmed salary amount
       -> conservatively treat as a one-payroll override when
          an existing historical salary stream is available.

    Explicit confirmed evidence takes precedence over historical
    inference, but temporary evidence must not permanently replace
    a stable historical base salary.
    """

    request_date = pd.Timestamp(
        request_date
    )

    # =========================================================
    # 1. EMPLOYMENT / RECURRING INCOME EXPLICITLY ENDED
    # =========================================================

    if any(
        item.evidence_type == "income_ended"
        for item in evidence
    ):
        return None

    # =========================================================
    # 2. COLLECT CONFIRMED SALARY EVIDENCE
    # =========================================================

    salary_amount_items = [
        item
        for item in evidence
        if (
            item.evidence_type == "salary_amount"
            and item.confirmed
            and item.amount is not None
        )
    ]

    salary_date_items = [
        item
        for item in evidence
        if (
            item.evidence_type == "salary_date"
            and item.confirmed
            and item.effective_date is not None
        )
    ]

    # Nothing explicit to modify.
    if (
        not salary_amount_items
        and not salary_date_items
    ):
        return stream

    newest_amount = (
        salary_amount_items[-1]
        if salary_amount_items
        else None
    )

    newest_date = (
        salary_date_items[-1]
        if salary_date_items
        else None
    )

    # =========================================================
    # 3. RESOLVE EXPLICIT EFFECTIVE DATE
    # =========================================================

    effective_date = None

    if (
        newest_amount is not None
        and newest_amount.effective_date is not None
    ):
        effective_date = pd.Timestamp(
            newest_amount.effective_date
        )

    elif newest_date is not None:
        effective_date = pd.Timestamp(
            newest_date.effective_date
        )

    # =========================================================
    # 4. NO HISTORICAL SALARY STREAM
    # =========================================================
    #
    # If salary history could not establish a recurring stream,
    # explicit evidence must contain both an amount/currency and
    # enough timing information before we create one.
    #
    # We deliberately do not invent a payroll date.
    # =========================================================

    if stream is None:

        if newest_amount is None:
            return None

        currency = newest_amount.currency

        if currency is None:
            return None

        if effective_date is None:
            return None

        # last_date is deliberately one month before the
        # confirmed effective date so salary generation produces
        # the effective date as the first occurrence.
        return SalaryStream(
            amount=money(
                newest_amount.amount
            ),
            currency=currency,
            last_date=(
                effective_date
                - pd.DateOffset(months=1)
            ),
            description=(
                "confirmed salary from message"
            ),
            cadence_days=30,
            monthly=True,
        )

    # =========================================================
    # 5. HISTORICAL STREAM EXISTS
    # =========================================================

    updated = stream

    if newest_amount is not None:

        # -----------------------------------------------------
        # A. TEMPORARY / NEXT-PAYROLL SALARY
        # -----------------------------------------------------
        #
        # Examples from the dataset:
        #
        # "Your next salary is reduced to EUR ..."
        #
        # "Your temporary monthly pay is EUR ...
        #  The reduced amount continues for the next payroll."
        #
        # These must NOT permanently replace the historical
        # recurring base salary.
        # -----------------------------------------------------

        if _is_temporary_salary(
            newest_amount
        ):

            updated = replace(
                updated,
                next_amount_override=money(
                    newest_amount.amount
                ),
                next_currency_override=(
                    newest_amount.currency
                    or updated.currency
                ),
                next_override_date=(
                    effective_date
                ),
            )

        # -----------------------------------------------------
        # B. FIRST SALARY
        # -----------------------------------------------------
        #
        # Example:
        #
        # "Your first salary will be EUR 1661.
        #  The confirmed credit date is 2026-01-15."
        #
        # The explicit amount establishes the salary and the
        # confirmed date anchors the payroll cycle.
        # -----------------------------------------------------

        elif _is_first_salary(
            newest_amount
        ):

            updated = replace(
                updated,
                amount=money(
                    newest_amount.amount
                ),
                currency=(
                    newest_amount.currency
                    or updated.currency
                ),
            )

            if effective_date is not None:
                updated = replace(
                    updated,
                    last_date=(
                        effective_date
                        - pd.DateOffset(
                            months=1
                        )
                    ),
                )

        # -----------------------------------------------------
        # C. CONFIRMED BASE / PERSISTENT SALARY
        # -----------------------------------------------------
        #
        # Example:
        #
        # "Gaji pokok yang dikonfirmasi adalah
        #  IDR 38760000."
        #
        # This explicitly establishes the recurring base salary.
        # -----------------------------------------------------

        elif _is_persistent_salary(
            newest_amount
        ):

            updated = replace(
                updated,
                amount=money(
                    newest_amount.amount
                ),
                currency=(
                    newest_amount.currency
                    or updated.currency
                ),
            )

            if effective_date is not None:
                updated = replace(
                    updated,
                    last_date=(
                        effective_date
                        - pd.DateOffset(
                            months=1
                        )
                    ),
                )

        # -----------------------------------------------------
        # D. AMBIGUOUS CONFIRMED SALARY AMOUNT
        # -----------------------------------------------------
        #
        # When an existing historical salary stream is available
        # but the message does not clearly establish that the new
        # amount is permanent, use the safer interpretation:
        # change only the next payroll.
        # -----------------------------------------------------

        else:

            updated = replace(
                updated,
                next_amount_override=money(
                    newest_amount.amount
                ),
                next_currency_override=(
                    newest_amount.currency
                    or updated.currency
                ),
                next_override_date=(
                    effective_date
                ),
            )

    # =========================================================
    # 6. DATE-ONLY SALARY EVIDENCE
    # =========================================================
    #
    # A confirmed salary date can re-anchor the payroll cycle.
    #
    # Do NOT re-anchor a temporary salary override unless the
    # temporary evidence itself explicitly provides the date.
    # =========================================================

    if (
        newest_date is not None
        and (
            newest_amount is None
            or not _is_temporary_salary(
                newest_amount
            )
        )
    ):

        updated = replace(
            updated,
            last_date=(
                pd.Timestamp(
                    newest_date.effective_date
                )
                - pd.DateOffset(
                    months=1
                )
            ),
        )

    return updated