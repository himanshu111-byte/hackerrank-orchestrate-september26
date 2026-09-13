from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import re

import pandas as pd

from money import money


@dataclass
class MessageEvidence:
    message_id: str
    user_id: str

    evidence_type: str

    amount: Decimal | None = None
    currency: str | None = None
    effective_date: pd.Timestamp | None = None

    related_event_id: str | None = None

    percentage_change: Decimal | None = None

    confirmed: bool = True

    raw_text: str = ""


CURRENCY_PATTERN = r"(INR|IDR|USD|EUR|ZAR)"

AMOUNT_PATTERN = (
    r"(?:INR|IDR|USD|EUR|ZAR)\s*"
    r"([0-9]+(?:[.,][0-9]+)?)"
)


def _parse_number(value: str) -> Decimal:
    # Dataset messages use '.' as decimal separator.
    # Remove comma thousands separators if present.
    value = value.replace(",", "")
    return money(value)


def parse_message(
    row: pd.Series,
) -> list[MessageEvidence]:

    text = str(
        row["message_text"]
    )

    lower = text.lower()

    message_id = row["message_id"]
    user_id = row["user_id"]

    related_event_id = (
        None
        if pd.isna(
            row["related_event_id"]
        )
        else str(
            row["related_event_id"]
        )
    )

    evidence = []

    # ---------------------------------------------------------
    # Explicit "still pending / not credited / not withdrawable"
    # ---------------------------------------------------------

    pending_phrases = [
        "still pending",
        "has not reached your account",
        "has not been credited",
        "not withdrawable",
        "belum disetujui",
        "belum dikonfirmasi",
        "masih menunggu",
    ]

    if any(
        phrase in lower
        for phrase in pending_phrases
    ):
        evidence.append(
            MessageEvidence(
                message_id=message_id,
                user_id=user_id,
                evidence_type=(
                    "unconfirmed_credit"
                ),
                related_event_id=(
                    related_event_id
                ),
                confirmed=False,
                raw_text=text,
            )
        )

    # ---------------------------------------------------------
    # Confirmed salary amount
    # ---------------------------------------------------------

    salary_markers = [
        "salary",
        "monthly pay",
        "gaji bulanan",
        "gaji pokok",
        "first salary",
    ]

    if any(
        marker in lower
        for marker in salary_markers
    ):

        currency_match = re.search(
            CURRENCY_PATTERN,
            text,
            flags=re.IGNORECASE,
        )

        amount_match = re.search(
            AMOUNT_PATTERN,
            text,
            flags=re.IGNORECASE,
        )

        if (
    currency_match
    and amount_match
):

            evidence.append(
                MessageEvidence(
                    message_id=message_id,
                    user_id=user_id,
                    evidence_type=(
                        "salary_amount"
                    ),
                    amount=_parse_number(
                        amount_match.group(1)
                    ),
                    currency=(
                        currency_match
                        .group(1)
                        .upper()
                    ),
                    raw_text=text,
                )
            )

    # ---------------------------------------------------------
    # Effective date YYYY-MM-DD
    # ---------------------------------------------------------

    date_match = re.search(
        r"\b(20\d{2}-\d{2}-\d{2})\b",
        text,
    )

    if date_match:
        effective_date = pd.Timestamp(
            date_match.group(1)
        )

        # Attach date to most recent relevant salary evidence.
        for item in evidence:
            if item.evidence_type in {
                "salary_amount",
            }:
                item.effective_date = (
                    effective_date
                )

    # ---------------------------------------------------------
    # Salary date replacement
    # ---------------------------------------------------------

    if (
        "salary" in lower
        and
        (
            "expected on" in lower
            or
            "confirmed credit date" in lower
        )
    ):
        date_match = re.search(
            r"\b(20\d{2}-\d{2}-\d{2})\b",
            text,
        )

        if date_match:
            evidence.append(
                MessageEvidence(
                    message_id=message_id,
                    user_id=user_id,
                    evidence_type=(
                        "salary_date"
                    ),
                    effective_date=pd.Timestamp(
                        date_match.group(1)
                    ),
                    raw_text=text,
                )
            )

    # ---------------------------------------------------------
    # Recurring rent percentage increase
    # ---------------------------------------------------------

    rent_change = re.search(
        r"rent by\s+([0-9]+(?:\.[0-9]+)?)%",
        lower,
    )

    if rent_change:
        evidence.append(
            MessageEvidence(
                message_id=message_id,
                user_id=user_id,
                evidence_type=(
                    "rent_percentage_change"
                ),
                percentage_change=(
                    _parse_number(
                        rent_change.group(1)
                    )
                    / Decimal("100")
                ),
                raw_text=text,
            )
        )

    # ---------------------------------------------------------
    # Contract ended / no future income confirmed
    # ---------------------------------------------------------

    if (
        "contract has ended" in lower
        and
        (
            "no off-season income" in lower
            or
            "no" in lower
            and "income" in lower
        )
    ):
        evidence.append(
            MessageEvidence(
                message_id=message_id,
                user_id=user_id,
                evidence_type=(
                    "income_ended"
                ),
                raw_text=text,
            )
        )

    # ---------------------------------------------------------
    # Internal transfer:
    # matching debit and credit should net to zero.
    # ---------------------------------------------------------

    if (
        "transfer between your two accounts"
        in lower
    ):
        evidence.append(
            MessageEvidence(
                message_id=message_id,
                user_id=user_id,
                evidence_type=(
                    "internal_transfer"
                ),
                raw_text=text,
            )
        )

    # ---------------------------------------------------------
    # Confirmed one-time proceeds already reached account.
    # Do not project another occurrence.
    # ---------------------------------------------------------

    if (
        "proceeds have reached your account"
        in lower
        and
        "no further scheduled payments"
        in lower
    ):
        evidence.append(
            MessageEvidence(
                message_id=message_id,
                user_id=user_id,
                evidence_type=(
                    "one_time_income_completed"
                ),
                related_event_id=(
                    related_event_id
                ),
                raw_text=text,
            )
        )

    return evidence


def extract_message_evidence(
    messages: pd.DataFrame,
    user_id: str,
    request_id: str | None = None,
) -> list[MessageEvidence]:

    rows = messages[
        messages["user_id"] == user_id
    ].copy()

    # User-level messages matter even when request_id is blank.
    # Request-specific messages for another request should not leak.
    if request_id is not None:
        rows = rows[
            rows["request_id"].isna()
            |
            (
                rows["request_id"]
                == request_id
            )
        ]

    result = []

    for _, row in rows.iterrows():
        result.extend(
            parse_message(row)
        )

    return result