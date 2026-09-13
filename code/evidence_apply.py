from __future__ import annotations

from decimal import Decimal

import pandas as pd

from evidence import MessageEvidence
from recurrence import RecurringSeries
from money import money


def apply_message_evidence_to_series(
    series_list: list[RecurringSeries],
    evidence: list[MessageEvidence],
) -> list[RecurringSeries]:

    updated = list(series_list)

    income_ended = any(
        item.evidence_type
        == "income_ended"
        for item in evidence
    )

    if income_ended:
        updated = [
            s
            for s in updated
            if s.direction != "credit"
            or s.category != "salary"
        ]

    salary_amounts = [
        item
        for item in evidence
        if (
            item.evidence_type
            == "salary_amount"
            and item.confirmed
            and item.amount is not None
        )
    ]

    if salary_amounts:

        newest = salary_amounts[-1]

        for series in updated:

            if (
                series.direction
                == "credit"
                and
                series.category
                == "salary"
            ):
                series.amount = money(
                    newest.amount
                )

                if (
                    newest.effective_date
                    is not None
                    and
                    newest.effective_date
                    > series.last_date
                ):
                    # Set last_date so next generated occurrence
                    # begins from the new effective cycle.
                    series.last_date = (
                        newest.effective_date
                        - pd.DateOffset(
                            months=1
                        )
                        if series.monthly
                        else
                        newest.effective_date
                        - pd.Timedelta(
                            days=(
                                series.cadence_days
                                or 30
                            )
                        )
                    )

    rent_changes = [
        item
        for item in evidence
        if (
            item.evidence_type
            == "rent_percentage_change"
            and
            item.percentage_change
            is not None
        )
    ]

    if rent_changes:

        change = rent_changes[-1]

        for series in updated:

            if (
                series.direction
                == "debit"
                and
                series.category
                == "rent"
            ):
                series.amount = (
                    series.amount
                    *
                    (
                        Decimal("1")
                        +
                        change.percentage_change
                    )
                )

    return updated