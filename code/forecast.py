from dataclasses import dataclass
from decimal import Decimal

import pandas as pd

from evidence import (
    extract_message_evidence,
)

from evidence_apply import (
    apply_message_evidence_to_series,
)

from lifecycle import (
    CashFlow,
    future_explicit_cashflows,
)

from money import (
    ExchangeRateBook,
    money,
)

from recurrence import (
    generate_recurring_cashflows,
    infer_recurring_series,
)

from salary_resolver import (
    generate_salary_occurrences,
    infer_stable_salary_stream,
)

from salary_evidence import (
    apply_salary_evidence,
)


FORECAST_DAYS = 90


@dataclass
class ForecastResult:
    request_date: pd.Timestamp
    forecast_end: pd.Timestamp
    opening_balance: Decimal

    minimum_balance_required: Decimal
    minimum_forecast_balance: Decimal
    minimum_forecast_date: pd.Timestamp

    flows: list[CashFlow]
    daily: pd.DataFrame


def build_baseline_cashflows(
    events: pd.DataFrame,
    request_date: pd.Timestamp,
    home_currency: str,
    fx: ExchangeRateBook,
    messages: pd.DataFrame | None = None,
    user_id: str | None = None,
    request_id: str | None = None,
) -> list[CashFlow]:

    request_date = pd.Timestamp(
        request_date
    )

    forecast_end = (
        request_date
        + pd.Timedelta(
            days=FORECAST_DAYS
        )
    )

    # =========================================================
    # 1. EXPLICIT FUTURE CASH FLOWS
    # =========================================================

    explicit = (
        future_explicit_cashflows(
            events=events,
            request_date=request_date,
            forecast_end=forecast_end,
            home_currency=home_currency,
            fx=fx,
        )
    )

    # =========================================================
    # 2. MESSAGE EVIDENCE
    # =========================================================
    #
    # Extract once and reuse for both:
    #
    # - recurring expense adjustments
    # - salary overrides / salary termination
    # =========================================================

    evidence = []

    if (
        messages is not None
        and user_id is not None
    ):
        evidence = (
            extract_message_evidence(
                messages=messages,
                user_id=user_id,
                request_id=request_id,
            )
        )

    # =========================================================
    # 3. GENERIC RECURRING FLOWS
    # =========================================================
    #
    # IMPORTANT:
    #
    # recurrence.py now excludes salary.
    #
    # This section should therefore handle recurring:
    #
    # - expenses
    # - subscriptions
    # - debt payments
    #
    # Salary is handled separately below.
    # =========================================================

    recurring_series = (
        infer_recurring_series(
            events=events,
            request_date=request_date,
            home_currency=home_currency,
            fx=fx,
        )
    )

    if evidence:
        recurring_series = (
            apply_message_evidence_to_series(
                recurring_series,
                evidence,
            )
        )

    recurring = (
        generate_recurring_cashflows(
            series_list=recurring_series,
            request_date=request_date,
            forecast_end=forecast_end,
            explicit_flows=explicit,
        )
    )

    # Start with all non-generated-salary flows.
    flows: list[CashFlow] = (
        list(explicit)
        + list(recurring)
    )

    # =========================================================
    # 4. DEDICATED SALARY RESOLVER
    # =========================================================
    #
    # This avoids mixing:
    #
    # - base salary
    # - commission
    # - bonus
    # - arrears
    # - reimbursements
    # - adjustments
    #
    # Only stable / confirmed salary gets projected.
    # =========================================================

    salary_stream = (
        infer_stable_salary_stream(
            events=events,
            request_date=request_date,
        )
    )

    salary_stream = (
        apply_salary_evidence(
            stream=salary_stream,
            evidence=evidence,
            request_date=request_date,
        )
    )

    # =========================================================
    # 5. GENERATE FUTURE SALARY FLOWS
    # =========================================================

    if salary_stream is not None:

        salary_occurrences = (
            generate_salary_occurrences(
                stream=salary_stream,
                request_date=request_date,
                forecast_end=forecast_end,
                home_currency=home_currency,
                fx=fx,
            )
        )

        for (
            salary_date,
            salary_amount,
        ) in salary_occurrences:

            salary_date = (
                pd.Timestamp(
                    salary_date
                )
                .normalize()
            )

            salary_amount = money(
                salary_amount
            )

            # -------------------------------------------------
            # Explicit salary takes precedence.
            #
            # Example:
            #
            # request_25 already contains an explicit salary
            # around 2024-03-15.
            #
            # We must suppress ONLY that generated occurrence,
            # not every later salary occurrence.
            # -------------------------------------------------

            duplicate_salary = False

            for flow in explicit:

                flow_category = str(
                    getattr(
                        flow,
                        "category",
                        "",
                    )
                ).strip().lower()

                if flow_category != "salary":
                    continue

                flow_date = (
                    pd.Timestamp(
                        flow.date
                    )
                    .normalize()
                )

                date_difference = abs(
                    (
                        flow_date
                        - salary_date
                    ).days
                )

                if date_difference <= 2:
                    duplicate_salary = True
                    break

            if duplicate_salary:
                continue

            # Salary must always be a positive cash inflow.
            if salary_amount <= Decimal("0"):
                continue

            flows.append(
                CashFlow(
                    date=salary_date,
                    amount=salary_amount,
                    source="recurring_salary",
                    category="salary",
                    flexibility="fixed",
                )
            )

    # =========================================================
    # 6. RETURN CHRONOLOGICALLY SORTED FLOWS
    # =========================================================

    return sorted(
        flows,
        key=lambda flow: (
            pd.Timestamp(
                flow.date
            ),
            str(
                flow.source
            ),
        ),
    )


def run_forecast(
    opening_balance,
    minimum_balance,
    request_date: pd.Timestamp,
    flows: list[CashFlow],
    additional_flows: list[
        CashFlow
    ] | None = None,
) -> ForecastResult:

    request_date = pd.Timestamp(
        request_date
    )

    forecast_end = (
        request_date
        + pd.Timedelta(
            days=FORECAST_DAYS
        )
    )

    opening_balance = money(
        opening_balance
    )

    minimum_balance = money(
        minimum_balance
    )

    all_flows = list(
        flows
    )

    if additional_flows:
        all_flows.extend(
            additional_flows
        )

    # =========================================================
    # GROUP CASH MOVEMENT BY DATE
    # =========================================================

    by_date: dict[
        pd.Timestamp,
        Decimal,
    ] = {}

    for flow in all_flows:

        flow_date = (
            pd.Timestamp(
                flow.date
            )
            .normalize()
        )

        by_date[flow_date] = (
            by_date.get(
                flow_date,
                Decimal("0"),
            )
            + flow.amount
        )

    # =========================================================
    # DAILY BALANCE SIMULATION
    # =========================================================

    balance = opening_balance

    records = []

    current = (
        request_date
        .normalize()
    )

    forecast_end_normalized = (
        forecast_end
        .normalize()
    )

    while (
        current
        <= forecast_end_normalized
    ):

        net_flow = (
            by_date.get(
                current,
                Decimal("0"),
            )
        )

        balance += net_flow

        records.append(
            {
                "date": current,
                "net_flow": float(
                    net_flow
                ),
                "balance": float(
                    balance
                ),
            }
        )

        current += pd.Timedelta(
            days=1
        )

    daily = pd.DataFrame(
        records
    )

    # =========================================================
    # MINIMUM BALANCE DURING THE 90-DAY WINDOW
    # =========================================================

    minimum_index = (
        daily["balance"]
        .idxmin()
    )

    minimum_forecast_balance = (
        money(
            daily.loc[
                minimum_index,
                "balance",
            ]
        )
    )

    minimum_date = (
        pd.Timestamp(
            daily.loc[
                minimum_index,
                "date",
            ]
        )
    )

    return ForecastResult(
        request_date=request_date,
        forecast_end=forecast_end,
        opening_balance=opening_balance,
        minimum_balance_required=(
            minimum_balance
        ),
        minimum_forecast_balance=(
            minimum_forecast_balance
        ),
        minimum_forecast_date=(
            minimum_date
        ),
        flows=all_flows,
        daily=daily,
    )


def is_safe(
    result: ForecastResult,
) -> bool:

    return (
        result.minimum_forecast_balance
        >=
        result.minimum_balance_required
    )