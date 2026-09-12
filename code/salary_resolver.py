from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from statistics import median
import re

import pandas as pd

from money import ExchangeRateBook, money


VARIABLE_INCOME_MARKERS = {
    "commission",
    "bonus",
    "arrears",
    "adjustment",
    "overtime",
    "incentive",
    "reimbursement",
    "refund",
    "award",
    "prize",
    "windfall",
}


FINAL_PAYROLL_MARKERS = {
    "final employer payroll",
    "final payroll",
    "final salary",
    "last salary",
    "final paycheck",
    "last paycheck",
}


UNCONFIRMED_FUTURE_SALARY_MARKERS = {
    "unconfirmed",
    "not confirmed",
    "not approved",
    "awaiting approval",
    "estimated",
    "estimate",
    "forecast",
    "possible",
    "potential",
    "expected bonus",
    "provisional",
}


CONFIRMED_FUTURE_SALARY_MARKERS = {
    "confirmed salary",
    "next confirmed salary",
    "confirmed payroll",
    "scheduled salary",
    "scheduled payroll",
    "payroll credit",
    "salary credit",
}


CONFIRMED_FUTURE_STATUSES = {
    "confirmed",
    "scheduled",
}


@dataclass
class SalaryStream:
    amount: Decimal
    currency: str
    last_date: pd.Timestamp
    description: str

    cadence_days: int = 30
    monthly: bool = True

    # Optional override for exactly one future payroll cycle.
    next_amount_override: Decimal | None = None
    next_currency_override: str | None = None
    next_override_date: pd.Timestamp | None = None


def normalize_description(
    value: str,
) -> str:
    value = str(
        value
    ).strip().lower()

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value


def is_variable_salary_description(
    description: str,
) -> bool:

    text = normalize_description(
        description
    )

    return any(
        marker in text
        for marker in VARIABLE_INCOME_MARKERS
    )


def is_final_payroll_description(
    description: str,
) -> bool:

    text = normalize_description(
        description
    )

    return any(
        marker in text
        for marker in FINAL_PAYROLL_MARKERS
    )


def is_unconfirmed_future_salary_description(
    description: str,
) -> bool:

    text = normalize_description(
        description
    )

    return any(
        marker in text
        for marker in UNCONFIRMED_FUTURE_SALARY_MARKERS
    )


def is_confirmed_future_salary_description(
    description: str,
) -> bool:

    text = normalize_description(
        description
    )

    return any(
        marker in text
        for marker in CONFIRMED_FUTURE_SALARY_MARKERS
    )


def _is_monthly_dates(
    dates: list[pd.Timestamp],
) -> bool:

    if len(dates) < 3:
        return False

    dates = sorted(
        pd.Timestamp(d)
        for d in dates
    )

    gaps = [
        (
            dates[i]
            - dates[i - 1]
        ).days
        for i in range(
            1,
            len(dates)
        )
    ]

    if not gaps:
        return False

    monthly_gaps = [
        gap
        for gap in gaps
        if 27 <= gap <= 32
    ]

    return (
        len(monthly_gaps)
        >= max(
            2,
            len(gaps) - 1,
        )
    )


def _infer_stable_amount(
    amounts: list[Decimal],
) -> Decimal | None:
    """
    Infer the dominant recurring salary amount while tolerating
    occasional temporary reductions or adjustments.

    Example:

        1422.85
        1422.85
        1422.85
        1422.85
        782.57

    The stable salary should still be inferred as 1422.85.
    """

    if len(amounts) < 3:
        return None

    values = [
        money(value)
        for value in amounts
    ]

    clusters: list[
        list[Decimal]
    ] = []

    for value in values:

        placed = False

        for cluster in clusters:

            centre = money(
                median(
                    [
                        float(x)
                        for x in cluster
                    ]
                )
            )

            # Allow small payroll variation while keeping
            # materially different temporary amounts separate.
            tolerance = max(
                Decimal("0.01"),
                abs(centre)
                * Decimal("0.02"),
            )

            if (
                abs(
                    value
                    - centre
                )
                <= tolerance
            ):
                cluster.append(
                    value
                )
                placed = True
                break

        if not placed:
            clusters.append(
                [value]
            )

    # Prefer the salary value supported by the largest number
    # of historical observations.
    clusters.sort(
        key=lambda cluster: (
            len(cluster),
            max(
                float(value)
                for value in cluster
            ),
        ),
        reverse=True,
    )

    best_cluster = clusters[0]

    # Require repeated historical evidence before calling an
    # amount a stable recurring salary.
    if len(best_cluster) < 3:
        return None

    stable_amount = money(
        median(
            [
                float(value)
                for value in best_cluster
            ]
        )
    )

    if stable_amount <= 0:
        return None

    return stable_amount


def _prepare_salary_dataframe(
    events: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize event dates and isolate salary-credit events.
    """

    if events.empty:
        return events.copy()

    df = events.copy()

    df["event_date"] = pd.to_datetime(
        df["event_date"],
        errors="coerce",
    )

    if "settlement_date" in df.columns:
        df["settlement_date"] = pd.to_datetime(
            df["settlement_date"],
            errors="coerce",
        )

    salary = df[
        (
            df["event_type"]
            .astype(str)
            .str.lower()
            == "income"
        )
        &
        (
            df["category"]
            .astype(str)
            .str.lower()
            == "salary"
        )
        &
        (
            df["direction"]
            .astype(str)
            .str.lower()
            == "credit"
        )
    ].copy()

    salary = salary[
        salary["event_date"].notna()
    ].copy()

    salary = salary[
        salary["amount"].notna()
    ].copy()

    return salary


def _infer_confirmed_future_salary_stream(
    salary: pd.DataFrame,
    request_date: pd.Timestamp,
) -> SalaryStream | None:
    """
    Seed a recurring salary stream from a confirmed future payroll
    when historical salary exists but there is not yet enough
    settled history to satisfy the normal >=3 observation rule.

    This is intentionally conservative.

    Requirements:
      - at least one earlier settled non-variable salary exists;
      - no latest historical final-payroll signal;
      - future event is a salary credit;
      - future event is either explicitly scheduled/confirmed OR
        its description clearly says that the salary/payroll is
        confirmed;
      - pending/unconfirmed/estimated salary descriptions are
        rejected;
      - amount and currency must be known.

    The confirmed future occurrence becomes the stream anchor.
    Generation therefore starts one month AFTER that explicit
    salary event, avoiding double-counting it.
    """

    request_date = pd.Timestamp(
        request_date
    )

    historical = salary[
        (
            salary["event_date"]
            <= request_date
        )
        &
        (
            salary["status"]
            .astype(str)
            .str.lower()
            == "settled"
        )
    ].copy()

    if historical.empty:
        return None

    historical = historical[
        ~historical[
            "description"
        ].apply(
            is_variable_salary_description
        )
    ].copy()

    if historical.empty:
        return None

    latest_historical = (
        historical
        .sort_values(
            "event_date"
        )
        .iloc[-1]
    )

    if is_final_payroll_description(
        latest_historical[
            "description"
        ]
    ):
        return None

    future = salary[
        salary["event_date"]
        > request_date
    ].copy()

    if future.empty:
        return None

    future = future[
        ~future[
            "description"
        ].apply(
            is_variable_salary_description
        )
    ].copy()

    future = future[
        ~future[
            "description"
        ].apply(
            is_final_payroll_description
        )
    ].copy()

    future = future[
        ~future[
            "description"
        ].apply(
            is_unconfirmed_future_salary_description
        )
    ].copy()

    if future.empty:
        return None

    future["_status_norm"] = (
        future["status"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    future["_description_confirmed"] = (
        future["description"]
        .apply(
            is_confirmed_future_salary_description
        )
    )

    future = future[
        (
            future["_status_norm"]
            .isin(
                CONFIRMED_FUTURE_STATUSES
            )
        )
        |
        (
            future[
                "_description_confirmed"
            ]
        )
    ].copy()

    if future.empty:
        return None

    future = (
        future
        .sort_values(
            "event_date"
        )
        .copy()
    )

    # Prefer the earliest confirmed future payroll occurrence.
    row = future.iloc[0]

    amount = money(
        row["amount"]
    )

    if amount <= 0:
        return None

    currency = str(
        row["currency"]
    ).strip().upper()

    if (
        not currency
        or currency == "NAN"
    ):
        return None

    return SalaryStream(
        amount=amount,
        currency=currency,
        last_date=pd.Timestamp(
            row["event_date"]
        ),
        description=normalize_description(
            row["description"]
        ),
        cadence_days=30,
        monthly=True,
    )


def infer_stable_salary_stream(
    events: pd.DataFrame,
    request_date: pd.Timestamp,
) -> SalaryStream | None:
    """
    Infer guaranteed recurring salary separately from bonuses,
    commissions and other variable income.

    Resolution order:

    1. Prefer a genuine stable historical monthly salary stream.
    2. If historical salary exists but is too short to establish a
       stable stream, allow a confirmed future payroll event to
       seed the recurring monthly stream.

    Temporary salary/message overrides remain the responsibility
    of salary_evidence.py.
    """

    if events.empty:
        return None

    request_date = pd.Timestamp(
        request_date
    )

    salary_all = (
        _prepare_salary_dataframe(
            events
        )
    )

    if salary_all.empty:
        return None

    # =========================================================
    # HISTORICAL SETTLED SALARY
    # =========================================================

    salary = salary_all[
        (
            salary_all["status"]
            .astype(str)
            .str.lower()
            == "settled"
        )
        &
        (
            salary_all["event_date"]
            <= request_date
        )
    ].copy()

    if salary.empty:
        return None

    # =========================================================
    # EXPLICIT FINAL PAYROLL
    # =========================================================
    #
    # If the latest settled salary explicitly states that it is
    # the final payroll / final salary, do not project future
    # recurring salary.
    # =========================================================

    latest_salary = (
        salary
        .sort_values(
            "event_date"
        )
        .iloc[-1]
    )

    if is_final_payroll_description(
        latest_salary[
            "description"
        ]
    ):
        return None

    # =========================================================
    # REMOVE VARIABLE / NON-GUARANTEED SALARY-LIKE INCOME
    # =========================================================

    salary = salary[
        ~salary[
            "description"
        ].apply(
            is_variable_salary_description
        )
    ].copy()

    salary = salary[
        salary[
            "amount"
        ].notna()
    ].copy()

    if salary.empty:
        return None

    salary[
        "_description_norm"
    ] = (
        salary[
            "description"
        ]
        .apply(
            normalize_description
        )
    )

    candidates = []

    # =========================================================
    # BUILD CANDIDATE HISTORICAL BASE-SALARY STREAMS
    # =========================================================

    for (
        description,
        group,
    ) in salary.groupby(
        "_description_norm"
    ):

        group = (
            group
            .sort_values(
                "event_date"
            )
            .copy()
        )

        if len(group) < 3:
            continue

        dates = (
            group[
                "event_date"
            ]
            .dropna()
            .tolist()
        )

        if not _is_monthly_dates(
            dates
        ):
            continue

        amounts = [
            money(value)
            for value in group[
                "amount"
            ].tolist()
            if pd.notna(
                value
            )
        ]

        if len(amounts) < 3:
            continue

        stable_amount = (
            _infer_stable_amount(
                amounts
            )
        )

        if stable_amount is None:
            continue

        currencies = (
            group[
                "currency"
            ]
            .dropna()
            .astype(str)
            .str.upper()
            .unique()
            .tolist()
        )

        if len(currencies) != 1:
            continue

        candidates.append(
            (
                len(group),
                group[
                    "event_date"
                ].max(),
                SalaryStream(
                    amount=stable_amount,
                    currency=(
                        currencies[0]
                    ),
                    last_date=pd.Timestamp(
                        group[
                            "event_date"
                        ].max()
                    ),
                    description=(
                        description
                    ),
                    cadence_days=30,
                    monthly=True,
                ),
            )
        )

    if candidates:

        # Prefer the stream with the most observations.
        # Break ties using the most recent stream.
        candidates.sort(
            key=lambda item: (
                item[0],
                item[1],
            ),
            reverse=True,
        )

        return candidates[0][2]

    # =========================================================
    # FALLBACK: CONFIRMED FUTURE PAYROLL ANCHOR
    # =========================================================
    #
    # This handles users who recently started a job or otherwise
    # have too little settled payroll history to satisfy the normal
    # three-occurrence stability requirement.
    #
    # We still require prior settled salary context and a strongly
    # confirmed/scheduled future salary event.
    # =========================================================

    return _infer_confirmed_future_salary_stream(
        salary=salary_all,
        request_date=request_date,
    )


def next_monthly_date(
    last_date: pd.Timestamp,
) -> pd.Timestamp:
    """
    Preserve the salary calendar day where possible.
    """

    return (
        pd.Timestamp(
            last_date
        )
        + pd.DateOffset(
            months=1
        )
    )


def generate_salary_occurrences(
    stream: SalaryStream,
    request_date: pd.Timestamp,
    forecast_end: pd.Timestamp,
    home_currency: str,
    fx: ExchangeRateBook,
) -> list[
    tuple[
        pd.Timestamp,
        Decimal,
    ]
]:
    """
    Generate future salary occurrences.

    If a temporary / next-payroll override exists, it is applied
    to exactly one matching future salary occurrence.

    Later salary occurrences revert to the stable base salary.

    Each occurrence is converted into the user's home currency
    using the FX rate applicable to that salary date.
    """

    result = []

    request_date = pd.Timestamp(
        request_date
    )

    forecast_end = pd.Timestamp(
        forecast_end
    )

    date = next_monthly_date(
        stream.last_date
    )

    override_consumed = False

    while date <= forecast_end:

        if date >= request_date:

            salary_amount = (
                stream.amount
            )

            salary_currency = (
                stream.currency
            )

            use_override = False

            # =================================================
            # TEMPORARY ONE-CYCLE OVERRIDE
            # =================================================

            if (
                stream.next_amount_override
                is not None
                and not override_consumed
            ):

                # If no exact override date was supplied,
                # use the next generated payroll occurrence.
                if (
                    stream.next_override_date
                    is None
                ):
                    use_override = True

                else:

                    generated_date = (
                        pd.Timestamp(
                            date
                        )
                        .normalize()
                    )

                    override_date = (
                        pd.Timestamp(
                            stream
                            .next_override_date
                        )
                        .normalize()
                    )

                    # Allow a small tolerance for payroll posting
                    # dates around weekends / processing dates.
                    if (
                        abs(
                            (
                                generated_date
                                - override_date
                            ).days
                        )
                        <= 2
                    ):
                        use_override = True

            if use_override:

                salary_amount = (
                    stream
                    .next_amount_override
                )

                salary_currency = (
                    stream
                    .next_currency_override
                    or stream.currency
                )

                override_consumed = True

            # =================================================
            # FX CONVERSION ON OCCURRENCE DATE
            # =================================================

            converted = fx.convert(
                amount=salary_amount,
                rate_date=date,
                from_currency=(
                    salary_currency
                ),
                to_currency=(
                    home_currency
                ),
            )

            result.append(
                (
                    pd.Timestamp(
                        date
                    ),
                    abs(
                        converted
                    ),
                )
            )

        date = (
            pd.Timestamp(
                date
            )
            + pd.DateOffset(
                months=1
            )
        )

    return result