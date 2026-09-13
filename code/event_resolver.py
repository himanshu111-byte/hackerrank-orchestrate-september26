from __future__ import annotations

import pandas as pd


TERMINAL_NEGATIVE_STATUSES = {
    "cancelled",
    "failed",
}

STATUS_PRIORITY = {
    "cancelled": 100,
    "failed": 95,
    "settled": 90,
    "completed": 90,
    "confirmed": 80,
    "scheduled": 70,
    "pending": 60,
    "forecast": 40,
    "estimated": 30,
    "unrealized": 10,
}


def _status_priority(status) -> int:
    if pd.isna(status):
        return 0

    return STATUS_PRIORITY.get(
        str(status).lower(),
        0,
    )


def resolve_linked_events(
    events: pd.DataFrame,
) -> pd.DataFrame:
    """
    Preserve genuine linked lifecycle records but prevent the engine from
    interpreting linked replacements as unrelated duplicate cash events.

    Important:
      - refunds remain real separate cash flows
      - non-cash investment valuations remain non-cash
      - cancelled/failed lifecycle rows suppress their originating event
        only when they clearly represent the terminal state of that event
      - settled replacements take precedence over pending/estimated rows
    """

    df = events.copy()

    if df.empty:
        return df

    df["event_date"] = pd.to_datetime(
        df["event_date"],
        errors="coerce",
    )

    if "settlement_date" in df.columns:
        df["settlement_date"] = pd.to_datetime(
            df["settlement_date"],
            errors="coerce",
        )

    df["_drop_resolver"] = False

    # Map IDs to dataframe indices.
    id_to_index = {
        row["event_id"]: idx
        for idx, row in df.iterrows()
    }

    for idx, child in df[
        df["linked_event_id"].notna()
    ].iterrows():

        parent_id = child["linked_event_id"]

        if parent_id not in id_to_index:
            continue

        parent_idx = id_to_index[parent_id]

        parent = df.loc[parent_idx]

        child_status = str(
            child["status"]
        ).lower()

        parent_status = str(
            parent["status"]
        ).lower()

        child_type = str(
            child["event_type"]
        ).lower()

        parent_type = str(
            parent["event_type"]
        ).lower()

        # Refunds/reimbursements are actual new cash credits.
        # Do NOT erase the original transaction.
        if child_type in {
            "refund",
            "reimbursement",
        }:
            continue

        # Non-cash investment valuations do not replace the
        # investment transaction for cash purposes.
        if str(child["direction"]).lower() == "non_cash":
            continue

        # Explicit cancellation/failure of the lifecycle means
        # the original pending/forecast item should not be charged.
        if child_status in TERMINAL_NEGATIVE_STATUSES:
            df.at[parent_idx, "_drop_resolver"] = True
            df.at[idx, "_drop_resolver"] = True
            continue

        # If two records represent the same cash direction and category,
        # assume the stronger/newer lifecycle state replaces the weaker one.
        same_direction = (
            str(child["direction"]).lower()
            ==
            str(parent["direction"]).lower()
        )

        same_category = (
            str(child["category"]).lower()
            ==
            str(parent["category"]).lower()
        )

        if same_direction and same_category:

            child_priority = _status_priority(
                child_status
            )

            parent_priority = _status_priority(
                parent_status
            )

            if child_priority >= parent_priority:
                df.at[
                    parent_idx,
                    "_drop_resolver",
                ] = True
            else:
                df.at[
                    idx,
                    "_drop_resolver",
                ] = True

    resolved = df[
        ~df["_drop_resolver"]
    ].drop(
        columns=["_drop_resolver"]
    )

    return (
        resolved
        .sort_values(
            ["event_date", "event_id"]
        )
        .reset_index(drop=True)
    )