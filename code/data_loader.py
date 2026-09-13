from image_evidence import patch_image_amounts
from event_resolver import resolve_linked_events
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "dataset"


@dataclass
class DatasetBundle:
    requests: pd.DataFrame
    sample_requests: pd.DataFrame
    profiles: pd.DataFrame
    events: pd.DataFrame
    exchange_rates: pd.DataFrame
    payment_options: pd.DataFrame
    messages: pd.DataFrame
    images: pd.DataFrame


def _read_csv(name: str) -> pd.DataFrame:
    path = DATASET_DIR / name

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}"
        )

    return pd.read_csv(path)


def load_datasets() -> DatasetBundle:
    bundle = DatasetBundle(
        requests=_read_csv("requests.csv"),
        sample_requests=_read_csv("sample_requests.csv"),
        profiles=_read_csv("financial_profiles.csv"),
        events=_read_csv("financial_events.csv"),
        exchange_rates=_read_csv("exchange_rates.csv"),
        payment_options=_read_csv(
            "request_payment_options.csv"
        ),
        messages=_read_csv("messages.csv"),
        images=_read_csv("images.csv"),
    )

    bundle.events = (
        patch_image_amounts(
            events=bundle.events,
            images=bundle.images,
        )
    )

    # Parse dates once.
    for frame, columns in [
        (
            bundle.requests,
            [
                "request_date",
                "desired_completion_date",
            ],
        ),
        (
            bundle.sample_requests,
            [
                "request_date",
                "desired_completion_date",
                "earliest_date_for_full_payment",
            ],
        ),
        (
            bundle.events,
            [
                "event_date",
                "settlement_date",
            ],
        ),
        (
            bundle.exchange_rates,
            [
                "rate_date",
            ],
        ),
        (
            bundle.payment_options,
            [
                "first_payment_date",
            ],
        ),
    ]:
        for column in columns:
            if column in frame.columns:
                frame[column] = pd.to_datetime(
                    frame[column],
                    errors="coerce",
                )

    return bundle


def get_profile(
    profiles: pd.DataFrame,
    user_id: str,
) -> pd.Series:
    rows = profiles[
        profiles["user_id"] == user_id
    ]

    if len(rows) != 1:
        raise ValueError(
            f"Expected exactly one profile for "
            f"{user_id}, found {len(rows)}"
        )

    return rows.iloc[0]


def get_user_events(
    events: pd.DataFrame,
    user_id: str,
) -> pd.DataFrame:

    user_events = (
        events[
            events["user_id"] == user_id
        ]
        .copy()
        .sort_values(
            ["event_date", "event_id"]
        )
        .reset_index(drop=True)
    )

    return resolve_linked_events(
        user_events
    )