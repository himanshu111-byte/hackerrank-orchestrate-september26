from __future__ import annotations

import json
from pathlib import Path
from decimal import Decimal

import pandas as pd

from money import money


ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = (
    ROOT
    / "dataset"
    / "media"
    / "images"
)

CACHE_PATH = (
    ROOT
    / "code"
    / "image_evidence_cache.json"
)


def load_image_cache() -> dict:

    if not CACHE_PATH.exists():
        return {}

    with open(
        CACHE_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def save_image_cache(
    cache: dict,
):
    with open(
        CACHE_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            cache,
            f,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )


def patch_image_amounts(
    events: pd.DataFrame,
    images: pd.DataFrame,
) -> pd.DataFrame:

    result = events.copy()

    cache = load_image_cache()

    for _, image_row in (
        images.iterrows()
    ):

        event_id = (
            image_row[
                "related_event_id"
            ]
        )

        image_id = (
            image_row[
                "image_id"
            ]
        )

        if pd.isna(event_id):
            continue

        matches = (
            result[
                result["event_id"]
                == event_id
            ]
        )

        if matches.empty:
            continue

        idx = matches.index[0]

        if not pd.isna(
            result.at[idx, "amount"]
        ):
            continue

        if image_id not in cache:
            continue

        value = cache[image_id]

        if (
            not isinstance(
                value,
                dict,
            )
            or
            "amount" not in value
        ):
            continue

        result.at[
            idx,
            "amount",
        ] = float(
            money(
                value["amount"]
            )
        )

        if value.get("currency"):
            result.at[
                idx,
                "currency",
            ] = value[
                "currency"
            ]

    return result