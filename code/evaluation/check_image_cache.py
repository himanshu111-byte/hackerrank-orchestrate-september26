from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(ROOT / "code"),
)


from data_loader import load_datasets
from image_evidence import load_image_cache


def main():

    data = load_datasets()
    cache = load_image_cache()

    blank = data.events[
        data.events["amount"].isna()
    ]

    print("=" * 90)
    print("IMAGE EVIDENCE STATUS")
    print("=" * 90)

    print(
        "Cached images:",
        len(cache),
    )

    print(
        "Remaining blank event amounts:",
        len(blank),
    )

    if not blank.empty:
        print()
        print(
            blank[
                [
                    "event_id",
                    "user_id",
                    "description",
                    "category",
                    "currency",
                ]
            ].to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()