from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# PATH SETUP
# =============================================================================

ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "code"

sys.path.insert(
    0,
    str(CODE),
)


# =============================================================================
# PROJECT IMPORTS
# =============================================================================

import forecast  # noqa: E402

from capacity import (  # noqa: E402
    amount_safe_to_pay,
    earliest_full_payment_date,
)
from data_loader import (  # noqa: E402
    get_profile,
    get_user_events,
    load_datasets,
)
from forecast import (  # noqa: E402
    build_baseline_cashflows,
    run_forecast,
)
from money import ExchangeRateBook  # noqa: E402


# =============================================================================
# TEST STRATEGIES
#
# IMPORTANT:
# We do NOT edit code/recurrence.py.
#
# Instead:
#   1. Read its source.
#   2. Create temporary variants.
#   3. Replace only the recurring amount calculation.
#   4. Temporarily inject that implementation into forecast.py.
#   5. Evaluate all 25 labeled sample requests.
# =============================================================================

STRATEGIES = {
    "CURRENT_MEDIAN_5": {
        "history_window": 5,
        "expression": "np.median(recent_amounts)",
    },

    "LATEST": {
        "history_window": 5,
        "expression": "recent_amounts[-1]",
    },

    "MEDIAN_3": {
        "history_window": 3,
        "expression": "np.median(recent_amounts)",
    },

    "MEAN_3": {
        "history_window": 3,
        "expression": "np.mean(recent_amounts)",
    },

    "MEAN_5": {
        "history_window": 5,
        "expression": "np.mean(recent_amounts)",
    },

    "MEDIAN_8": {
        "history_window": 8,
        "expression": "np.median(recent_amounts)",
    },

    "MEAN_8": {
        "history_window": 8,
        "expression": "np.mean(recent_amounts)",
    },
}


# =============================================================================
# HELPERS
# =============================================================================

def fmt_date(value) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    return pd.Timestamp(
        value
    ).strftime(
        "%Y-%m-%d"
    )


def load_recurrence_variant(
    strategy_name: str,
    history_window: int,
    amount_expression: str,
):
    """
    Create an isolated temporary version of recurrence.py.

    Only two things are changed:

      group.tail(5)
          ->
      group.tail(history_window)

    and:

      np.median(recent_amounts)
          ->
      selected amount strategy

    Production code remains untouched.
    """

    recurrence_path = (
        CODE
        / "recurrence.py"
    )

    source = recurrence_path.read_text(
        encoding="utf-8"
    )

    # -------------------------------------------------------------------------
    # We intentionally match the exact production block rather than doing a
    # broad global replacement.
    # -------------------------------------------------------------------------

    original_window_block = """group
            .tail(5)
            .iterrows()"""

    replacement_window_block = f"""group
            .tail({history_window})
            .iterrows()"""

    if (
        original_window_block
        not in source
    ):
        raise RuntimeError(
            "Could not find the expected "
            "group.tail(5) amount-history block "
            "inside code/recurrence.py.\n"
            "The production file may have changed."
        )

    source = source.replace(
        original_window_block,
        replacement_window_block,
        1,
    )

    original_amount_block = """forecast_amount = money(
            np.median(
                recent_amounts
            )
        )"""

    replacement_amount_block = f"""forecast_amount = money(
            {amount_expression}
        )"""

    if (
        original_amount_block
        not in source
    ):
        raise RuntimeError(
            "Could not find the expected "
            "np.median(recent_amounts) block "
            "inside code/recurrence.py.\n"
            "Please check recurrence.py before "
            "changing production logic."
        )

    source = source.replace(
        original_amount_block,
        replacement_amount_block,
        1,
    )

    # -------------------------------------------------------------------------
    # Write only to a temporary directory.
    # -------------------------------------------------------------------------

    temp_dir = tempfile.TemporaryDirectory()

    temp_path = (
        Path(temp_dir.name)
        / f"recurrence_{strategy_name.lower()}.py"
    )

    temp_path.write_text(
        source,
        encoding="utf-8",
    )

    module_name = (
        f"_recurrence_test_"
        f"{strategy_name.lower()}"
    )

    spec = (
        importlib.util
        .spec_from_file_location(
            module_name,
            temp_path,
        )
    )

    if (
        spec is None
        or spec.loader is None
    ):
        temp_dir.cleanup()

        raise RuntimeError(
            f"Unable to load recurrence "
            f"variant {strategy_name}"
        )

    module = (
        importlib.util
        .module_from_spec(
            spec
        )
    )

    sys.modules[
        module_name
    ] = module

    spec.loader.exec_module(
        module
    )

    # Keep the TemporaryDirectory alive for the lifetime of the module.
    module._temp_dir_handle = (
        temp_dir
    )

    return module


def evaluate_strategy(
    strategy_name: str,
    recurrence_module,
    data,
    fx,
):
    """
    Evaluate one recurrence strategy across all labeled sample requests.
    """

    # -------------------------------------------------------------------------
    # Save production functions before monkeypatching.
    # -------------------------------------------------------------------------

    original_infer = (
        forecast
        .infer_recurring_series
    )

    # -------------------------------------------------------------------------
    # forecast.py imported these symbols from recurrence.py.
    # Replace them only for this test.
    # -------------------------------------------------------------------------

    forecast.infer_recurring_series = (
        recurrence_module
        .infer_recurring_series
    )

    rows = []

    try:

        for _, request in (
            data
            .sample_requests
            .iterrows()
        ):

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

            profile = get_profile(
                data.profiles,
                user_id,
            )

            events = get_user_events(
                data.events,
                user_id,
            )

            flows = (
                build_baseline_cashflows(
                    events=events,
                    request_date=(
                        request_date
                    ),
                    home_currency=(
                        profile[
                            "home_currency"
                        ]
                    ),
                    fx=fx,
                    messages=(
                        data.messages
                    ),
                    user_id=user_id,
                    request_id=(
                        request_id
                    ),
                )
            )

            baseline = run_forecast(
                opening_balance=(
                    profile[
                        "current_available_balance"
                    ]
                ),
                minimum_balance=(
                    profile[
                        "minimum_balance_to_keep"
                    ]
                ),
                request_date=(
                    request_date
                ),
                flows=flows,
            )

            predicted_safe = (
                amount_safe_to_pay(
                    baseline=baseline,
                    requested_amount=(
                        request[
                            "requested_amount"
                        ]
                    ),
                )
            )

            predicted_earliest = (
                earliest_full_payment_date(
                    opening_balance=(
                        profile[
                            "current_available_balance"
                        ]
                    ),
                    minimum_balance=(
                        profile[
                            "minimum_balance_to_keep"
                        ]
                    ),
                    request_date=(
                        request_date
                    ),
                    requested_amount=(
                        request[
                            "requested_amount"
                        ]
                    ),
                    baseline_flows=flows,
                )
            )

            expected_safe = float(
                request[
                    "amount_safe_to_pay"
                ]
            )

            predicted_safe_float = (
                float(
                    predicted_safe
                )
            )

            difference = (
                predicted_safe_float
                - expected_safe
            )

            expected_earliest = (
                fmt_date(
                    request[
                        "earliest_date_for_full_payment"
                    ]
                )
            )

            predicted_earliest_text = (
                fmt_date(
                    predicted_earliest
                )
            )

            rows.append(
                {
                    "strategy":
                        strategy_name,

                    "request_id":
                        request_id,

                    "expected_safe":
                        expected_safe,

                    "predicted_safe":
                        predicted_safe_float,

                    "safe_diff":
                        difference,

                    "abs_error":
                        abs(
                            difference
                        ),

                    "amount_exact":
                        abs(
                            difference
                        )
                        <= 0.01,

                    "expected_earliest":
                        expected_earliest,

                    "predicted_earliest":
                        predicted_earliest_text,

                    "earliest_exact":
                        (
                            expected_earliest
                            ==
                            predicted_earliest_text
                        ),

                    "baseline_min":
                        float(
                            baseline
                            .minimum_forecast_balance
                        ),

                    "minimum_required":
                        float(
                            baseline
                            .minimum_balance_required
                        ),

                    "flow_count":
                        len(
                            flows
                        ),
                }
            )

    finally:

        # ---------------------------------------------------------------------
        # Always restore production behavior, even if a strategy crashes.
        # ---------------------------------------------------------------------

        forecast.infer_recurring_series = (
            original_infer
        )

    return pd.DataFrame(
        rows
    )


def build_summary(
    result_df: pd.DataFrame,
):
    summaries = []

    for (
        strategy,
        group,
    ) in result_df.groupby(
        "strategy",
        sort=False,
    ):

        summaries.append(
            {
                "strategy":
                    strategy,

                "amount_exact":
                    int(
                        group[
                            "amount_exact"
                        ].sum()
                    ),

                "amount_total":
                    len(
                        group
                    ),

                "mae":
                    group[
                        "abs_error"
                    ].mean(),

                "median_abs_error":
                    group[
                        "abs_error"
                    ].median(),

                "max_abs_error":
                    group[
                        "abs_error"
                    ].max(),

                "earliest_exact":
                    int(
                        group[
                            "earliest_exact"
                        ].sum()
                    ),

                "earliest_total":
                    len(
                        group
                    ),
            }
        )

    summary = pd.DataFrame(
        summaries
    )

    # Lower MAE is better.
    # Exact amount and exact earliest are used as secondary measures.
    summary = summary.sort_values(
        by=[
            "mae",
            "amount_exact",
            "earliest_exact",
        ],
        ascending=[
            True,
            False,
            False,
        ],
    ).reset_index(
        drop=True
    )

    return summary


def compare_to_current(
    all_results: pd.DataFrame,
):
    """
    Show which requests improve/regress compared with CURRENT_MEDIAN_5.
    """

    current = (
        all_results[
            all_results[
                "strategy"
            ]
            ==
            "CURRENT_MEDIAN_5"
        ][
            [
                "request_id",
                "abs_error",
                "amount_exact",
                "earliest_exact",
            ]
        ]
        .rename(
            columns={
                "abs_error":
                    "current_abs_error",

                "amount_exact":
                    "current_amount_exact",

                "earliest_exact":
                    "current_earliest_exact",
            }
        )
    )

    comparisons = []

    for strategy in (
        all_results[
            "strategy"
        ].unique()
    ):

        if (
            strategy
            ==
            "CURRENT_MEDIAN_5"
        ):
            continue

        variant = (
            all_results[
                all_results[
                    "strategy"
                ]
                ==
                strategy
            ][
                [
                    "request_id",
                    "abs_error",
                    "amount_exact",
                    "earliest_exact",
                ]
            ]
            .rename(
                columns={
                    "abs_error":
                        "variant_abs_error",

                    "amount_exact":
                        "variant_amount_exact",

                    "earliest_exact":
                        "variant_earliest_exact",
                }
            )
        )

        merged = current.merge(
            variant,
            on="request_id",
            how="inner",
        )

        merged[
            "error_improvement"
        ] = (
            merged[
                "current_abs_error"
            ]
            -
            merged[
                "variant_abs_error"
            ]
        )

        merged[
            "classification"
        ] = np.where(
            merged[
                "error_improvement"
            ]
            > 0.01,
            "IMPROVED",
            np.where(
                merged[
                    "error_improvement"
                ]
                < -0.01,
                "REGRESSED",
                "UNCHANGED",
            ),
        )

        merged[
            "strategy"
        ] = strategy

        comparisons.append(
            merged
        )

    if not comparisons:
        return pd.DataFrame()

    return pd.concat(
        comparisons,
        ignore_index=True,
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print(
        "=" * 120
    )

    print(
        "RECURRING AMOUNT STRATEGY A/B TEST"
    )

    print(
        "=" * 120
    )

    print(
        "\nProduction recurrence.py will NOT be modified."
    )

    print(
        "Testing all strategies against "
        "the 25 labeled public samples.\n"
    )

    data = (
        load_datasets()
    )

    fx = (
        ExchangeRateBook(
            data.exchange_rates
        )
    )

    all_results = []

    # -------------------------------------------------------------------------
    # Evaluate each strategy independently.
    # -------------------------------------------------------------------------

    for (
        strategy_name,
        config,
    ) in STRATEGIES.items():

        print(
            f"Testing {strategy_name} "
            f"(window={config['history_window']})..."
        )

        recurrence_variant = (
            load_recurrence_variant(
                strategy_name=(
                    strategy_name
                ),
                history_window=(
                    config[
                        "history_window"
                    ]
                ),
                amount_expression=(
                    config[
                        "expression"
                    ]
                ),
            )
        )

        result = (
            evaluate_strategy(
                strategy_name=(
                    strategy_name
                ),
                recurrence_module=(
                    recurrence_variant
                ),
                data=data,
                fx=fx,
            )
        )

        all_results.append(
            result
        )

    all_results_df = (
        pd.concat(
            all_results,
            ignore_index=True,
        )
    )

    # -------------------------------------------------------------------------
    # Overall strategy ranking.
    # -------------------------------------------------------------------------

    summary = (
        build_summary(
            all_results_df
        )
    )

    pd.set_option(
        "display.max_columns",
        None,
    )

    pd.set_option(
        "display.width",
        220,
    )

    pd.set_option(
        "display.max_rows",
        500,
    )

    print(
        "\n"
        + "=" * 120
    )

    print(
        "OVERALL RESULTS"
    )

    print(
        "=" * 120
    )

    display_summary = (
        summary.copy()
    )

    display_summary[
        "amount_accuracy"
    ] = (
        display_summary[
            "amount_exact"
        ].astype(str)
        +
        "/"
        +
        display_summary[
            "amount_total"
        ].astype(str)
    )

    display_summary[
        "earliest_accuracy"
    ] = (
        display_summary[
            "earliest_exact"
        ].astype(str)
        +
        "/"
        +
        display_summary[
            "earliest_total"
        ].astype(str)
    )

    print(
        display_summary[
            [
                "strategy",
                "amount_accuracy",
                "mae",
                "median_abs_error",
                "max_abs_error",
                "earliest_accuracy",
            ]
        ]
        .to_string(
            index=False
        )
    )

    # -------------------------------------------------------------------------
    # Improvement/regression counts versus current production model.
    # -------------------------------------------------------------------------

    comparisons = (
        compare_to_current(
            all_results_df
        )
    )

    print(
        "\n"
        + "=" * 120
    )

    print(
        "VERSUS CURRENT MEDIAN-OF-5"
    )

    print(
        "=" * 120
    )

    if not comparisons.empty:

        comparison_summary = (
            comparisons
            .groupby(
                [
                    "strategy",
                    "classification",
                ]
            )
            .size()
            .unstack(
                fill_value=0
            )
            .reset_index()
        )

        for column in [
            "IMPROVED",
            "REGRESSED",
            "UNCHANGED",
        ]:
            if (
                column
                not in
                comparison_summary.columns
            ):
                comparison_summary[
                    column
                ] = 0

        print(
            comparison_summary[
                [
                    "strategy",
                    "IMPROVED",
                    "REGRESSED",
                    "UNCHANGED",
                ]
            ]
            .to_string(
                index=False
            )
        )

    # -------------------------------------------------------------------------
    # Request-level details.
    # -------------------------------------------------------------------------

    print(
        "\n"
        + "=" * 120
    )

    print(
        "REQUEST-LEVEL AMOUNT ERRORS"
    )

    print(
        "=" * 120
    )

    pivot = (
        all_results_df
        .pivot(
            index="request_id",
            columns="strategy",
            values="safe_diff",
        )
        .reset_index()
    )

    strategy_order = [
        name
        for name in STRATEGIES
        if name in pivot.columns
    ]

    print(
        pivot[
            [
                "request_id",
                *strategy_order,
            ]
        ]
        .to_string(
            index=False
        )
    )

    # -------------------------------------------------------------------------
    # Specifically inspect our three remaining decision mismatches.
    # -------------------------------------------------------------------------

    focus_ids = [
        "request_06",
        "request_11",
        "request_21",
    ]

    print(
        "\n"
        + "=" * 120
    )

    print(
        "FOCUS REQUESTS — 06 / 11 / 21"
    )

    print(
        "=" * 120
    )

    focus = (
        all_results_df[
            all_results_df[
                "request_id"
            ]
            .isin(
                focus_ids
            )
        ][
            [
                "request_id",
                "strategy",
                "expected_safe",
                "predicted_safe",
                "safe_diff",
                "abs_error",
                "expected_earliest",
                "predicted_earliest",
            ]
        ]
        .sort_values(
            [
                "request_id",
                "abs_error",
            ]
        )
    )

    print(
        focus.to_string(
            index=False
        )
    )

    # -------------------------------------------------------------------------
    # Identify best strategy.
    # -------------------------------------------------------------------------

    winner = (
        summary.iloc[0]
    )

    print(
        "\n"
        + "=" * 120
    )

    print(
        "BEST STRATEGY BY MAE"
    )

    print(
        "=" * 120
    )

    print(
        "Strategy:",
        winner[
            "strategy"
        ],
    )

    print(
        "Exact amount-safe:",
        f"{int(winner['amount_exact'])}"
        f"/{int(winner['amount_total'])}",
    )

    print(
        "MAE:",
        round(
            float(
                winner[
                    "mae"
                ]
            ),
            4,
        ),
    )

    print(
        "Median absolute error:",
        round(
            float(
                winner[
                    "median_abs_error"
                ]
            ),
            4,
        ),
    )

    print(
        "Earliest-date exact:",
        f"{int(winner['earliest_exact'])}"
        f"/{int(winner['earliest_total'])}",
    )

    # -------------------------------------------------------------------------
    # Save detailed output.
    # -------------------------------------------------------------------------

    output_dir = (
        CODE
        / "evaluation"
    )

    detailed_path = (
        output_dir
        / "recurrence_strategy_results.csv"
    )

    summary_path = (
        output_dir
        / "recurrence_strategy_summary.csv"
    )

    comparison_path = (
        output_dir
        / "recurrence_strategy_comparison.csv"
    )

    all_results_df.to_csv(
        detailed_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    comparisons.to_csv(
        comparison_path,
        index=False,
    )

    print(
        "\nSaved:"
    )

    print(
        detailed_path
    )

    print(
        summary_path
    )

    print(
        comparison_path
    )


if __name__ == "__main__":
    main()