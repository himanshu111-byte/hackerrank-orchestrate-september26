from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

import pandas as pd

from lifecycle import CashFlow
from money import money, round_money


# =============================================================
# REPRESENTATION
# =============================================================


@dataclass(frozen=True)
class SpendingAction:
    action_type: str
    event_id: str
    category: str
    new_amount: Decimal | None = None

    @property
    def output_text(
        self,
    ) -> str:

        if (
            self.action_type
            == "stop"
        ):
            return (
                f"stop:{self.event_id}"
            )

        if (
            self.action_type
            == "reduce_to"
            and self.new_amount
            is not None
        ):
            return (
                f"reduce_to:"
                f"{self.event_id}:"
                f"{_fmt_money_fixed_if_needed(self.new_amount)}"
            )

        raise ValueError(
            "Invalid spending action."
        )


# =============================================================
# HELPERS
# =============================================================


def _split_categories(
    value,
) -> set[str]:

    if pd.isna(
        value
    ):
        return set()

    return {
        item.strip().lower()
        for item in str(
            value
        ).split("|")
        if item.strip()
    }


def _fmt_money_fixed_if_needed(
    value,
) -> str:
    """
    Required spending-change formatting.

    Examples:
        665950  -> 665950
        23.50   -> 23.50
        19.00   -> 19
    """

    value = round_money(
        money(
            value
        )
    )

    if (
        value
        == value.to_integral()
    ):
        return str(
            value.quantize(
                Decimal("1")
            )
        )

    return format(
        value,
        ".2f",
    )


def _eligible_projection_groups(
    baseline_flows: list[
        CashFlow
    ],
) -> dict[
    str,
    list[CashFlow],
]:

    groups: dict[
        str,
        list[CashFlow],
    ] = {}

    for flow in baseline_flows:

        if (
            flow.source
            != "recurring_projection"
        ):
            continue

        if flow.event_id is None:
            continue

        # Only expenses can be modified.
        if (
            money(
                flow.amount
            )
            >= 0
        ):
            continue

        groups.setdefault(
            str(
                flow.event_id
            ),
            [],
        ).append(
            flow
        )

    return groups


def _reduction_target(
    flow: CashFlow,
) -> Decimal | None:
    """
    Return the legal reduction target supplied by the dataset.

    A reducible expense may only be reduced when the underlying
    financial event provides an explicit minimum_allowed_amount.

    Never invent an arbitrary reduction percentage or amount.
    """

    recurring_amount = abs(
        money(
            flow.amount
        )
    )

    supplied_minimum = getattr(
        flow,
        "minimum_allowed_amount",
        None,
    )

    if supplied_minimum is None:
        return None

    target = round_money(
        abs(
            money(
                supplied_minimum
            )
        )
    )

    if (
        Decimal("0")
        <= target
        < recurring_amount
    ):
        return target

    return None

def generate_spending_actions(
    profile: pd.Series,
    baseline_flows: list[
        CashFlow
    ],
) -> list[
    SpendingAction
]:
    """
    Generate individually legal optional spending changes.

    Rules:
      - recurring expenses only
      - protected categories cannot be touched
      - user must permit the category to be stopped/reduced
      - event flexibility must also permit the operation
      - reduction uses an explicit source floor when supplied
    """

    protected = (
        _split_categories(
            profile[
                "expense_categories_to_protect"
            ]
        )
    )

    reducible_categories = (
        _split_categories(
            profile[
                "expense_categories_user_is_willing_to_reduce"
            ]
        )
    )

    stoppable_categories = (
        _split_categories(
            profile[
                "expense_categories_user_is_willing_to_stop"
            ]
        )
    )

    groups = (
        _eligible_projection_groups(
            baseline_flows
        )
    )

    actions: list[
        SpendingAction
    ] = []

    for (
        event_id,
        flows,
    ) in groups.items():

        first = flows[0]

        category = str(
            first.category
            or ""
        ).strip().lower()

        flexibility = str(
            first.flexibility
            or ""
        ).strip().lower()

        if not category:
            continue

        if (
            category
            in protected
        ):
            continue

        recurring_amount = abs(
            money(
                first.amount
            )
        )

        if (
            recurring_amount
            <= 0
        ):
            continue

        # -----------------------------------------------------
        # STOP
        # -----------------------------------------------------

        may_stop = (
            category
            in stoppable_categories
            and
            flexibility
            in {
                "stoppable",
                "reducible_or_stoppable",
            }
        )

        if may_stop:

            actions.append(
                SpendingAction(
                    action_type=(
                        "stop"
                    ),
                    event_id=(
                        event_id
                    ),
                    category=(
                        category
                    ),
                )
            )

        # -----------------------------------------------------
        # REDUCE
        # -----------------------------------------------------

        may_reduce = (
            category
            in reducible_categories
            and
            flexibility
            in {
                "reducible",
                "reducible_or_stoppable",
            }
        )

        if may_reduce:

            reduced_amount = (
                _reduction_target(
                    first
                )
            )

            if (
                reduced_amount
                is not None
            ):

                actions.append(
                    SpendingAction(
                        action_type=(
                            "reduce_to"
                        ),
                        event_id=(
                            event_id
                        ),
                        category=(
                            category
                        ),
                        new_amount=(
                            reduced_amount
                        ),
                    )
                )

    actions.sort(
        key=lambda action: (
            action.event_id,
            0
            if (
                action.action_type
                == "stop"
            )
            else 1,
        )
    )

    return actions


# =============================================================
# APPLY ACTIONS
# =============================================================


def apply_spending_actions(
    baseline_flows: list[
        CashFlow
    ],
    actions: list[
        SpendingAction
    ],
) -> list[
    CashFlow
]:

    stop_ids = {
        action.event_id
        for action
        in actions
        if (
            action.action_type
            == "stop"
        )
    }

    reduce_targets = {
        action.event_id:
        action.new_amount
        for action
        in actions
        if (
            action.action_type
            == "reduce_to"
            and action.new_amount
            is not None
        )
    }

    result: list[
        CashFlow
    ] = []

    for flow in baseline_flows:

        event_id = (
            str(
                flow.event_id
            )
            if (
                flow.event_id
                is not None
            )
            else None
        )

        # Spending changes affect generated recurring projections,
        # never historical/explicit financial events.
        if (
            flow.source
            != "recurring_projection"
            or event_id
            is None
        ):

            result.append(
                flow
            )
            continue

        # Stop the recurring series completely.
        if (
            event_id
            in stop_ids
        ):
            continue

        # Reduce all future occurrences in this recurring series.
        if (
            event_id
            in reduce_targets
        ):

            target = abs(
                money(
                    reduce_targets[
                        event_id
                    ]
                )
            )

            result.append(
                CashFlow(
                    date=(
                        flow.date
                    ),
                    amount=(
                        -target
                        if (
                            money(
                                flow.amount
                            )
                            < 0
                        )
                        else target
                    ),
                    source=(
                        flow.source
                    ),
                    event_id=(
                        flow.event_id
                    ),
                    category=(
                        flow.category
                    ),
                    flexibility=(
                        flow.flexibility
                    ),
                    minimum_allowed_amount=(
                        flow.minimum_allowed_amount
                    ),
                )
            )

            continue

        result.append(
            flow
        )

    return result


# =============================================================
# COMBINATION VALIDATION
# =============================================================


def actions_are_compatible(
    actions: list[
        SpendingAction
    ],
) -> bool:
    """
    The same event cannot simultaneously be stopped and reduced.
    """

    seen = set()

    for action in actions:

        if (
            action.event_id
            in seen
        ):
            return False

        seen.add(
            action.event_id
        )

    return True


def format_spending_actions(
    actions: list[
        SpendingAction
    ],
) -> str:

    if not actions:
        return "none"

    return "|".join(
        action.output_text
        for action in actions
    )