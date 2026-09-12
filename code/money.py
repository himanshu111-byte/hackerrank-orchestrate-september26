from decimal import Decimal, ROUND_HALF_UP

import pandas as pd


MONEY_QUANT = Decimal("0.01")


def money(value) -> Decimal:
    if value is None:
        return Decimal("0")

    if pd.isna(value):
        return Decimal("0")

    return Decimal(str(value))


def round_money(value: Decimal) -> Decimal:
    return value.quantize(
        MONEY_QUANT,
        rounding=ROUND_HALF_UP,
    )


class ExchangeRateBook:

    def __init__(
        self,
        exchange_rates: pd.DataFrame,
    ):
        rates = exchange_rates.copy()

        rates["rate_date"] = pd.to_datetime(
            rates["rate_date"]
        )

        self.rates = rates

    def get_rate(
        self,
        rate_date,
        from_currency: str,
        to_currency: str,
    ) -> Decimal:

        if from_currency == to_currency:
            return Decimal("1")

        rate_date = pd.Timestamp(rate_date)

        direct = self.rates[
            (
                self.rates["rate_date"]
                == rate_date
            )
            & (
                self.rates["from_currency"]
                == from_currency
            )
            & (
                self.rates["to_currency"]
                == to_currency
            )
        ]

        if not direct.empty:
            return money(
                direct.iloc[0]["rate"]
            )

        # If direct pair is missing, try inverse.
        inverse = self.rates[
            (
                self.rates["rate_date"]
                == rate_date
            )
            & (
                self.rates["from_currency"]
                == to_currency
            )
            & (
                self.rates["to_currency"]
                == from_currency
            )
        ]

        if not inverse.empty:
            value = money(
                inverse.iloc[0]["rate"]
            )

            if value == 0:
                raise ValueError(
                    "Inverse FX rate cannot be zero"
                )

            return Decimal("1") / value

        raise KeyError(
            f"No FX rate for "
            f"{rate_date.date()} "
            f"{from_currency}->{to_currency}"
        )

    def convert(
        self,
        amount,
        rate_date,
        from_currency: str,
        to_currency: str,
    ) -> Decimal:

        amount = money(amount)

        if from_currency == to_currency:
            return amount

        rate = self.get_rate(
            rate_date,
            from_currency,
            to_currency,
        )

        return round_money(
            amount * rate
        )