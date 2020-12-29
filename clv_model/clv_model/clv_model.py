from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas

from transactions_model.transactions_model import TransactionsModel
from value_model.value_model import ValueModel

__all__ = ('CLVModel',)


@dataclass
class CLVModel:
    value_model: ValueModel
    transactions_model: TransactionsModel

    def fit(
        self,
        data: pandas.DataFrame,
        value_model_kwargs: Optional[Dict[str, Any]] = None,
        transactions_model_kwargs: Optional[Dict[str, Any]] = None
    ) -> CLVModel:
        if not self.value_model.is_fitted():
            if value_model_kwargs is None:
                value_model_kwargs = {}

            self.value_model.fit(data=data, **value_model_kwargs)

        if not self.transactions_model.is_fitted():
            if transactions_model_kwargs is None:
                transactions_model_kwargs = {}

            self.transactions_model.fit(data=data, **transactions_model_kwargs)

        return self

    def is_fitted(self) -> bool:
        return (
            self.value_model.is_fitted()
            and self.transactions_model.is_fitted()
        )

    def predict(
        self,
        data: pandas.DataFrame,
        periods: int,
        discount_rate: float
    ) -> pandas.DataFrame:
        if not self.is_fitted():
            raise ValueError('Model must be fitted.')

        if not 0 <= discount_rate < 1:
            raise ValueError('Discount rate must be in [0,1).')

        transactions = self.transactions_model.predict(data, periods)
        values = self.value_model.predict(data)

        non_discounted_clv = (
            transactions
            .merge(values, on='id')
            .assign(clv=lambda df: df.transactions * df.value)
        )

        if discount_rate == 0:
            return non_discounted_clv[['id', 'clv']]

        return CLVModel._compute_discounted_clv(
            non_discounted_clv=non_discounted_clv,
            periods=periods,
            discount_rate=discount_rate
        )

    @staticmethod
    def _compute_discounted_clv(
        non_discounted_clv: pandas.DataFrame,
        periods: int,
        discount_rate: float
    ) -> pandas.DataFrame:
        return (
            non_discounted_clv
            .assign(
                discount_factor=lambda df:
                    (1 - (1 - discount_rate) ** df.transactions)
                    / (periods * discount_rate),
                clv=lambda df: df.clv * df.discount_factor
            )
            [['id', 'clv']]
        )
