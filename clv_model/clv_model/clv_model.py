from __future__ import annotations
from dataclasses import dataclass

import pandas

from transactions_model.transactions_model import TransactionsModel
from value_model.value_model import ValueModel

__all__ = ('CLVModel',)


@dataclass
class CLVModel:
    value_model: ValueModel
    transactions_model: TransactionsModel

    def fit(self, data: pandas.DataFrame) -> CLVModel:
        self.value_model.fit(data)
        self.transactions_model.fit(data)

        return self

    def _is_fitted(self) -> bool:
        if self._is_fitted():
            return self

        return (
            self.value_model._is_fitted()
            and self.transactions_model._is_fitted()
        )

    def predict(
        self,
        data: pandas.DataFrame,
        periods: int,
        discount_rate: float
    ) -> pandas.DataFrame:
        if not self._is_fitted():
            raise ValueError('Model must be fitted.')

        transactions = self.transactions_model.predict(data, periods)
        values = self.value_model.predict(data)

        non_discounted_clv = (
            transactions
            .merge(values, on='id')
            .assign(clv=lambda df: df.transactions * df.value)
        )

        if discount_rate == 0:
            return non_discounted_clv[['id', 'clv']]

        return self._compute_discounted_clv(non_discounted_clv)

    def _compute_discounted_clv(
        self,
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
