from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas

from .transactions_model.transactions_model import TransactionsModel
from .value_model.value_model import ValueModel

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
        """
        Predict CLV for the interval [1, T + periods], where T is the
        number of periods the customer has been observed for.
        For the interval [1, T], historic values are used, and for the
        interval (T, T + periods], value_model and transactions_model
        are used to predict the CLV.
        """
        if not self.is_fitted():
            raise ValueError('Model must be fitted.')

        if not 0 <= discount_rate < 1:
            raise ValueError('Discount rate must be in [0,1).')

        historic_clv = self._compute_historic_clv(
            rfm_df=data,
            discount_rate=discount_rate
        )
        future_clv = self._compute_future_clv(
            data=data,
            periods=periods,
            discount_rate=discount_rate
        )

        return (
            pandas.concat([historic_clv, future_clv])
            .groupby('id', as_index=False, sort=False)
            .sum()
            .assign(clv=lambda df: df.clv.round(2))
        )

    def _compute_future_clv(
        self,
        data: pandas.DataFrame,
        periods: int,
        discount_rate: float
    ) -> pandas.DataFrame:
        transactions = self.transactions_model.predict(data, periods)
        values = self.value_model.predict(data)

        alpha = 1 / (1 + discount_rate)
        discounted_time = (
            periods if alpha == 1
            else ((1 - alpha**periods) / (1 - alpha))
        )

        return (
            transactions
            .merge(values, on='id')
            .merge(data[['id', 'T']], on='id')
            .assign(
                transaction_rate=lambda df: df.transactions / periods,
                base_discount_factor=lambda df: alpha ** df['T'],
                clv=lambda df: (
                    df.transaction_rate
                    * df.value
                    * df.base_discount_factor
                    * discounted_time
                )
            )
            [['id', 'clv']]
        )

    @staticmethod
    def _compute_historic_clv(
        rfm_df: pandas.DataFrame,
        discount_rate: float
    ) -> pandas.DataFrame:
        alpha = 1 / (1 + discount_rate)
        return (
            rfm_df
            .assign(
                transaction_rate=lambda df: df.frequency / df['T'],
                discounted_time=lambda df:
                ((1 - alpha**df['T']) / (1 - alpha)).fillna(df['T']),
                clv=lambda df: (
                    df.transaction_rate
                    * df.value
                    * df.discounted_time
                ).round(2)
            )
            [['id', 'clv']]
        )
