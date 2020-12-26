from __future__ import annotations
import typing

import numpy
import pandas

from clv_model.clv_model.stan_model_meta import StanModelMeta
from clv_model.clv_model.transactions_model.transactions_model \
    import TransactionsModel

__all__ = ('ParetoNBDModel',)


class ParetoNBDModel(TransactionsModel, metaclass=StanModelMeta):
    __model_name__ = 'pareto_nbd'

    lambda_shape: typing.Optional[numpy.ndarray] = None
    lambda_rate: typing.Optional[numpy.ndarray] = None
    mu_shape: typing.Optional[numpy.ndarray] = None
    mu_rate: typing.Optional[numpy.ndarray] = None

    def predict(
        self,
        data: pandas.DataFrame,
        periods: int
    ) -> pandas.DataFrame:
        self._check_fit()

        freq = data.frequency.values.reshape(-1, 1)
        rec = data.recency.values.reshape(-1, 1)
        total_time = data['T'].values.reshape(-1, 1)

        # posterior mean of
        # E(transactions periods | frequency, recency, T)
        expected_value = (
            ... # TODO
        ).mean(1)

        return pandas.DataFrame(
            data={
                'id': data.customer_id,
                'transactions': expected_value
            }
        )
