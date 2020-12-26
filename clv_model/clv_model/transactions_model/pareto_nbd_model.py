from __future__ import annotations
from dataclasses import dataclass
from importlib import resources
import typing

import numpy
import pandas
import pystan

from clv_model.clv_model.transactions_model.transactions_model \
    import TransactionsModel

__all__ = ('ParetoNBDModel',)


@dataclass
class ParetoNBDModel(TransactionsModel):
    lambda_shape: typing.Optional[numpy.ndarray] = None
    lambda_rate: typing.Optional[numpy.ndarray] = None
    mu_shape: typing.Optional[numpy.ndarray] = None
    mu_rate: typing.Optional[numpy.ndarray] = None

    def __post_init__(self) -> None:
        self._stan_model: typing.Optional[pystan.StanModel] = None

    @classmethod
    def from_file(cls, file_path: str) -> ParetoNBDModel:
        parameters_df = pandas.read_csv(file_path)

        return cls(
            lambda_shape=parameters_df['lambda_shape'],
            lambda_rate=parameters_df['lambda_rate'],
            mu_shape=parameters_df['mu_shape'],
            mu_rate=parameters_df['mu_rate'],
        )

    def to_file(self, file_path: str) -> None:
        self._check_fit()

        pandas.DataFrame(
            data={
                'lambda_shape': self.lambda_shape,
                'lambda_rate': self.lambda_rate,
                'mu_shape': self.mu_shape,
                'mu_rate': self.mu_rate,
            }
        ).to_csv(file_path, index=False)

    def fit(self, data: pandas.DataFrame, **kwargs) -> TransactionsModel:
        if self._is_fitted():
            return self

        if self._stan_model is None:
            self._compile_stan_model()

        data_dict = {
            **dict(data[['recency', 'frequency', 'T']]),
            'N': len(data)
        }
        fit = self._stan_model.sampling(data=data_dict, **kwargs)

        posteriors = fit.extract(permuted=True)
        self.lambda_shape = posteriors['lambda_shape']
        self.lambda_rate = posteriors['lambda_rate']
        self.mu_shape = posteriors['mu_shape']
        self.mu_rate = posteriors['mu_rate']

        return self

    def _is_fitted(self) -> bool:
        return (
            self.lambda_shape is not None
            and self.lambda_rate is not None
            and self.mu_shape is not None
            and self.mu_rate is not None
        )

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

    def _compile_stan_model(self) -> pystan.StanModel:
        with resources.open_text(
            'clv_model.stan_models',
            'pareto_nbd.stan'
        ) as model_file:
            self._stan_model = pystan.StanModel(model_file)

        return self._stan_model
