from __future__ import annotations
from dataclasses import dataclass
from importlib import resources
import typing

import numpy
import pandas
import pystan

from clv_based_bidder.clv_model.value_model.value_model import ValueModel

__all__ = ('GammaGammaModel',)


@dataclass
class GammaGammaModel(ValueModel):
    p: typing.Optional[numpy.ndarray] = None
    q: typing.Optional[numpy.ndarray] = None
    mu: typing.Optional[numpy.ndarray] = None

    def __post_init__(self) -> None:
        self._stan_model: typing.Optional[pystan.StanModel] = None

    @classmethod
    def from_file(cls, file_path: str) -> GammaGammaModel:
        parameters_df = pandas.read_csv(file_path)

        return cls(
            p=parameters_df['p'],
            q=parameters_df['q'],
            mu=parameters_df['mu']
        )

    def to_file(self, file_path: str) -> None:
        self._check_fit()

        pandas.DataFrame(
            data={
                'p': self.p,
                'q': self.q,
                'mu': self.mu
            }
        ).to_csv(file_path, index=False)

    def fit(self, data: pandas.DataFrame, **kwargs) -> ValueModel:
        if self._is_fitted():
            return self

        if self._stan_model is None:
            self._compile_stan_model()

        data_dict = {
            **dict(data[['value', 'frequency']]),
            'N': len(data)
        }
        fit = self._stan_model.sampling(data=data_dict, **kwargs)

        posteriors = fit.extract(permuted=True)
        self.p = posteriors['p']
        self.q = posteriors['q']
        self.mu = posteriors['mu']

        return self

    def _is_fitted(self) -> bool:
        return (
            self.p is not None
            and self.q is not None
            and self.mu is not None
        )

    def predict(self, data: pandas.DataFrame) -> pandas.DataFrame:
        self._check_fit()

        freq = data.frequency.values.reshape(-1, 1)
        val = data.value.values.reshape(-1, 1)

        # posterior mean of E_{p, q, mu}(value | freq, mean_value)
        expected_value = (
            self.p * (self.mu + freq * val) / (self.p * freq + self.q - 1)
        ).mean(1)

        return pandas.DataFrame(
            data={
                'id': data.customer_id,
                'value': expected_value
            }
        )

    def _compile_stan_model(self) -> pystan.StanModel:
        with resources.open_text(
            'clv_based_bidder.stan_models',
            'gamma_gamma.stan'
        ) as model_file:
            self._stan_model = pystan.StanModel(model_file)

        return self._stan_model
