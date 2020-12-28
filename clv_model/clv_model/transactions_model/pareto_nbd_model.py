from __future__ import annotations
import typing

import numpy
import pandas
from scipy.special import gamma, hyp2f1

from clv_model.clv_model.stan_model_base import StanModelBase
from clv_model.clv_model.transactions_model.transactions_model \
    import TransactionsModel

__all__ = ('ParetoNBDModel',)


class ParetoNBDModel(
    StanModelBase,
    TransactionsModel,
    model_name='pareto_nbd'
):
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

        frequency = data.frequency.values.reshape(-1, 1)
        recency = data.recency.values.reshape(-1, 1)
        observation_period = data['T'].values.reshape(-1, 1)

        likes = self._likelihoods(
            frequency, recency, observation_period
        )

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

    def _likelihoods(
        self,
        frequency: numpy,
        recency: numpy.ndarray,
        observation_period: numpy.ndarray
    ) -> numpy.ndarray:
        self._check_fit()

        denom1 = numpy.where(
            self.lambda_rate >= self.mu_rate,
            self.lambda_rate + recency,
            self.mu_rate + recency
        )

        lambda_rate_t = self.lambda_rate + observation_period
        mu_rate_t = self.mu_rate + observation_period

        denom2 = numpy.where(
            self.lambda_rate >= self.mu_rate,
            lambda_rate_t,
            mu_rate_t
        )

        middle_hypergeom_arg = numpy.where(
            self.lambda_rate >= self.mu_rate,
            self.mu_shape + 1,
            self.lambda_shape + frequency
        )

        shape_frequency = self.lambda_shape + frequency
        denom_exponent = shape_frequency + self.mu_shape

        abs_diff = numpy.abs(self.lambda_rate - self.mu_rate)

        a_0 = (
            hyp2f1(
                denom_exponent,
                middle_hypergeom_arg,
                denom_exponent + 1,
                abs_diff / denom1
            ) / (denom1 ** denom_exponent)
            - hyp2f1(
                denom_exponent,
                middle_hypergeom_arg,
                denom_exponent + 1,
                abs_diff / denom2
            ) / (denom2 ** denom_exponent)
        )

        return (
            (
                gamma(shape_frequency)
                * (self.lambda_rate ** self.lambda_shape)
                * (self.mu_rate ** self.mu_shape)
                / gamma(self.lambda_shape)
            )
            * (
                1 / (
                    (lambda_rate_t ** shape_frequency)
                    * (mu_rate_t ** self.mu_shape)
                )
                + self.mu_shape * a_0 / denom_exponent
            )
        )

    def probability_alive(
        self,
        frequency: numpy,
        recency: numpy.ndarray,
        observation_period: numpy.ndarray,
    ) -> pandas.DataFrame:
        likelihoods = self._likelihoods(frequency, recency, observation_period)
        shape_frequency = self.lambda_shape + frequency

        return (
            gamma(shape_frequency)
            * (self.lambda_rate ** self.lambda_shape)
            * (self.mu_rate ** self.mu_shape)
            / (
                gamma(self.lambda_shape)
                * (self.lambda_rate + observation_period) ** shape_frequency
                * (self.mu_rate + observation_period) ** self.mu_shape
                * likelihoods
            )
        )
