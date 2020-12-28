from typing import Optional

import numpy
import pandas

from clv_model.clv_model.stan_model_base import StanModelBase
from clv_model.clv_model.value_model.value_model import ValueModel

__all__ = ('GammaGammaModel',)


class GammaGammaModel(StanModelBase, ValueModel, model_name='gamma_gamma'):
    p: Optional[numpy.ndarray] = None
    q: Optional[numpy.ndarray] = None
    mu: Optional[numpy.ndarray] = None

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
