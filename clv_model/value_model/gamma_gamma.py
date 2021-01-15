import pandas

from ..stan_model_base import Parameter, StanModelBase
from .value_model import ValueModel

__all__ = ('GammaGamma',)


class GammaGamma(StanModelBase, ValueModel, model_name='gamma_gamma'):
    p: Parameter
    q: Parameter
    mu: Parameter

    def predict(self, data: pandas.DataFrame) -> pandas.DataFrame:
        self._check_fit()

        freq = data.frequency.values.reshape(-1, 1)
        val = data.value.values.reshape(-1, 1)

        # Posterior mean of E_{p, q, mu}(value | frequency, mean_value).
        # This is equation (5) in
        # https://www.brucehardie.com/notes/025/gamma_gamma.pdf
        expected_value = (
            self.p * (self.mu + freq * val) / (self.p * freq + self.q - 1)
        ).mean(1)

        return (
            pandas.DataFrame(
                data={
                    'id': data.id,
                    'value': expected_value
                }
            )
            .round({'value': 2})
        )
