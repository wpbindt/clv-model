from dataclasses import dataclass
from numbers import Real
import typing

import pandas

from .value_model import ValueModel

__all__ = ('GlobalMeanValue',)


@dataclass
class GlobalMeanValue(ValueModel):
    global_mean: typing.Optional[Real] = None

    def fit(self, data: pandas.DataFrame, **kwargs) -> ValueModel:
        total_transactions = data.frequency.sum()
        self.global_mean = (
            data
            .assign(weighted_value=lambda df: df.value * df.frequency)
            .weighted_value
            .sum()
            / total_transactions
        )
        return self

    def is_fitted(self) -> bool:
        return self.global_mean is not None

    def predict(self, data: pandas.DataFrame) -> pandas.DataFrame:
        self._check_fit()
        return (
            data
            .assign(value=self.global_mean)
            [['id', 'value']]
        )
