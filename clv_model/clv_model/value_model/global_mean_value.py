from dataclasses import dataclass
import typing

import pandas

from value_model import ValueModel

__all__ = ('GlobalMeanValue',)


@dataclass
class GlobalMeanValue(ValueModel):
    def __post_init__(self):
        self.global_mean: typing.Optional[float] = None

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

    def _is_fitted(self) -> bool:
        return self.global_mean is not None

    def predict(self, data: pandas.DataFrame) -> pandas.DataFrame:
        self._check_fit()
        return (
            data
            .assign(value=self.global_mean)
            [['customer_id', 'value']]
        )
