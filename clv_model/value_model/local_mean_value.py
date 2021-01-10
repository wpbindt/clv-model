from dataclasses import dataclass

import pandas

from .value_model import ValueModel

__all__ = ('LocalMeanValue',)


@dataclass
class LocalMeanValue(ValueModel):
    def fit(self, data: pandas.DataFrame, **kwargs) -> ValueModel:
        return self

    def is_fitted(self) -> bool:
        return True

    def predict(self, data: pandas.DataFrame) -> pandas.DataFrame:
        return data[['id', 'value']]
