from dataclasses import dataclass
from value_model import ValueModel

import pandas

__all__ = ('LocalMeanValue',)


@dataclass
class LocalMeanValue(ValueModel):
    def fit(self, data: pandas.DataFrame, **kwargs) -> ValueModel:
        return self

    def is_fitted(self) -> bool:
        return True

    def predict(self, data: pandas.DataFrame) -> pandas.DataFrame:
        return data[['customer_id', 'value']]
