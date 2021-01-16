from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas

__all__ = ('ValueModel',)


@dataclass
class ValueModel(ABC):
    @abstractmethod
    def fit(self, data: pandas.DataFrame, **kwargs) -> ValueModel:
        ...

    @abstractmethod
    def is_fitted(self) -> bool:
        ...

    @abstractmethod
    def predict(self, data: pandas.DataFrame) -> pandas.DataFrame:
        ...

    def _check_fit(self) -> None:
        if not self.is_fitted():
            raise ValueError(
                'Model must be fitted by calling fit before calling predict.'
            )
