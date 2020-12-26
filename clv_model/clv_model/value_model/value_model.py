from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas


@dataclass
class ValueModel(ABC):
    @abstractmethod
    def fit(self, data: pandas.DataFrame, **kwargs) -> ValueModel:
        ...

    @abstractmethod
    def _is_fitted(self) -> bool:
        ...

    @abstractmethod
    def predict(self, data: pandas.DataFrame) -> pandas.DataFrame:
        ...

    def _check_fit(self) -> None:
        if not self._is_fitted():
            raise ValueError('Model is not fit.')
