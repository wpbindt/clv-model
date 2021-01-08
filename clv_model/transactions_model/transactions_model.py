from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas

__all__ = ('TransactionsModel',)


@dataclass
class TransactionsModel(ABC):
    @abstractmethod
    def fit(self, data: pandas.DataFrame, **kwargs) -> TransactionsModel:
        ...

    @abstractmethod
    def is_fitted(self) -> bool:
        ...

    @abstractmethod
    def predict(
        self,
        data: pandas.DataFrame,
        periods: int
    ) -> pandas.DataFrame:
        ...

    def _check_fit(self) -> None:
        if not self.is_fitted():
            raise ValueError('Model is not fit.')
