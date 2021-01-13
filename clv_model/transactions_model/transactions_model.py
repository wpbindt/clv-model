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
        """
        Should predict the number of purchases the customer will make
        over the next `periods` periods. That is, previous purchases are
        not counted.
        The exact shape of the dataframe `data` will be dependent on the
        model. For RFM based models, like the Pareto-NBD model, it
        should have columns ('id', 'recency', 'frequency', 'T'), where
        T is the number of periods the customer has been observed for.
        This method should then predict the number of transactions
        occurring in the interval (T, T + periods].
        """
        ...

    def _check_fit(self) -> None:
        if not self.is_fitted():
            raise ValueError('Model is not fit.')
