from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import pandas

from clv_model.clv_model.transactions_model.transactions_model \
    import TransactionsModel

__all__ = ('GlobalMeanTransactionRateModel',)


@dataclass
class GlobalMeanTransactionRateModel(TransactionsModel):
    mean_transaction_rate: Optional[float] = None

    def fit(self, data: pandas.DataFrame, **kwargs) -> TransactionsModel:
        if not self._is_fitted():
            self.mean_transaction_rate = (data.frequency / data['T']).mean()

        return self

    def _is_fitted(self) -> bool:
        return self.mean_transaction_rate is not None

    def predict(
        self,
        data: pandas.DataFrame,
        periods: int
    ) -> pandas.DataFrame:
        self._check_fit()

        return (
            data
            .assign(transactions=periods * self.mean_transaction_rate)
            [['id', 'transactions']]
        )
