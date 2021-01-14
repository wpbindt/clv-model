from dataclasses import dataclass

import pandas

from .transactions_model import TransactionsModel

__all__ = ('LocalTransactionRateModel',)


@dataclass
class LocalTransactionRateModel(TransactionsModel):
    def fit(self, data: pandas.DataFrame, **kwargs) -> TransactionsModel:
        return self

    def is_fitted(self) -> bool:
        return True

    def predict(
        self,
        data: pandas.DataFrame,
        periods: int
    ) -> pandas.DataFrame:
        return (
            data
            .assign(
                transactions=lambda df: (df.frequency / df['T']) * periods
            )
            [['id', 'transactions']]
        )
