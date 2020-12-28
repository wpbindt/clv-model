import pandas

from clv_model.clv_model.stan_model_base import Parameter, StanModelBase
from clv_model.clv_model.transactions_model.transactions_model \
    import TransactionsModel

__all__ = ('BetaGeometricNBDModel',)


class BetaGeometricNBDModel(
    StanModelBase,
    TransactionsModel,
    model_name='beta_geometric_nbd'
):
    lambda_shape: Parameter
    lambda_rate: Parameter
    alpha: Parameter
    beta: Parameter

    def predict(
        self,
        data: pandas.DataFrame,
        periods: int
    ) -> pandas.DataFrame:
        raise NotImplementedError('Predict method is not yet implemented.')
