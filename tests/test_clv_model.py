import unittest

import pandas
from pandas.testing import assert_frame_equal

from clv_model.clv_model import CLVModel
from clv_model.transactions_model.global_transaction_rate_model \
    import GlobalTransactionRateModel
from clv_model.value_model.global_mean_value import GlobalMeanValue


class TestCLVModel(unittest.TestCase):
    def test_predict(self) -> None:
        value_model = GlobalMeanValue(global_mean=1)
        transactions_model = GlobalTransactionRateModel(
            mean_transaction_rate=1
        )
        model = CLVModel(
            value_model=value_model,
            transactions_model=transactions_model
        )
        data = pandas.DataFrame(
            data={
                'id': [0, 1, 2],
                'recency': [1, 1, 1],
                'frequency': [1, 2, 2],
                'T': [2, 2, 1],
                'value': [1, 1, 2]
            }
        )
        actual = model.predict(data=data, periods=1, discount_rate=0.15)
        expected = pandas.DataFrame(
            data={
                'id': [0, 1, 2],
                'clv': [
                    0.5 + 0.5/1.15 + 1/1.15**2,
                    1 + 1/1.15 + 1/1.15**2,
                    4 + 1/1.15
                ]
            }
        ).assign(clv=lambda df: df.clv.round(2))
        assert_frame_equal(actual, expected)

        actual = model.predict(data=pandas.DataFrame(
            columns={
                'id',
                'recency',
                'frequency',
                'T',
                'value'
            }
        ))
        self.assertTrue(actual.empty)
