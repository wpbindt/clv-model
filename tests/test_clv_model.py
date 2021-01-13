import unittest

import pandas
from pandas.testing import assert_frame_equal

from clv_model.clv_model import CLVModel
from clv_model.transactions_model.global_transaction_rate_model \
    import GlobalTransactionRateModel
from clv_model.value_model.global_mean_value import GlobalMeanValue


class TestCLVModel(unittest.TestCase):
    def _get_df(self) -> pandas.DataFrame:
        return pandas.DataFrame(
            data={
                'id': [0, 1, 2],
                'recency': [1, 1, 1],
                'frequency': [1, 2, 2],
                'T': [2, 2, 1],
                'value': [1, 1, 2]
            }
        )

    def _get_model(self) -> CLVModel:
        value_model = GlobalMeanValue(global_mean=1)
        transactions_model = GlobalTransactionRateModel(
            mean_transaction_rate=1
        )
        return CLVModel(
            value_model=value_model,
            transactions_model=transactions_model
        )

    def test_predict(self) -> None:
        data = self._get_df()
        model = self._get_model()
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

    def test_predict_no_discount(self) -> None:
        data = self._get_df()
        model = self._get_model()
        actual = model.predict(data=data, periods=1, discount_rate=0)
        expected = pandas.DataFrame(
            data={
                'id': [0, 1, 2],
                'clv': [
                    0.5 + 0.5 + 1,
                    1 + 1 + 1,
                    4 + 1
                ]
            }
        ).assign(clv=lambda df: df.clv.round(2))
        assert_frame_equal(actual, expected, check_dtype=False)

    def test_predict_no_future(self) -> None:
        data = self._get_df()
        model = self._get_model()
        actual = model.predict(data=data, periods=0, discount_rate=0.15)
        expected = pandas.DataFrame(
            data={
                'id': [0, 1, 2],
                'clv': [
                    0.5 + 0.5/1.15,
                    1 + 1/1.15,
                    4,
                ]
            }
        ).assign(clv=lambda df: df.clv.round(2))
        assert_frame_equal(actual, expected)

    def test_predict_no_future_no_discount(self) -> None:
        data = self._get_df()
        model = self._get_model()
        actual = model.predict(data=data, periods=0, discount_rate=0)
        expected = pandas.DataFrame(
            data={
                'id': [0, 1, 2],
                'clv': [
                    0.5 + 0.5,
                    1 + 1,
                    4,
                ]
            }
        ).assign(clv=lambda df: df.clv.round(2))
        assert_frame_equal(actual, expected)

    def test_predict_empty(self) -> None:
        model = self._get_model()
        actual = model.predict(
            data=pandas.DataFrame(
                columns={
                    'id',
                    'recency',
                    'frequency',
                    'T',
                    'value'
                }
            ),
            periods=1,
            discount_rate=0.15
        )
        self.assertTrue(actual.empty)
