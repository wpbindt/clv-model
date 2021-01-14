from numbers import Real
from typing import Optional
import unittest

import pandas
from pandas.testing import assert_frame_equal

from clv_model.clv_model import CLVModel
from clv_model.transactions_model import GlobalTransactionRateModel
from clv_model.value_model import GlobalMeanValue


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

    def _get_model(
        self,
        global_mean: Optional[Real] = 1,
        mean_transaction_rate: Optional[Real] = 1
    ) -> CLVModel:
        value_model = GlobalMeanValue(global_mean=global_mean)
        transactions_model = GlobalTransactionRateModel(
            mean_transaction_rate=mean_transaction_rate
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

    def test_predict_unfit(self) -> None:
        model = self._get_model(global_mean=None)
        data = self._get_df()
        with self.assertRaises(ValueError) as error:
            model.predict(
                data=data,
                periods=1,
                discount_rate=0.2
            )
        self.assertEqual(
            str(error.exception),
            'Model must be fitted with a call to fit before '
            'predict can be called.'
        )

    def test_predict_bad_discount_rate(self) -> None:
        model = self._get_model()
        data = self._get_df()
        with self.assertRaises(ValueError) as error:
            model.predict(
                data=data,
                periods=1,
                discount_rate=1
            )
        self.assertEqual(
            str(error.exception),
            'Discount rate must be in [0,1).'
        )

        with self.assertRaises(ValueError) as error:
            model.predict(
                data=data,
                periods=1,
                discount_rate=-1
            )
        self.assertEqual(
            str(error.exception),
            'Discount rate must be in [0,1).'
        )

        with self.assertRaises(ValueError) as error:
            model.predict(
                data=data,
                periods=1,
                discount_rate=1729.01
            )
        self.assertEqual(
            str(error.exception),
            'Discount rate must be in [0,1).'
        )

    def test_is_fitted(self) -> None:
        model = self._get_model(global_mean=None, mean_transaction_rate=None)
        self.assertFalse(model.is_fitted())

        model = self._get_model(mean_transaction_rate=None)
        self.assertFalse(model.is_fitted())

        model = self._get_model(global_mean=None)
        self.assertFalse(model.is_fitted())

        model = self._get_model()
        self.assertTrue(model.is_fitted())

    def test_fit(self) -> None:
        model = self._get_model(global_mean=None, mean_transaction_rate=None)
        training_data = pandas.DataFrame(
            data={
                'id': [0, 1],
                'recency': [1, 1],
                'frequency': [3, 2],
                'T': [2, 2],
                'value': [1, 1]
            }
        )
        model.fit(data=training_data)
        expected_mean_value = round((3 * 1 + 2 * 1) / (3 + 2), 2)
        self.assertEqual(
            model.value_model.global_mean,
            expected_mean_value
        )

        expected_transaction_rate = (3 / 2 + 2 / 2) / 2
        self.assertEqual(
            model.transactions_model.mean_transaction_rate,
            expected_transaction_rate
        )
