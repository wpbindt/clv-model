import unittest

import pandas
from pandas.testing import assert_frame_equal

from clv_model.transactions_model import GlobalTransactionRate


class TestGlobalTransactionRate(unittest.TestCase):
    def test_fit(self) -> None:
        data = pandas.DataFrame(
            data={
                'id': [0, 1],
                'frequency': [5, 2],
                'T': [10, 2],
            }
        )
        model = GlobalTransactionRate()
        model.fit(data)

        expected_mean_transaction_rate = (5 / 10 + 2 / 2) / 2

        self.assertEqual(
            model.mean_transaction_rate,
            expected_mean_transaction_rate
        )

    def test_predict(self) -> None:
        data = pandas.DataFrame(
            data={
                'id': [0, 1],
                'frequency': [2, 2],
                'T': [5, 10]
            }
        )
        model = GlobalTransactionRate(0.5)
        actual = model.predict(
            data=data,
            periods=10
        )
        expected = pandas.DataFrame(
            data={
                'id': [0, 1],
                'transactions': [5, 5]
            }
        )

        assert_frame_equal(actual, expected, check_dtype=False)

    def test_is_fitted(self) -> None:
        model = GlobalTransactionRate()
        self.assertFalse(model.is_fitted())

        model.mean_transaction_rate = 9
        self.assertTrue(model.is_fitted())
