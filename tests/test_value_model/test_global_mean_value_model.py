import unittest

import pandas
from pandas.testing import assert_frame_equal

from clv_model.value_model.global_mean_value import GlobalMeanValue


class TestGlobalMeanValue(unittest.TestCase):
    def test_fit(self) -> None:
        data = pandas.DataFrame(
            data={
                'id': [0, 1],
                'frequency': [2, 1],
                'value': [5, 10],
            }
        )
        model = GlobalMeanValue()
        model.fit(data)

        self.assertEqual(model.global_mean, 20 / 3)

    def test_predict(self) -> None:
        model = GlobalMeanValue(global_mean=9)
        data = pandas.DataFrame(
            data={
                'id': [0, 1],
                'frequency': [3, 1],
            }
        )
        actual = model.predict(data)
        expected = pandas.DataFrame(
            data={
                'id': [0, 1],
                'value': [9, 9]
            }
        )

        assert_frame_equal(actual, expected)

    def test_is_fitted(self) -> None:
        model = GlobalMeanValue()
        self.assertFalse(model.is_fitted())
