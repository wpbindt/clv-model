import unittest

import numpy
import pandas
from pandas.testing import assert_frame_equal

from clv_model.value_model import GammaGamma


class TestGammaGamma(unittest.TestCase):
    def _get_model(self) -> GammaGamma:
        return GammaGamma(
            p=numpy.array([3, 2]),
            q=numpy.array([9, 2]),
            mu=numpy.array([10, 2])
        )

    def test_predict(self) -> None:
        model = self._get_model()
        data = pandas.DataFrame(
            data={
                'id': [0, 1],
                'frequency': [9, 4],
                'value': [1, 6]
            }
        )
        actual = model.predict(data)
        expected = pandas.DataFrame(
            data={
                'id': [0, 1],
                'value': [1.39, 5.44]
            }
        )

        assert_frame_equal(actual, expected)

    def test_predict_empty(self) -> None:
        model = self._get_model()
        data = pandas.DataFrame(columns={'id', 'frequency', 'value'})
        actual = model.predict(data)
        expected = pandas.DataFrame(columns=['id', 'value'])

        assert_frame_equal(actual, expected)
