from logging import getLogger, Logger
import typing
import unittest
from unittest.mock import Mock

import numpy
import pandas
from pandas.testing import assert_frame_equal

from clv_model.value_model import GammaGamma


class TestGammaGamma(unittest.TestCase):
    def _get_model(self) -> GammaGamma:
        return GammaGamma(
            p=numpy.array([3, 2]),
            q=numpy.array([9, 2]),
            mu=numpy.array([10, 2]),
            logger=getLogger()
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

    def test_predict_not_fitted(self) -> None:
        model = GammaGamma(logger=getLogger())
        data = pandas.DataFrame(columns={'id', 'frequency', 'value'})
        with self.assertRaises(ValueError) as error:
            model.predict(data)
        self.assertEqual(
            str(error.exception),
            'Model must be fitted by calling fit before calling predict.'
        )

    def test_predict_warning(self) -> None:
        mock_logger = typing.cast(Logger, Mock())
        model = GammaGamma(
            p=numpy.array([2, 3, 9]),
            q=numpy.array([9, 0.2, 9]),
            mu=numpy.array([10, 10, 10]),
            logger=mock_logger
        )
        data = pandas.DataFrame(columns={'id', 'frequency', 'value'})
        model.predict(data)
        mock_logger.warning.assert_called()
