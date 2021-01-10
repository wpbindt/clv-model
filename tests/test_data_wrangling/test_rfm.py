from datetime import date
import unittest

import pandas
from pandas.testing import assert_frame_equal

from clv_model.data_wrangling.rfm import rfm


class TestDataWrangling(unittest.TestCase):
    def test_rfm(self) -> None:
        transactions = pandas.DataFrame(
            data={
                'id': [0, 0, 0, 1, 1, 2],
                'order_date': [
                    date(2020, 1, 1),
                    date(2020, 1, 4),
                    date(2020, 1, 5),
                    date(2020, 1, 2),
                    date(2020, 1, 6),
                    date(2020, 1, 3)
                ],
                'invoice': [10, 10, 20, 0, 5, 100]
            }
        )

        actual = rfm(
            transactions=transactions,
            customer_id_col='id',
            date_col='order_date',
            value_col='invoice',
        )
        expected = pandas.DataFrame(
            data={
                'customer_id': [0, 1, 2],
                'recency': [1, 0, 3],
                'frequency': [3, 2, 1],
                'T': [5, 4, 3],
                'value': [13.33, 2.5, 100]
            }
        )
        assert_frame_equal(actual, expected, check_dtype=False)

        actual = rfm(
            transactions=transactions,
            customer_id_col='id',
            date_col='order_date',
        )
        expected = pandas.DataFrame(
            data={
                'customer_id': [0, 1, 2],
                'recency': [1, 0, 3],
                'frequency': [3, 2, 1],
                'T': [5, 4, 3],
            }
        )
        assert_frame_equal(actual, expected, check_dtype=False)

        transactions = pandas.DataFrame(
            data={
                'customer_id': [0, 0, 0, 1, 1],
                'date': [
                    date(2020, 1, 1),
                    date(2020, 1, 4),
                    date(2020, 1, 15),
                    date(2020, 1, 2),
                    date(2020, 1, 6),
                ],
                'invoice': [10, 10, 20, 0, 5]
            }
        )
        actual = rfm(
            transactions=transactions,
            customer_id_col='customer_id',
            date_col='date',
            period='W',
            value_col='invoice'
        )
        expected = pandas.DataFrame(
            data={
                'customer_id': [0, 1],
                'recency': [0, 1],
                'frequency': [2, 2],
                'T': [2, 2],
                'value': [15, 2.5]
            }
        )

        assert_frame_equal(actual, expected)
