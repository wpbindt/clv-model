import typing

import numpy
import pandas

__all__ = ('rfm',)


def rfm(
    transactions: pandas.DataFrame,
    customer_id_col: str,
    date_col: str,
    value_col: typing.Optional[str] = None,
    period: str = 'D',
    observation_period_end: typing.Optional[typing.Any] = None
) -> pandas.DataFrame:
    """
    Transforms transactional data of the form
        (customer_id, order_date, <monetary_value>)
    to an rfm table of the form
        (customer_id, recency, frequency, <value>, T).

    - recency is the amount of time passed between the customer's
      final observed transaction up to the end of the observation
      period
    - frequency is the number of purchases the customer made during
      the observation period, minus 1
    - T is the amount of time passed since the customer's first
      observed transaction up to the end of the observation period
    - monetary value is the average value of the customer's observed
      transactions. The first transaction is not counted, so for a
      customer with exactly one purchase, the value will be 0

    Here, time is measured in units specified by period, which defaults
    to day. Multiple transactions occurring in the same period will be
    counted as a single transaction whose value is the sum of the values
    of the component transactions.
    """
    if value_col is not None:
        assert value_col in transactions.columns
    _check_column_presence(
        wanted={date_col, customer_id_col},
        present=set(transactions.columns)
    )

    if observation_period_end is None:
        observation_period_end = pandas.to_datetime(
            transactions[date_col].max()
        )

    transactions_by_period = (
        transactions
        .rename(
            columns={
                customer_id_col: 'customer_id',
                date_col: 'date',
                value_col: 'value',
            }
        )
        .query('date <= @observation_period_end')
        .pipe(_group_by_period, period=period)
    )

    rf = _determine_recency_frequency(
        transactions=transactions_by_period,
        observation_period_end=observation_period_end,
        period=period
    )

    if value_col is None:
        return rf

    m = _determine_monetary_value(transactions_by_period)
    return (
        rf
        .merge(right=m, on='customer_id', how='left')
        .fillna(0)
    )


def _determine_monetary_value(
    transactions: pandas.DataFrame
) -> pandas.DataFrame:
    _check_column_presence(
        wanted={'value', 'customer_id'},
        present=set(transactions.columns)
    )

    return (
        transactions
        .pipe(_drop_first_transaction)
        [['value', 'customer_id']]
        .groupby('customer_id', as_index=False, sort=False)
        .mean()
    )


def _determine_recency_frequency(
    transactions: pandas.DataFrame,
    observation_period_end: typing.Any,
    period: str,
) -> pandas.DataFrame:
    _check_column_presence(
        wanted={'date', 'customer_id'},
        present=set(transactions.columns)
    )

    return (
        transactions
        .groupby('customer_id', sort=False)
        .date
        .agg(['min', 'max', 'count'])
        .reset_index()
        .assign(
            T=lambda df: _timedelta_to_int(
                observation_period_end - df['min'],
                period
            ),
            recency=lambda df: _timedelta_to_int(
                observation_period_end - df['max'],
                period
            ),
            frequency=lambda df: df['count'] - 1
        )
        [['customer_id', 'recency', 'frequency', 'T']]
    )


def _timedelta_to_int(delta_series, period):
    return delta_series / numpy.timedelta64(1, period)


def _group_by_period(
    transactions: pandas.DataFrame,
    period: str
) -> pandas.DataFrame:
    _check_column_presence(
        wanted={'date', 'customer_id'},
        present=set(transactions.columns)
    )

    return (
        transactions
        .assign(
            date=lambda df: pandas.to_datetime(df.date).dt.to_period(period)
        )
        .groupby(['customer_id', 'date'], sort=False, as_index=False)
        .sum()
        .assign(date=lambda df: df.date.dt.to_timestamp())
    )


def _drop_first_transaction(
    transactions: pandas.DataFrame
) -> pandas.DataFrame:
    _check_column_presence(
        wanted={'date', 'customer_id'},
        present=set(transactions.columns)
    )

    return (
        transactions
        .sort_values(by='date', ascending=True)
        .assign(
            first=lambda df: ~ df.customer_id.duplicated()
        )
        .query('~first')
        .drop('first', axis=1)
    )


def _check_column_presence(
    wanted: typing.Set[str],
    present: typing.Set[str]
) -> None:
    for column in wanted:
        assert column in present
