import typing

import pandas

__all__ = ('rfm',)


def rfm(
    transactions: pandas.DataFrame,
    customer_id_col: str,
    date_col: str,
    value_col: typing.Optional[str] = None,
    period: str = 'D',
    observation_period_end: typing.Optional[typing.Any] = None,
) -> pandas.DataFrame:
    """
    Transforms transactional data of the form
        (id, order_date, <monetary_value>)
    to an rfm table of the form
        (id, recency, frequency, <value>, T).

    - recency is the amount of time passed between the customer's
      final observed transaction up to the end of the observation
      period
    - frequency is the number of purchases the customer made during
      the observation period, minus 1
    - T is the amount of time passed since the customer's first
      observed transaction up to the end of the observation period
    - monetary value is the average value of the customer's observed
      transactions.

    Here, time is measured in units specified by period, which defaults
    to day. Multiple transactions occurring in the same period will be
    counted as a single transaction whose value is the mean of the values
    of the component transactions.
    """
    if value_col is not None:
        wanted_columns = {date_col, customer_id_col, value_col}
    else:
        wanted_columns = {date_col, customer_id_col}
    _check_column_presence(
        wanted=wanted_columns,
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
                customer_id_col: 'id',
                date_col: 'date',
                value_col: 'value',
            }
        )
        .query('date <= @observation_period_end')
        .pipe(_group_by_period, period=period)
    )

    observation_period_end = observation_period_end.to_period(period)

    rf = _determine_recency_frequency(
        transactions=transactions_by_period,
        observation_period_end=observation_period_end,
    )

    if value_col is None:
        return rf

    m = _determine_monetary_value(transactions_by_period)
    return (
        rf
        .merge(right=m, on='id', how='left')
        .fillna(0)
    )


def _determine_monetary_value(
    transactions: pandas.DataFrame,
) -> pandas.DataFrame:
    _check_column_presence(
        wanted={'value', 'id'},
        present=set(transactions.columns)
    )

    return (
        transactions
        [['value', 'id']]
        .groupby('id', as_index=False, sort=False)
        .mean()
        .assign(
            value=lambda df: df.value.round(2)
        )
    )


def _determine_recency_frequency(
    transactions: pandas.DataFrame,
    observation_period_end: typing.Any,
) -> pandas.DataFrame:
    _check_column_presence(
        wanted={'date', 'id'},
        present=set(transactions.columns)
    )

    return (
        transactions
        .groupby('id', sort=False)
        .date
        .agg(['min', 'max', 'count'])
        .reset_index()
        .assign(
            T=lambda df:
            (observation_period_end - df['min']).apply(lambda x: x.n),
            recency=lambda df:
            (observation_period_end - df['max']).apply(lambda x: x.n),
            frequency=lambda df: df['count']
        )
        [['id', 'recency', 'frequency', 'T']]
    )


def _group_by_period(
    transactions: pandas.DataFrame,
    period: str
) -> pandas.DataFrame:
    _check_column_presence(
        wanted={'date', 'id'},
        present=set(transactions.columns)
    )

    return (
        transactions
        .assign(
            date=lambda df: pandas.to_datetime(df.date).dt.to_period(period)
        )
        .groupby(['id', 'date'], sort=False, as_index=False)
        .mean()
    )


def _check_column_presence(
    wanted: typing.Set[str],
    present: typing.Set[str]
) -> None:
    for column in wanted:
        if column not in present:
            raise ValueError(f'Column "{column}" not found in dataset.')
