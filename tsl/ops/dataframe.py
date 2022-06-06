from typing import Union, Callable

import numpy as np
import pandas as pd

from tsl.typing import Index


def to_numpy(df):
    if df.columns.nlevels == 1:
        return df.to_numpy()
    cols = [df.columns.unique(i) for i in range(df.columns.nlevels)]
    cols = pd.MultiIndex.from_product(cols)
    df = df.reindex(columns=cols)
    return df.values.reshape((-1, *cols.levshape))


def aggregate(df: pd.DataFrame, index: Index, aggr_fn: Callable = np.sum,
              axis: int = 1, level: int = 0):
    """Aggregate rows/columns in (MultiIndexed) DataFrame according to a new
    index.

    Args:
        df (pd.DataFrame): :class:`~pandas.DataFrame` to be aggregated.
        index (Index): A sequence of :obj:`cluster_id` with length equal to
            the index over which aggregation is performed. The :obj:`i`-th
            element of index at :obj:`axis` and :obj:`level` will be mapped to
            :obj:`index[i]`-th position in new index.
        aggr_fn (Callable): Function to be used for aggregation.
        axis (int): Axis over which performing aggregation, :obj:`0` for index,
        :obj:`1` for columns.
        (default :obj:`1`)
        level (int): Level over which performing aggregation if :obj:`axis` is
        a :class:`~pandas.MultiIndex`.
        (default :obj:`0`)
    """
    if axis == 0:
        df = df.groupby(index, axis=0).aggregate(np.min)
    elif axis == 1:
        cols = [df.columns.unique(i).values for i in range(df.columns.nlevels)]
        cols[level] = index
        grouper = pd.MultiIndex.from_product(cols, names=df.columns.names)
        df = df.groupby(grouper, axis=1).aggregate(aggr_fn)
        df.columns = pd.MultiIndex.from_tuples(df.columns, names=grouper.names)
    return df


def compute_mean(x: Union[pd.DataFrame, np.ndarray],
                 index: pd.DatetimeIndex = None
                 ) -> Union[pd.DataFrame, np.ndarray]:
    """Compute the mean values for each row.

    The mean is first computed hourly over the week of the year. Further
    :obj:`NaN` values are imputed using hourly mean over the same month through
    the years. If other :obj:`NaN` are present, they are replaced with the mean
    of the sole hours. Remaining missing values are filled with :obj:`ffill` and
    :obj:`bfill`.

    Args:
        x (np.array | pd.Dataframe): Array-like with missing values.
        index (pd.DatetimeIndex | pd.PeriodIndex | pd.TimedeltaIndex, optional):
            Temporal index if x is not a :obj:'~pandas.Dataframe' with a
            temporal index. Must have same length as :obj:`x`.
            (default :obj:`None`)
    """
    if index is not None:
        if not isinstance(index, pd.DatetimeIndex):
            # try casting
            index = pd.to_datetime(index)
        assert len(index) == len(x)
        if isinstance(x, pd.DataFrame):
            # override index of x
            df_mean = x.copy().set_index(index)
        else:
            # try casting to np.ndarray
            x = np.asarray(x)
            shape = x.shape
            # x can be N-dimensional, we flatten all but the first dimensions
            x = x.reshape((shape[0], -1))
            df_mean = pd.DataFrame(x, index=index)
    elif isinstance(x, pd.DataFrame):
        df_mean = x.copy()
    else:
        raise TypeError("`x` must be a pd.Dataframe or a np.ndarray.")
    cond0 = [df_mean.index.year, df_mean.index.isocalendar().week,
             df_mean.index.hour]
    cond1 = [df_mean.index.year, df_mean.index.month, df_mean.index.hour]
    conditions = [cond0, cond1, cond1[1:], cond1[2:]]
    while df_mean.isna().values.sum() and len(conditions):
        nan_mean = df_mean.groupby(conditions[0]).transform(np.nanmean)
        df_mean = df_mean.fillna(nan_mean)
        conditions = conditions[1:]
    if df_mean.isna().values.sum():
        df_mean = df_mean.fillna(method='ffill')
        df_mean = df_mean.fillna(method='bfill')
    if isinstance(x, np.ndarray):
        df_mean = df_mean.values.reshape(shape)
    return df_mean
