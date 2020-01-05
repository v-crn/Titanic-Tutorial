from vmlkit import utility as utl
import numpy as np
import scipy.stats as stats
from sklearn.preprocessing import (StandardScaler, MinMaxScaler,
                                   PowerTransformer, RobustScaler,
                                   QuantileTransformer)


class ReScaler():

    def __init__(self):
        self.std = StandardScaler()

    def standardize(self, df):
        std = StandardScaler()
        cols_std = utl.get_columns_for_std(df)
        df.loc[:, cols_std] = std.fit_transform(df[cols_std])
        self.std = std

        return df

    def re_standardize(self, df):
        cols_std = utl.get_columns_for_std(df)
        df.loc[:, cols_std] = self.std.transform(df[cols_std])

        return df


def standardize(df):
    std = StandardScaler()
    cols_std = utl.get_columns_for_std(df)
    df.loc[:, cols_std] = std.fit_transform(df[cols_std])

    return df


def minimax_scale(df):
    """
    Transform features by scaling each feature to a given range.

    Notion:
    - The averages are not always 0
    - Easily affected by outlier
    """
    mms = MinMaxScaler()
    num_cols = utl.get_numerical_columns(df)
    df[num_cols] = mms.fit_transform(df[num_cols])

    return df


def robust_scale(df, quantile_range=(25.0, 75.0)):
    rbs = RobustScaler(with_centering=True, with_scaling=True,
                       quantile_range=quantile_range, copy=True)
    num_cols = utl.get_numerical_columns(df)
    df[num_cols] = rbs.fit_transform(df[num_cols])

    return df


def gaussian_transform(df, n_quantiles=1000,
                       ignore_implicit_zeros=False, subsample=100000,
                       random_state=None, copy=True):
    qt = QuantileTransformer(n_quantiles=n_quantiles,
                             output_distribution='normal',
                             ignore_implicit_zeros=ignore_implicit_zeros,
                             subsample=subsample,
                             random_state=random_state, copy=copy)
    num_cols = utl.get_numerical_columns(df)
    df[num_cols] = qt.fit_transform(df[num_cols])

    return df


def uniform_transform(df, n_quantiles=1000,
                      ignore_implicit_zeros=False, subsample=100000,
                      random_state=None, copy=True):
    qt = QuantileTransformer(n_quantiles=n_quantiles,
                             output_distribution='uniform',
                             ignore_implicit_zeros=ignore_implicit_zeros,
                             subsample=subsample,
                             random_state=random_state, copy=copy)
    num_cols = utl.get_numerical_columns(df)
    df[num_cols] = qt.fit_transform(df[num_cols])

    return df


def log_abs_transform(df):
    num_cols = utl.get_numerical_columns(df)
    x = df[num_cols]
    df[num_cols] = np.sign(x) * np.log(np.abs(x))

    return df


def box_cox_transform(df, include_missing_value=False):
    num_cols = utl.get_numerical_columns(df)
    if include_missing_value:
        pos_cols = [c for c in num_cols if ~(df[c] <= 0.0).all()]
    else:
        pos_cols = [c for c in num_cols if (df[c] > 0.0).all()]

    pt = PowerTransformer(method='box-cox')
    df[pos_cols] = pt.fit_transform(df[pos_cols])

    return df


def yeo_johnson_transform(df):
    num_cols = utl.get_numerical_columns(df)
    pt = PowerTransformer(method='yeo-johnson')
    df[num_cols] = pt.fit_transform(df[num_cols])

    return df


def rank_transform(df, method='average'):
    """
    Assign ranks to df, dealing with ties appropriately.

    Parameters:
        df : array_like
            The array of values to be ranked.
            The array is first flattened.

        method : str, optional
            The method used to assign ranks to tied elements.
            The options are ‘average’, ‘min’, ‘max’, ‘dense’
            and ‘ordinal’.

        ‘average’:
            The average of the ranks that would have been assigned to all
            the tied values is assigned to each value.

        ‘min’:
            The minimum of the ranks that would have been assigned to all
            the tied values is assigned to each value. (This is also
            referred to as “competition” ranking.)

        ‘max’:
            The maximum of the ranks that would have been assigned to all
            the tied values is assigned to each value.

        ‘dense’:
            Like ‘min’, but the rank of the next highest element is
            assigned the rank immediately after those assigned to the
            tied elements.

        ‘ordinal’:
            All values are given a distinct rank, corresponding to the
            order that the values occur in a.

    Returns:
        ranks : ndarray
            An array of length equal to the size of a, containing rank
            scores.

    [Notion]
    - Apply to train df and test df together
    """
    num_cols = utl.get_numerical_columns(df)
    df[num_cols] = stats.rankdata(df[num_cols], method)

    return df
