from vmlkit import utility as utl
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.neighbors import LocalOutlierFactor


def drop_correlatives(df, threshold=0.95, inplace=False):
    # Create correlation matrix
    corr_matrix = df.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find index of feature columns with correlation greater than threshold
    to_drop = [column for column in upper.columns if any(
        upper[column] > threshold)]

    return df.drop(df[to_drop], axis=1, inplace=inplace)


def replace_categorical_na(df, alt='mode'):
    cat_cols = utl.get_categorical_columns(df)
    alt_value = df[cat_cols].mode()

    if alt == 'nan':
        alt_value = np.nan

    for col in cat_cols:
        df[col].fillna(alt_value[col], inplace=True)

    return df


def replace_numerical_na(df, alt='mean'):
    num_cols = utl.get_numerical_columns(df)
    alt_value = df[num_cols].mean(skipna=True)

    if alt == 'Bayesian':
        alt_value = bayesian_average(df[num_cols])

    for col in num_cols:
        df[col].fillna(alt_value[col], inplace=True)

    return df


def bayesian_average(df):
    n = len(df)
    s = df.sum()
    m = df.mean(skipna=True)
    n0 = df.isnull().sum()

    return (s + n0 * m) / (n + n0)


def clip_outlier_by_std(df, alt='minmax', n_std=3):
    """
    Clip n-times-of-std outlier in numerical features
    """
    for col in utl.get_numerical_columns(df):
        x = df[col]
        average = np.nanmean(x, axis=0)
        std = np.std(x)

        # 外れ値の基準点
        outlier_min = average - std * n_std
        outlier_max = average + std * n_std

        # 範囲から外れている値を置換
        if alt == 'minmax':
            df[col][x < outlier_min] = outlier_min
            df[col][x > outlier_max] = outlier_max
        elif alt == 'mean':
            df[col][x < outlier_min] = average
            df[col][x > outlier_max] = average
        elif alt == 'mode':
            mode = np.mode(df[col])[0]
            df[col][x < outlier_min] = mode
            df[col][x > outlier_max] = mode
        else:
            df[col][x < outlier_min] = None
            df[col][x > outlier_max] = None

    return df


def clip_outlier_by_lof(df, alt='mean',
                        n_neighbors=20, algorithm='auto',
                        leaf_size=30, metric='minkowski',
                        p=2, metric_params=None,
                        contamination=5e-3,
                        novelty=True, n_jobs=None):
    num_cols = utl.get_numerical_columns(df)

    for col in num_cols:
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, algorithm=algorithm,
                                 leaf_size=leaf_size, metric=metric,
                                 p=p, metric_params=metric_params,
                                 contamination=contamination,
                                 novelty=novelty, n_jobs=n_jobs)

        d = np.array(df[col]).reshape(-1, 1)
        lof.fit(d)
        pred = lof.predict(d)
        idx_outlier = [i for i, v in enumerate(pred) if v == -1]

        if alt == 'mean':
            df[col].iloc[idx_outlier] = np.nanmean(df[col], axis=0)
        elif alt == 'mode':
            df[col].iloc[idx_outlier] = np.mode(df[col])[0]
        else:
            df[col].iloc[idx_outlier] = None

    return df


def clip_outlier_with_smirnov_grubbs(
        df, alpha=0.05, alt=None, outlier=False):
    """
    Generate outlier-removed df with Smirnov-Grubbs.

    Parameters:
        df: array_like
            Numeric df values.
        alpha: float
            Significance level. (two-sided)

    Returns:
        outlier removed df (if outlier=True)
        outlier df
    """
    outlier = pd.Series()
    while True:
        n = len(df)
        t = stats.t.isf(q=(alpha / n) / 2, df=n - 2)
        tau = (n - 1) * t / np.sqrt(n * (n - 2) + n * t * t)
        i_min, i_max = np.argmin(df), np.argmax(df)
        ave, std = np.nanmean(df, axis=0), np.std(df, ddof=1)
        if np.abs(df[i_max] - ave) > np.abs(df[i_min] - ave):
            i_far = i_max
        else:
            i_far = i_min
        tau_far = np.abs((df[i_far] - ave) / std)
        if tau_far < tau:
            break
        outlier.append(df[i_far])

        if alt == 'mean':
            df[i_far] = ave
        else:
            df[i_far] = None
    if outlier:
        return df, outlier

    return df
