from vmlkit import utility as utl
from vmlkit.preprocessing import cleaner as cln
from vmlkit.preprocessing import scaler
from vmlkit.preprocessing import encoder as enc

import joblib
import pandas as pd


class Preprocessor():

    def __init__(self):
        self.rs = scaler.ReScaler()
        self.encoder = None
        self.cols_to_drop = []

    def exe(self, df, encoder='ordinal',
            exclusive_features=None,
            dropped_features=None,
            thresh_nan_ratio_per_col=0.5,
            thresh_corr=0.95,
            alt_num='mean',
            alt_cat='mode',
            path_train_prp=None):
        print('Shape before preprocessing:\n', df.shape)

        # Exclude specified features from preprocessing
        cols = utl.get_columns(df)
        cols_exclusive = utl.intersect([exclusive_features, cols])
        df_ = df.drop(cols_exclusive, axis=1)

        # Columns to drop
        cols_to_drop = []
        cols_to_drop.extend(dropped_features)

        # Columns with at least (thresh_n_nan_per_col) non-NA values
        thresh_n_nan_per_col = round(
            df_.shape[0] * thresh_nan_ratio_per_col)
        counts_nan = df_.isnull().sum()
        cols_nan = list(counts_nan[counts_nan > thresh_n_nan_per_col].index)
        cols_to_drop.extend(cols_nan)

        # Highly Correlated columns to drop
        cols_to_drop.extend(utl.get_correlative_columns(
            df_, threshold=thresh_corr))

        # Drop columns
        print('\nDropped columns:', cols_to_drop)
        df_.drop(cols_to_drop, axis=1, errors='ignore', inplace=True)
        self.cols_to_drop = cols_to_drop

        # Replace numerical NaN
        df_ = cln.replace_numerical_na(df_, alt=alt_num)

        # Standardization
        df_ = self.rs.standardize(df_)

        # Replace categorical NaN
        df_ = cln.replace_categorical_na(df_, alt=alt_cat)

        # Convert categorical data to numerical format
        if encoder == 'one-hot':
            df_, self.encoder = enc.one_hot_encode(df_, need_encoder=True)
        if encoder == 'ordinal':
            df_, self.encoder = enc.ordinal_encode(df_, need_encoder=True)
        if encoder == 'target':
            df_, self.encoder = enc.target_encode(df_, need_encoder=True)

        df = pd.concat([df[cols_exclusive], df_], axis=1)
        print('\nShape after preprocessing:\n', df.shape)

        if path_train_prp:
            joblib.dump(df, path_train_prp, compress=3)

        return df

    def exe_test(self, df,
                 exclusive_features=None,
                 alt_num='mean', alt_cat='mode',
                 save=True,
                 path_test_prp='test_prp.joblib'):
        print('Shape before preprocessing:\n', df.shape)

        # Exclude specified features from preprocessing
        cols = utl.get_columns(df)
        cols_exclusive = utl.intersect([exclusive_features, cols])
        df_ = df.drop(cols_exclusive, axis=1)

        # Drop columns as same as exe()
        print('\nDropped columns:', self.cols_to_drop)
        df_.drop(self.cols_to_drop, axis=1, inplace=True)

        # Replace numerical NaN
        df_ = cln.replace_numerical_na(df_, alt=alt_num)

        # Standardization
        df_ = self.rs.re_standardize(df_)

        # Replace categorical NaN
        df_ = cln.replace_categorical_na(df_, alt=alt_cat)

        # Convert categorical data to numerical format
        cat_cols = utl.get_categorical_columns(df_)
        encoded = self.encoder.transform(df_[cat_cols])
        df_ = pd.concat([df_.drop(cat_cols, axis=1), encoded], axis=1)

        df = pd.concat([df[cols_exclusive], df_], axis=1)
        print('\nShape after preprocessing:\n', df.shape)

        if path_test_prp:
            joblib.dump(df, path_test_prp, compress=3)

        return df
