from vmlkit import utility as utl
from vmlkit.preprocessing import cleaner as cln
from vmlkit.preprocessing import scaler
from vmlkit.preprocessing import encoder as enc
import pandas as pd


class Preprocessor():

    def __init__(self):
        self.rs = scaler.ReScaler()
        self.encoder = None
        self.cols_to_drop = []

    def exe(self, df, encoder='ordinal',
            exclusive_features=None,
            dropped_features=None, thresh_drop=4,
            thresh_corr=0.95,
            alt_num='mean', alt_cat='mode',
            savepath='train_prp.pkl'):
        print('Shape before preprocessing:\n', df.shape)

        # Exclude specified features from preprocessing
        cols = utl.get_columns(df)
        cols_exclusive = utl.intersect([exclusive_features, cols])
        df_ = df.drop(cols_exclusive, axis=1)

        # Columns to drop
        cols_to_drop = []
        cols_to_drop.extend(dropped_features)

        # Highly Correlated columns to drop
        cols_to_drop.extend(utl.get_correlative_columns(
            df_, threshold=thresh_corr))

        # Drop columns
        print('\nDropped columns:', cols_to_drop)
        df_.drop(cols_to_drop, axis=1, errors='ignore', inplace=True)
        self.cols_to_drop = cols_to_drop

        # Drop rows occupied with NaN (thresh: not NaN columns number at a row)
        df_.dropna(thresh=thresh_drop, inplace=True)

        # Replace numerical NaN
        df_ = cln.replace_numerical_na(df_, alt=alt_num)

        # Standardization
        df_ = self.rs.standardize(df_)

        # Replace categorical NaN
        df_ = cln.replace_categorical_na(df_, alt=alt_cat)

        # Convert categorical data to numerical format
        if encoder == 'one-hot':
            df_, self.encoder = enc.one_hot_encode(df_, need_encoder=True)
        else:
            df_, self.encoder = enc.ordinal_encode(df_, need_encoder=True)

        df = pd.concat([df[cols_exclusive], df_], axis=1)
        print('\nShape after preprocessing:\n', df.shape)

        if savepath:
            utl.save(df, savepath)

        return df

    def exe_test(self, df,
                 exclusive_features=None,
                 alt_num='mean', alt_cat='mode',
                 save=True,
                 savepath='test_prp.pkl'):
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

        if save:
            utl.save(df, savepath)

        return df

    def exe_subsequently(self, df_list, exclusive_features=None,
                         dropped_features=None, thresh_drop=4,
                         thresh_corr=0.95,
                         alt_num='mean', alt_cat='mode',
                         save=True,
                         savepath='preprocessed_list.pkl'):
        isnt_list = False
        if not isinstance(df_list, list):
            df_list = [df_list]
            isnt_list = True

        keys = range(len(df_list))
        df = pd.concat(df_list, keys=keys)

        # Exclude specified features from preprocessing
        cols = utl.get_columns(df)
        ex = utl.intersect([exclusive_features, cols])
        df_ = df.drop(ex, axis=1, errors='ignore')

        # Drop unnecessary features
        df_.drop(dropped_features, axis=1, inplace=True)

        # Drop Highly Correlated Features
        df_ = cln.drop_correlatives(df_, threshold=thresh_corr)

        # Drop NaN
        df_.dropna(thresh=thresh_drop, inplace=True)

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

        df = pd.concat([df[ex], df_], axis=1)
        df_list = [df.xs(keys[i]) for i in keys]

        if isnt_list:
            df_list = df_list[0]
        if save:
            utl.save(df_list, savepath)

        return df_list
