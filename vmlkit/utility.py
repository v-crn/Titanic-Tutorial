import config as c
from vmlkit import utility as utl

import os
from ulid import ulid
import time
from contextlib import contextmanager
import joblib
import numpy as np
import pandas as pd
import csv
import json
import codecs


def get_path_trial_folder_ulid(create_new):
    # Make a cache directory
    if not utl.exists_dir(c.PATH_CACHE_FOLDER):
        utl.mkdir(c.PATH_CACHE_FOLDER)

    # Make a directory for trial
    if not utl.exists_dir(c.PATH_TRIAL_FOLDER):
        utl.mkdir(c.PATH_TRIAL_FOLDER)

    # For test
    if c.TRIAL_FOLDER_NAME_FOR_TEST is not None:
        return c.PATH_TRIAL_FOLDER_FOR_TEST

    # Make a directory with ulid per each trial unit
    if create_new:
        trial_id = ulid()
        path_trial_folder_ulid = c.MODEL_NAME + trial_id
        utl.mkdir(path_trial_folder_ulid)
        print(trial_id, file=codecs.open(c.PATH_ULID, 'w', 'utf-8'))
        print('New tirial folder created!')
    else:
        for line in open(c.PATH_ULID):
            trial_id = line.replace('\n', '')

    path_trial_folder_ulid\
        = c.PATH_TRIAL_FOLDER + trial_id + '_' + c.MODEL_NAME + '/'

    if not utl.exists_dir(path_trial_folder_ulid):
        utl.mkdir(path_trial_folder_ulid)

    return path_trial_folder_ulid


def get_columns_matched_bool_list(df, bool_list, boolean=True):
    idx = get_idx_matched_value(bool_list, boolean)

    return get_values_matched_idx(get_columns(df), idx)


def get_idx_matched_value(lst, value):
    return [idx for idx, v in enumerate(lst) if v == value]


def get_values_matched_idx(lst, idx_list):
    return [lst[i] for i in idx_list]


def bool2binary(lst):
    return [1 if x else 0 for x in lst]


def uniq(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if x not in seen and not seen_add(x)]


def intersect(lst):
    return list(set(lst[0]).intersection(*lst[1:]))


def flatten(lst):
    return [z for y in lst for z in (
        flatten(y)
        if hasattr(y, '__iter__')
        and not isinstance(y, str) else (y,))]


def get_idx_list(df_list):
    return [list(df.columns.values) for df in df_list]


def has_columns(lst, df):
    return set(lst).issubset(df.columns)


def exists(path):
    if path is None:
        return False

    return os.path.exists(path)


def exists_dir(path):
    if path is None:
        return False

    return os.path.isdir(path)


def mkdir(path):
    os.mkdir(path)


def load_csv_and_save_pkl(filepath, save=True):
    pkl_path = filepath.replace('csv', 'pkl')
    df = None

    if os.path.exists(pkl_path):
        df = joblib.load(pkl_path)
    elif os.path.exists(filepath):
        if '.csv' in filepath:
            df = pd.read_csv(filepath)
        if save:
            joblib.dump(df, pkl_path, compress=3)
    else:
        raise Exception("The file path doesn't exist.")
    return df


def write_text(text, path):
    with open(path, 'w') as f:
        if isinstance(text, dict):
            return f.write(json.dumps(text))

        return f.write(text)


def save(obj, path, compress=3):
    if isinstance(obj, pd.dfFrame):
        return obj.to_csv(path)

    if '.joblib' not in path:
        with open(path, 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            return writer.writerow(obj)

    return joblib.dump(obj, path, compress=compress)


def write_csv(obj, path):
    if isinstance(obj, pd.dfFrame):
        return obj.to_csv(path)

    with open(path, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        return writer.writerow(obj)


def read_dict_from_csv(file):
    with open(file, newline="") as f:
        read_dict = csv.DictReader(f, delimiter=",", quotechar='"')
        ks = read_dict.fieldnames
        return_dict = {k: [] for k in ks}

        for row in read_dict:
            for k, v in row.items():
                # notice the type of the value is always string.
                return_dict[k].append(v)

    return return_dict


def write_dict_as_csv(file, save_dict):
    with open(file, 'w') as f:
        w = csv.DictWriter(f, file.keys())
        w.writeheader()
        w.writerow(file)


def save_with_ulid(obj, savepath, ext='.joblib', compress=3):
    """
    Ex:
        parent = 'workfolder/filename_'
        ext = '.joblib'
    -> 'workfolder/filename_2989bc46-7263-44c1-9fe7-e9cd475a511f.joblib'
    """
    unq_id = ulid()
    if '.joblib' in savepath:
        savepath = savepath.replace('.joblib', '_')
    joblib.dump(obj, savepath + unq_id + ext, compress=compress)


class save_with_same_ulid:
    """
    Ex:
        parent = 'workfolder/filename_'
        ext = '.joblib'
    -> 'workfolder/filename_2989bc46-7263-44c1-9fe7-e9cd475a511f.joblib'
    """

    def __init__(self):
        self.unq_id = ulid()

    def __call__(self, obj, savepath, ext='.joblib', compress=3):
        if '.joblib' in savepath:
            savepath = savepath.replace('.joblib', '_')
        unq_id = self.unq_id
        return joblib.dump(obj, savepath + unq_id + ext, compress=compress)


class create_same_ulid_path:
    def __init__(self):
        self.unq_id = ulid()

    def __call__(self, savepath, ext='.joblib'):
        if '.joblib' in savepath:
            savepath = savepath.replace('.joblib', '_')
        return savepath + self.unq_id + ext


@contextmanager
def timer(name='process'):
    """
    Usage:
        with timer('process train'):
            (Process)
    """
    print(f'\n[{name}] start\n')
    start_time = time.time()
    yield
    print(f'\n[{name}] done in {time.time() - start_time:.2f} sec\n')


def except_for(df, columns):
    return df.drop(columns, axis=1)


def get_columns(df):
    return list(df)
    # return [c for c in df.columns]


def get_common_columns(df1, df2):
    return np.intersect1d(df1.columns, df2.columns)


def get_categorical_columns(df):
    cols = list(df)
    cat_cols = [c for c in cols if df[c].dtype == 'object']
    return cat_cols


def get_numerical_columns(df):
    cols = list(df)
    num_cols = [c for c in cols if df[c].dtype != 'object']
    return num_cols


def get_columns_for_std(df):
    cols = list(df)
    num_cols = [c for c in cols if df[c].dtype != 'object']
    cols_for_stdz = [c for c in num_cols if len(df[c].unique()) > 2]
    return cols_for_stdz


def get_correlative_columns(df, threshold=0.95, inplace=False):
    # Create correlation matrix
    corr_matrix = df.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find index of feature columns with correlation greater than threshold
    correlative_columns = [column for column in upper.columns if any(
        upper[column] > threshold)]

    return correlative_columns


def get_categorical_features(df):
    cols = list(df)
    cat_cols = [c for c in cols if df[c].dtype == 'object']
    return df[cat_cols]


def get_numerical_features(df):
    cols = list(df)
    num_cols = [c for c in cols if df[c].dtype != 'object']
    return df[num_cols]


def get_target_class_ratio(y, target_class):
    count_target = y[y == target_class].sum()
    count_others = y[y != target_class].sum()
    target_class_ratio = count_target / count_others
    return target_class_ratio


def get_null_count(df):
    return df.isnull().sum()
