# %load_ext autoreload
# %autoreload 2
from config import (CAT_ENCODER,
                    TARGET_COLUMN,
                    DISCARDED_FEATURES,
                    EXCLUSIVE_FEATURES,
                    PATH_TRAIN,
                    PATH_TEST,
                    PATH_TRAIN_PRP,
                    PATH_TEST_PRP,
                    PATH_PROFILE_REPORT,
                    PATH_PROFILE_REPORT_PRP)
from vmlkit import utility as utl
from vmlkit import visualizer as viz
from vmlkit.preprocessing.preprocessor import Preprocessor
import pandas as pd


"""
1. Preprocessing
"""


def main():
    # Loading data
    train = pd.read_csv(PATH_TRAIN)
    X = utl.except_for(train, TARGET_COLUMN)
    X_test = pd.read_csv(PATH_TEST)

    # Check if number of each class is inbalanced
    y = train[TARGET_COLUMN]
    print('class balance', y.value_counts())

    # Create Profile of the data
    if not utl.exists(PATH_PROFILE_REPORT):
        viz.create_profile(X, savepath=PATH_PROFILE_REPORT)

    prp = Preprocessor()

    train_prp = prp.exe(
        df=train,
        encoder=CAT_ENCODER,
        exclusive_features=EXCLUSIVE_FEATURES,
        dropped_features=DISCARDED_FEATURES,
        thresh_drop=4,
        thresh_corr=0.95,
        alt_num='mean', alt_cat='mode',
        savepath=PATH_TRAIN_PRP)

    X_test_prp = prp.exe_test(
        X_test, exclusive_features=EXCLUSIVE_FEATURES,
        alt_num='mean', alt_cat='mode',
        savepath=PATH_TEST_PRP)

    y_prp = train_prp[TARGET_COLUMN]
    X_prp = utl.except_for(train_prp, TARGET_COLUMN)

    print('X_prp.shape:', X_prp.shape)
    print('y_prp.shape:', y_prp.shape)
    print('X_test_prp.shape:', X_test_prp.shape)

    # Create Profile of the preprocessed data
    if not utl.exists(PATH_PROFILE_REPORT_PRP):
        viz.create_profile(X_prp, savepath=PATH_PROFILE_REPORT_PRP)


if __name__ == '__main__':
    with utl.timer('Preprocess'):
        main()
