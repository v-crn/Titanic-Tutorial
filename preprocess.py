import config as c
from vmlkit import utility as utl
from vmlkit import visualizer as viz
from vmlkit.preprocessing.preprocessor import Preprocessor
import pandas as pd


def main():
    # Loading data
    train = pd.read_csv(c.PATH_TRAIN)
    X = utl.except_for(train, c.TARGET_COLUMN)
    X_test = pd.read_csv(c.PATH_TEST)

    # Check if number of each class is inbalanced
    y = train[c.TARGET_COLUMN]
    print('class balance\n', y.value_counts())

    # Create Profile of the data
    if not utl.exists(c.PATH_PROFILE_REPORT_TRAIN):
        with utl.timer('Create report of train'):
            viz.create_profile(X, savepath=c.PATH_PROFILE_REPORT_TRAIN)

    if not utl.exists(c.PATH_PROFILE_REPORT_TEST):
        with utl.timer('Create report of test'):
            viz.create_profile(
                X_test, savepath=c.PATH_PROFILE_REPORT_TEST)

    prp = Preprocessor()
    prp.set_scaler(method=c.SCALING)

    with utl.timer('Preprocessing train'):
        train_prp = prp.exe(
            df=train,
            encoder=c.CAT_ENCODER,
            exclusive_features=c.EXCLUSIVE_FEATURES,
            dropped_features=c.DISCARDED_FEATURES,
            thresh_nan_ratio_per_col=c.THRESH_NAN_RATIO_PER_COL,
            thresh_corr=c.THRESH_CORR,
            alt_num=c.ALT_NUM,
            alt_cat=c.ALT_CAT,
            path_train_prp=c.PATH_TRAIN_PRP)

        y_prp = train_prp[c.TARGET_COLUMN]
        X_prp = utl.except_for(train_prp, c.TARGET_COLUMN)

        print('X_prp.shape:', X_prp.shape)
        print('y_prp.shape:', y_prp.shape)

    if not utl.exists(c.PATH_PROFILE_REPORT_TRAIN_PRP):
        with utl.timer('Create report of train_prp'):
            viz.create_profile(X_prp, savepath=c.PATH_PROFILE_REPORT_TRAIN_PRP)

    with utl.timer('Preprocessing test'):
        X_test_prp = prp.exe_test(
            X=X_test,
            y=None,
            exclusive_features=c.EXCLUSIVE_FEATURES,
            alt_num=c.ALT_NUM, alt_cat=c.ALT_CAT,
            path_test_prp=c.PATH_TEST_PRP)

        print('X_test_prp.shape:', X_test_prp.shape)

    if not utl.exists(c.PATH_PROFILE_REPORT_TEST_PRP):
        with utl.timer('Create report of test_prp'):
            viz.create_profile(
                X_test_prp, savepath=c.PATH_PROFILE_REPORT_TEST_PRP)


if __name__ == '__main__':
    with utl.timer('Preprocess'):
        main()
