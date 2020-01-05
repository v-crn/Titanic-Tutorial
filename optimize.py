from config import (SCORING,
                    DIRECTION,
                    TARGET_COLUMN,
                    PATH_TRAIN_PRP,
                    PATH_TRIAL_FOLDER,
                    TIMEOUT,
                    RATIO_MAX_N_FEATURES,
                    IS_UNDER_SAMPLING,
                    N_TRIALS_SELECT,
                    N_TRIALS_TUNE,
                    RANDOM_STATE,
                    N_CV_SPLITS,
                    TEST_SIZE_RATIO)
from vmlkit import utility as utl
from vmlkit.model_selection.optimizer import optimize
from vmlkit import visualizer as viz

import joblib
from ulid import ulid
import pandas as pd
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.linear_model import LogisticRegression


def main():
    # Make a directory for training model
    if not utl.exists_dir(PATH_TRIAL_FOLDER):
        utl.mkdir(PATH_TRIAL_FOLDER)

    # Make a directory with ulid per each trial unit
    ULID = ulid()
    PATH_TRIAL_FOLDER_ULID = PATH_TRIAL_FOLDER + ULID + '/'

    if not utl.exists_dir(PATH_TRIAL_FOLDER_ULID):
        utl.mkdir(PATH_TRIAL_FOLDER_ULID)

    # Mutable Paths
    PATH_FEATURES_OPT = PATH_TRIAL_FOLDER_ULID + 'optimized_features.csv'
    PATH_LOG_OPT_FEATURES\
        = PATH_TRIAL_FOLDER_ULID + 'log_optimize_features.csv'
    PATH_LOG_TUNEUP = PATH_TRIAL_FOLDER_ULID + 'log_tuneup.csv'
    PATH_LOG_TUNEUP_FEATURE_SELECT\
        = PATH_TRIAL_FOLDER_ULID + 'log_tuneup_feature_select.csv'
    PATH_MODEL = PATH_TRIAL_FOLDER_ULID + 'model.joblib'
    PATH_MODEL_FEATURE_SELECT\
        = PATH_TRIAL_FOLDER_ULID + 'model_feature_select.joblib'
    PATH_MODEL_PARAMS = PATH_TRIAL_FOLDER_ULID + 'model_params.txt'
    PATH_MODEL_PARAMS_FEATURE_SELECT\
        = PATH_TRIAL_FOLDER_ULID + 'model_params_feature_select.txt'
    PATH_ROC_CURVE = PATH_TRIAL_FOLDER_ULID + 'roc_curve.png'

    model_feature_select = LogisticRegression(
        C=0.5612715259596806, penalty='l1', solver='liblinear', n_jobs=-1)

    models = {
        'LogisticRegression': LogisticRegression,
    }

    params = {
        'LogisticRegression': {
            'penalty': 'l1',
            'C': ('log', 0.1, 5),
            'class_weight': 'balanced',
            'solver': ('cat', ('newton-cg', 'lbfgs', 'liblinear')),
            'n_jobs': -1,
            'random_state': RANDOM_STATE
        },
    }

    if IS_UNDER_SAMPLING:
        base_model = model_feature_select

        models = {
            'BalancedBaggingClassifier': BalancedBaggingClassifier
        }

        params = {
            'BalancedBaggingClassifier': {
                'base_estimator': base_model,
                'n_estimators': ('int', 45, 55),
                'n_jobs': -1,
                'sampling_strategy': 'auto',
                'random_state': RANDOM_STATE,
            }
        }

    # Loading preprocessed data
    train_prp = joblib.load(PATH_TRAIN_PRP)
    y_prp = train_prp[TARGET_COLUMN]
    X_prp = utl.except_for(train_prp, TARGET_COLUMN)

    max_n_features = RATIO_MAX_N_FEATURES * X_prp.shape[1]

    # Optimization
    model_optimized\
        = optimize(
            models=models, params=params, X=X_prp, y=y_prp,
            model_feature_select=model_feature_select,
            max_n_features=max_n_features,
            scoring=SCORING,
            direction=DIRECTION,
            n_trials_select=N_TRIALS_SELECT,
            n_trials_tune=N_TRIALS_TUNE,
            timeout=TIMEOUT, n_jobs=-1,
            path_model_feature_select=PATH_MODEL_FEATURE_SELECT,
            path_model_params_feature_select=PATH_MODEL_PARAMS_FEATURE_SELECT,
            path_log_tuneup_feature_select=PATH_LOG_TUNEUP_FEATURE_SELECT,
            path_model=PATH_MODEL,
            path_model_params=PATH_MODEL_PARAMS,
            path_log_tuneup=PATH_LOG_TUNEUP,
            path_features_opt=PATH_FEATURES_OPT,
            path_log_opt_features=PATH_LOG_OPT_FEATURES)

    """
    Plot ROC curve
    """
    selected_features = list(pd.read_csv(PATH_FEATURES_OPT))
    X_prp_slc = X_prp[selected_features]

    with utl.timer('Plot ROC curve'):
        viz.plot_roc_curve_with_cv(
            model_optimized, X_prp_slc, y_prp,
            cv='StratifiedKFold', n_splits=N_CV_SPLITS,
            test_size_ratio=TEST_SIZE_RATIO,
            savepath=PATH_ROC_CURVE)


if __name__ == '__main__':
    with utl.timer('Optimization'):
        main()
