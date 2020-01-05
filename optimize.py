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
                    RANDOM_STATE)
from vmlkit import utility as utl
from vmlkit.model_selection.optimizer import optimize

from ulid import ulid
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.linear_model import LogisticRegression


"""
Optimization
"""


def main():
    # Make a directory for training model
    if not utl.exists_dir(PATH_TRIAL_FOLDER):
        utl.mkdir(PATH_TRIAL_FOLDER)

    # Make a directory with ulid per each trial unit
    # ULID = ulid()
    ULID = '01DXTS20N9T09VMN82N0EYRGAR'
    PATH_TRIAL_FOLDER_ULID = PATH_TRIAL_FOLDER + ULID + '/'

    if not utl.exists_dir(PATH_TRIAL_FOLDER_ULID):
        utl.mkdir(PATH_TRIAL_FOLDER_ULID)

    # Mutable Paths
    PATH_FEATURES_OPT = PATH_TRIAL_FOLDER_ULID + 'optimized_features.csv'
    PATH_LOG_OPT_FEATURES\
        = PATH_TRIAL_FOLDER_ULID + 'log_optimize_features.csv'
    PATH_LOG_TUNEUP = PATH_TRIAL_FOLDER_ULID + 'log_tuneup.csv'
    PATH_LOG_TUNEUP_INIT = PATH_TRIAL_FOLDER_ULID + 'log_tuneup_init.csv'
    PATH_MODEL = PATH_TRIAL_FOLDER_ULID + 'model.pkl'
    PATH_MODEL_INIT = PATH_TRIAL_FOLDER_ULID + 'model_init.pkl'
    PATH_MODEL_PARAMS = PATH_TRIAL_FOLDER_ULID + 'model_params.txt'
    PATH_MODEL_PARAMS_INIT = PATH_TRIAL_FOLDER_ULID + 'model_params_init.txt'

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
        base_model = LogisticRegression()

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
    train_prp = utl.load(PATH_TRAIN_PRP)
    y_prp = train_prp[TARGET_COLUMN]
    X_prp = utl.except_for(train_prp, TARGET_COLUMN)

    max_n_features = RATIO_MAX_N_FEATURES * X_prp.shape[1]

    # Optimization
    optimize(models=models, params=params, X=X_prp, y=y_prp,
             max_n_features=max_n_features,
             scoring=SCORING,
             direction=DIRECTION,
             n_trials_select=N_TRIALS_SELECT,
             n_trials_tune=N_TRIALS_TUNE,
             timeout=TIMEOUT, n_jobs=-1,
             path_model_init=PATH_MODEL_INIT,
             path_model_params_init=PATH_MODEL_PARAMS_INIT,
             path_log_tuneup_init=PATH_LOG_TUNEUP_INIT,
             path_model=PATH_MODEL,
             path_model_params=PATH_MODEL_PARAMS,
             path_log_tuneup=PATH_LOG_TUNEUP,
             path_features_opt=PATH_FEATURES_OPT,
             path_log_opt_features=PATH_LOG_OPT_FEATURES)


if __name__ == '__main__':
    with utl.timer('Optimization'):
        main()
