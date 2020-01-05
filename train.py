from config import (SCORING,
                    TARGET_COLUMN,
                    PATH_TRAIN_PRP,
                    PATH_TRIAL_FOLDER,
                    TEST_SIZE_RATIO,
                    DIRECTION,
                    N_TRIALS_TUNE,
                    TIMEOUT,
                    N_CV_SPLITS,
                    RANDOM_STATE,
                    IS_UNDER_SAMPLING)
from vmlkit import utility as utl
from vmlkit.model_selection.tuneupper import tuneup
from vmlkit import visualizer as viz

from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import ExtraTreesClassifier
# from sklearn.linear_model import RidgeClassifier
# from sklearn.neighbors import KNeighborsClassifier


def main():
    # Make a directory for training model
    if not utl.exists_dir(PATH_TRIAL_FOLDER):
        utl.mkdir(PATH_TRIAL_FOLDER)

    # Make a directory with ulid per each trial unit
    # ULID = ulid()
    ULID = '01DXTWADP25B83FR6NPP5ENNTP'
    PATH_TRIAL_FOLDER_ULID = PATH_TRIAL_FOLDER + ULID + '/'

    if not utl.exists_dir(PATH_TRIAL_FOLDER_ULID):
        utl.mkdir(PATH_TRIAL_FOLDER_ULID)

    # Mutable Paths
    PATH_FEATURES_OPT = PATH_TRIAL_FOLDER_ULID + 'optimized_features.csv'
    PATH_LOG_TUNEUP = PATH_TRIAL_FOLDER_ULID + 'log_tuneup.csv'
    PATH_MODEL = PATH_TRIAL_FOLDER_ULID + 'model.pkl'
    PATH_MODEL_PARAMS = PATH_TRIAL_FOLDER_ULID + 'model_params.txt'

    PATH_LOG_TUNEUP_US = PATH_TRIAL_FOLDER_ULID + 'log_tuneup_us.csv'
    PATH_MODEL_US = PATH_TRIAL_FOLDER_ULID + 'model_us.pkl'
    PATH_MODEL_PARAMS_US = PATH_TRIAL_FOLDER_ULID + 'model_params_us.txt'

    PATH_ROC_CURVE = PATH_TRIAL_FOLDER_ULID + 'roc_curve.png'
    PATH_ROC_CURVE_US = PATH_TRIAL_FOLDER_ULID + 'roc_curve_us.png'

    # Loading preprocessed data
    train_prp = utl.load(PATH_TRAIN_PRP)
    y_prp = train_prp[TARGET_COLUMN]
    X_prp = utl.except_for(train_prp, TARGET_COLUMN)

    selected_features = utl.load(PATH_FEATURES_OPT, return_list=True)
    X_prp_slc = X_prp[selected_features]

    # Case 1: Basic
    if not IS_UNDER_SAMPLING:
        # Tuning Parameters
        models = {
            'LogisticRegression': LogisticRegression,
            # 'Extra Trees': ExtraTreesClassifier,
            # 'Ridge': RidgeClassifier,
            # 'kneighbor': KNeighborsClassifier,
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
            # 'Extra Trees': {
            #     'n_estimators': ('int', 1, 100),
            #     'max_depth': ('dis', 1, 100, 5),
            #     'random_state': 128
            # },
            # 'Ridge': {
            #     'alpha': ('log', 1e-2, 1e2)
            # },
        }

        with utl.timer('tuneup'):

            best_model = tuneup(models=models, params=params,
                                X=X_prp_slc, y=y_prp,
                                scoring=SCORING,
                                direction=DIRECTION,
                                n_trials=N_TRIALS_TUNE,
                                timeout=TIMEOUT,
                                n_jobs=-1,
                                path_model=PATH_MODEL,
                                path_model_params=PATH_MODEL_PARAMS,
                                path_log_tuneup=PATH_LOG_TUNEUP)

    # Case 2: Under Sampling + Bagging
    if IS_UNDER_SAMPLING:
        base_model = utl.load(PATH_MODEL)

        models = {
            'BalancedBaggingClassifier': BalancedBaggingClassifier
        }

        params = {
            'BalancedBaggingClassifier': {
                'base_estimator': base_model,
                'n_estimators': ('int', 45, 60),
                'n_jobs': -1,
                'sampling_strategy': 'auto',
                'random_state': RANDOM_STATE,
            }
        }

        with utl.timer('tuneup with bagging under sampling'):

            best_model_un\
                = tuneup(models=models, params=params,
                         X=X_prp_slc, y=y_prp,
                         scoring=SCORING,
                         direction=DIRECTION,
                         n_trials=N_TRIALS_TUNE,
                         timeout=TIMEOUT,
                         n_jobs=-1,
                         path_model=PATH_MODEL_US,
                         path_model_params=PATH_MODEL_PARAMS_US,
                         path_log_tuneup=PATH_LOG_TUNEUP_US)

    """
    Plot ROC curve
    """
    if IS_UNDER_SAMPLING:
        path_roc_curve = PATH_ROC_CURVE_US
        mdl = best_model_un
    else:
        path_roc_curve = PATH_ROC_CURVE
        mdl = best_model

    with utl.timer('Plot ROC curve'):
        viz.plot_roc_curve_with_cv(
            mdl, X_prp_slc, y_prp,
            cv='StratifiedKFold', n_splits=N_CV_SPLITS,
            test_size_ratio=TEST_SIZE_RATIO,
            savepath=path_roc_curve)


if __name__ == '__main__':
    with utl.timer('Train'):
        main()
