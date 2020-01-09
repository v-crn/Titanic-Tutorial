import config as c
from vmlkit import utility as utl
from vmlkit.model_selection.optimizer import optimize
from vmlkit import visualizer as viz
from vmlkit.modeler import ModelFactory

import joblib
import pandas as pd


def main():
    path_trial_folder_ulid = utl.get_path_trial_folder_ulid(c.CREATE_NEW)

    print('Trial:', path_trial_folder_ulid)

    # Mutable Paths
    path_optimized_features = path_trial_folder_ulid + c.OPTIMIZED_FEATURES
    path_log_optimized_features\
        = path_trial_folder_ulid + c.LOG_OPT_FEATURES
    path_log_tuneup = path_trial_folder_ulid + 'log_opt_tuneup.csv'
    path_log_tuneup_feature_select\
        = path_trial_folder_ulid + 'log_opt_tuneup_feature_select.csv'
    path_model = path_trial_folder_ulid + 'model_opt.joblib'
    path_model_feature_select\
        = path_trial_folder_ulid + 'model_opt_feature_select.joblib'
    path_model_params = path_trial_folder_ulid + 'optimized_model_params.txt'
    path_model_params_feature_select\
        = path_trial_folder_ulid + 'model_opt_params_feature_select.txt'
    path_roc_curve = path_trial_folder_ulid + 'roc_curve_opt.png'

    # Loading preprocessed data
    train_prp = joblib.load(c.PATH_TRAIN_PRP)
    y_prp = train_prp[c.TARGET_COLUMN]
    X_prp = utl.except_for(train_prp, c.TARGET_COLUMN)

    max_n_features = c.RATIO_MAX_N_FEATURES * X_prp.shape[1]

    model = ModelFactory(name=c.MODEL_NAME,
                         params=c.MODEL_PARAMS).model

    # Optimization
    model_optimized\
        = optimize(
            models=c.MODEL_LIST,
            params=c.MODEL_PARAMS,
            X=X_prp, y=y_prp,
            model_feature_select=model,
            max_n_features=max_n_features,
            scoring=c.SCORING,
            direction=c.DIRECTION,
            cv=c.CV,
            n_splits=c.N_CV_SPLITS,
            n_trials_select=c.N_TRIALS_SELECT,
            n_trials_tune=c.N_TRIALS_TUNE,
            timeout=c.TIMEOUT,
            n_jobs=-1,
            path_model_feature_select=path_model_feature_select,
            path_model_params_feature_select=path_model_params_feature_select,
            path_log_tuneup_feature_select=path_log_tuneup_feature_select,
            path_model=path_model,
            path_model_params=path_model_params,
            path_log_tuneup=path_log_tuneup,
            path_features_opt=path_optimized_features,
            path_log_opt_features=path_log_optimized_features)

    """
    Plot ROC curve
    """
    selected_features = list(pd.read_csv(path_optimized_features))
    X_prp_slc = X_prp[selected_features]

    with utl.timer('Plot ROC curve'):
        viz.plot_roc_curve_with_cv(
            model_optimized, X_prp_slc, y_prp,
            cv='StratifiedKFold', n_splits=c.N_CV_SPLITS,
            test_size_ratio=c.TEST_SIZE_RATIO,
            savepath=path_roc_curve)


if __name__ == '__main__':
    with utl.timer('Optimization'):
        main()
