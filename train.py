import config as c
from vmlkit import utility as utl
from vmlkit.model_selection.tuneupper import tuneup
from vmlkit import visualizer as viz

import joblib
import codecs
from ulid import ulid
import pandas as pd


def main():
    # Make a directory for training model
    if not utl.exists_dir(c.PATH_TRIAL_FOLDER):
        utl.mkdir(c.PATH_TRIAL_FOLDER)

    if c.CREATE_NEW:
        trial_id = ulid()
        path_trial_folder_ulid = c.MODEL_NAME + trial_id
        utl.mkdir(path_trial_folder_ulid)
        print(trial_id, file=codecs.open(c.PATH_ULID, 'w', 'utf-8'))
        print('New tirial folder created!')
    else:
        for line in open(c.PATH_ULID):
            trial_id = line.replace('\n', '')

    # trial_id Path
    # if c.CREATE_NEW:
    #     trial_id = ulid()
    #     path_trial_folder = c.MODEL_NAME + trial_id
    #     print(trial_id, file=codecs.open(path_trial_folder, 'w', 'utf-8'))
    # else:
    #     for line in open(trial_id):
    #         trial_id = line.replace('\n', '')

    path_trial_folder_ulid\
        = c.PATH_TRIAL_FOLDER + trial_id + '_' + c.MODEL_NAME + '/'

    print('Trial:', path_trial_folder_ulid)

    if not utl.exists_dir(path_trial_folder_ulid):
        utl.mkdir(path_trial_folder_ulid)

    # Mutable Paths
    path_selected_features = path_trial_folder_ulid + 'selected_features.csv'
    path_log_tuneup = path_trial_folder_ulid + 'log_tuneup.csv'
    path_model = path_trial_folder_ulid + 'model.joblib'
    path_model_params = path_trial_folder_ulid + 'model_params.txt'
    path_roc_curve = path_trial_folder_ulid + 'roc_curve.png'

    # Loading preprocessed data
    train = joblib.load(c.PATH_TRAIN_PRP)
    y = train[c.TARGET_COLUMN]
    X = utl.except_for(train, c.TARGET_COLUMN)

    if c.USE_SELECTED_FEATURES:
        if not utl.exists(path_selected_features):
            raise Exception("The file path doesn't exist.")

        selected_features = list(pd.read_csv(path_selected_features))
        X = X[selected_features]

    with utl.timer('tuneup'):

        best_model = tuneup(models=c.MODEL_LIST,
                            params=c.MODEL_PARAMS,
                            X=X, y=y,
                            scoring=c.SCORING,
                            direction=c.DIRECTION,
                            cv=c.CV,
                            n_splits=c.N_CV_SPLITS,
                            random_state=c.SEED,
                            n_trials=c.N_TRIALS_TUNE,
                            timeout=c.TIMEOUT,
                            n_jobs=-1,
                            path_model=path_model,
                            path_model_params=path_model_params,
                            path_log_tuneup=path_log_tuneup)

    """
    Plot ROC curve
    """
    with utl.timer('Plot ROC curve'):
        viz.plot_roc_curve_with_cv(
            best_model, X, y,
            cv=c.CV, n_splits=c.N_CV_SPLITS,
            test_size_ratio=c.TEST_SIZE_RATIO,
            savepath=path_roc_curve)


if __name__ == '__main__':
    with utl.timer('Train'):
        main()
