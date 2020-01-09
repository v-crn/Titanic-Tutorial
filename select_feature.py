import config as c
from vmlkit import utility as utl
from vmlkit.modeler import ModelFactory
from vmlkit.model_selection.feature_selector import select_features_by_rfe

import joblib


def main():
    path_trial_folder_ulid = utl.get_path_trial_folder_ulid(c.CREATE_NEW)
    print('Trial:', path_trial_folder_ulid)

    # Mutable Paths
    path_selected_features = path_trial_folder_ulid + c.SELECTED_FEATURES

    # Loading preprocessed data
    train = joblib.load(c.PATH_TRAIN_PRP)
    y = train[c.TARGET_COLUMN]
    X = utl.except_for(train, c.TARGET_COLUMN)

    if utl.exists_dir(c.PATH_LOG_FOLDER):
        utl.mkdir(c.PATH_LOG_FOLDER)

    model = ModelFactory(name=c.MODEL_NAME,
                         params=c.MODEL_PARAMS).model

    selected_features\
        = select_features_by_rfe(
            model=model, X=X, y=y,
            ratio_max_n_features=c.RATIO_MAX_N_FEATURES,
            path_selected_features=path_selected_features,
            path_study_name_opt_features=c.STUDY_NAME_OPT_FIEATURES,
            path_optuna_storage_opt_features=c.PATH_OPTUNA_STORAGE_OPT_FIEATURES)

    print(selected_features)


if __name__ == '__main__':
    with utl.timer('Select feature'):
        main()
