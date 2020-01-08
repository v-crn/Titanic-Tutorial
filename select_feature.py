import config as c
from vmlkit import utility as utl
from vmlkit.modeler import ModelFactory
from vmlkit.model_selection.feature_selector import select_features_by_rfe

import joblib
import codecs
from ulid import ulid


def main():
    # Make a directory for training model
    if not utl.exists_dir(c.PATH_TRIAL_FOLDER):
        utl.mkdir(c.PATH_TRIAL_FOLDER)

    # Make a directory per each trial unit with ulid
    if c.CREATE_NEW:
        trial_id = ulid()

        if not utl.exists_dir(c.PATH_CACHE_FOLDER):
            utl.mkdir(c.PATH_CACHE_FOLDER)

        print(trial_id, file=codecs.open(c.PATH_ULID, 'w', 'utf-8'))
    else:
        for line in open(c.PATH_ULID):
            trial_id = line.replace('\n', '')

    print('trial_id:', trial_id)

    path_trial_folder_ulid\
        = c.PATH_TRIAL_FOLDER + trial_id + '_' + c.MODEL_NAME + '/'

    if not utl.exists_dir(path_trial_folder_ulid):
        utl.mkdir(path_trial_folder_ulid)

    # Mutable Paths
    path_selected_features = path_trial_folder_ulid + 'selected_features.csv'

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
            path_selected_features=path_selected_features)

    print(selected_features)


if __name__ == '__main__':
    with utl.timer('Select feature'):
        main()
