from sklearn.model_selection import (StratifiedKFold,
                                     ShuffleSplit,
                                     TimeSeriesSplit,
                                     GroupShuffleSplit,
                                     GroupKFold,
                                     RepeatedKFold,
                                     RepeatedStratifiedKFold)


def get_cv(cv_name, n_splits=5, test_size_ratio=0.2,
           n_repeats=10, random_state=42, shuffle=True):

    cv = None

    if type(cv_name) is str:
        if cv_name == 'ShuffleSplit':
            cv = ShuffleSplit(n_splits=n_splits,
                              test_size=test_size_ratio,
                              train_size=None,
                              random_state=random_state)
        elif cv_name == 'TimeSeriesSplit':
            cv = TimeSeriesSplit(n_splits=n_splits, max_train_size=None)
        elif cv_name == 'GroupShuffleSplit':
            cv = GroupShuffleSplit(n_splits=n_splits,
                                   test_size=test_size_ratio,
                                   train_size=None,
                                   random_state=random_state)
        elif cv_name == 'GroupKFold':
            cv = GroupKFold(n_splits=n_splits)
        elif cv_name == 'RepeatedKFold':
            cv = RepeatedKFold(n_splits=n_splits,
                               n_repeats=n_repeats,
                               random_state=random_state)
        elif cv_name == 'RepeatedStratifiedKFold':
            cv = RepeatedStratifiedKFold(
                n_splits=n_splits,
                n_repeats=n_repeats,
                random_state=random_state)
        else:
            cv = StratifiedKFold(
                n_splits=n_splits, random_state=random_state, shuffle=shuffle)

    return cv
