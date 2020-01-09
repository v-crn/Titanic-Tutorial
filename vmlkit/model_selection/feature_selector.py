from vmlkit import utility as utl
from vmlkit import validator

import csv
import optuna
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold


def select_features_by_rfe(model, X, y, ratio_max_n_features=0.5,
                           path_selected_features=None):
    n_features = X.shape[1]
    n_features_to_select = int(n_features * ratio_max_n_features)
    rfe = RFE(model, n_features_to_select=n_features_to_select)
    rfe.fit(X, y)

    X_selected = X.drop(
        X.columns[np.where(rfe.support_ == False)[0]], axis=1)
    selected_features = utl.get_columns(X_selected)

    if path_selected_features:
        with open(path_selected_features, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(selected_features)

    return selected_features


def optimize_features(model, X, y, max_n_features=None, n_trials=20,
                      direction='maximize', scoring='roc_auc',
                      n_jobs=-1, timeout=None,
                      cv=None, n_splits=5, random_state=42, n_repeats=10,
                      path_features_opt=None,
                      path_log_opt_features=None,
                      path_study_name_opt_features=None,
                      path_optuna_storage_opt_features=None):
    objective = Objective(model=model, X=X, y=y,
                          max_n_features=max_n_features,
                          direction=direction, scoring=scoring,
                          cv=cv, n_splits=n_splits,
                          random_state=random_state, n_repeats=n_repeats,
                          path_features_opt=path_features_opt)

    study = optuna.create_study(
        direction=direction,
        sampler=optuna.samplers.RandomSampler(seed=random_state),
        pruner=optuna.pruners.MedianPruner(),
        study_name=path_study_name_opt_features,
        storage=path_optuna_storage_opt_features,
        load_if_exists=True)
    study.optimize(objective, n_trials=n_trials,
                   n_jobs=n_jobs, timeout=timeout)
    best_features = objective.best_features
    print('best features:', best_features)

    if path_features_opt:
        with open(path_features_opt, 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(best_features)

    best_score = study.best_value
    print('best score:', best_score)

    study_log = study.trials_dataframe()

    if path_log_opt_features:
        study_log.to_csv(path_log_opt_features)

    optuna.visualization.plot_optimization_history(study)
    optuna.visualization.plot_slice(study)
    # optuna.visualization.plot_contour(study, params=['param1', 'param2'])

    return best_features


class Objective():
    def __init__(self, model, X, y, max_n_features=None,
                 direction='maximize', scoring='roc_auc',
                 cv=None, n_splits=5, random_state=42, n_repeats=10,
                 path_features_opt=None):
        self.X = X
        self.y = y
        self.model = model
        self.direction = direction
        self.scoring = scoring
        self.best_n_features = None
        self.best_features = None
        self.best_score = 0 if direction == 'maximize' else 1
        self.path_features_opt = path_features_opt

        if cv is None:
            cv = StratifiedKFold(
                n_splits=5, random_state=42, shuffle=True)

        if type(cv) is str:
            cv = validator.get_cv(cv, n_splits=n_splits, test_size_ratio=0.2,
                                  n_repeats=n_repeats,
                                  random_state=random_state, shuffle=True)

        self.cv = cv

        if max_n_features is None:
            max_n_features = X.shape[1]
        self.max_n_features = max_n_features

    def __call__(self, trial):
        model = self.model
        X = self.X
        y = self.y
        direction = self.direction
        best_score = self.best_score

        # Trial parameter
        n_features_to_select = trial.suggest_int(
            'n_features_to_select', 1, self.max_n_features),

        # Method of feature selection
        rfe = RFE(estimator=model,
                  n_features_to_select=n_features_to_select)
        rfe.fit(X, y)

        # Selected features
        X_selected = X.drop(
            X.columns[np.where(rfe.support_ == False)[0]], axis=1)
        selected_features = utl.get_columns(X_selected)

        # Validation
        cv_result = cross_validate(model, X_selected, y,
                                   scoring=self.scoring,
                                   cv=self.cv, return_estimator=False)
        score = cv_result['test_score'].mean()

        # Score Log
        trial.set_user_attr('score', score)

        # Save selected features
        if ((score > best_score) and (direction == 'maximize'))\
                or ((score < best_score) and (direction != 'maximize')):
            self.best_score = score
            self.best_features = selected_features

        return score
