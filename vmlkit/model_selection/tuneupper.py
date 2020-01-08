from vmlkit import validator

import joblib
import codecs
import optuna
from sklearn.model_selection import StratifiedKFold, cross_validate


def tuneup(models, params,
           X, y, direction='maximize',
           scoring='roc_auc', n_trials=20,
           timeout=None,
           cv=None, n_splits=5, random_state=42, n_repeats=10,
           n_jobs=-1,
           path_model=None,
           path_model_params=None,
           path_log_tuneup=None):

    objective = Objective(
        models=models, params=params, X=X, y=y,
        direction=direction, scoring=scoring,
        cv=cv, n_splits=n_splits, random_state=random_state,
        n_repeats=n_repeats)

    study = optuna.create_study(
        direction=direction,
        sampler=optuna.samplers.RandomSampler(seed=random_state),
        pruner=optuna.pruners.MedianPruner())

    study.optimize(objective, n_trials=n_trials,
                   n_jobs=n_jobs, timeout=timeout)

    best_model = objective.best_model
    print('best model:', best_model)

    if path_model:
        joblib.dump(best_model, path_model, compress=3)

    best_model_params = best_model.get_params()
    if path_model_params:
        print(best_model_params, file=codecs.open(
            path_model_params, 'w', 'utf-8'))

    best_score = study.best_value
    print('best score:', best_score)

    study_log = study.trials_dataframe()

    if path_log_tuneup:
        study_log.to_csv(path_log_tuneup)

    optuna.visualization.plot_optimization_history(study)
    optuna.visualization.plot_slice(study)
    # optuna.visualization.plot_contour(study, params=['param1', 'param2'])

    return best_model


class Objective:
    def __init__(self, models, params, X, y, scoring,
                 direction='maximize', test_size=0.2,
                 random_state=None,
                 cv=None, n_splits=5, n_repeats=10):
        self.models = models
        self.params = params
        self.X = X
        self.y = y
        self.scoring = scoring
        self.direction = direction
        self.test_size = test_size
        self.random_state = random_state

        if cv is None:
            cv = StratifiedKFold(
                n_splits=n_splits, random_state=random_state, shuffle=True)

        if type(cv) is str:
            cv = validator.get_cv(cv, n_splits=n_splits,
                                  test_size_ratio=test_size,
                                  n_repeats=n_repeats,
                                  random_state=random_state,
                                  shuffle=True)
        self.cv = cv

        self.best_model = None
        self.best_score = 0 if direction == 'maximize' else 1

        """
        params:
            - int: integer. ex: ('int', 最小値, 最大値)
            - uni: a uniform float sampling. ex: ('uni', 最小値, 最大値)
            - log: a uniform float sampling on log scale. ex: ('log', 最小値, 最大値)
            - dis: a discretized uniform float sampling. ex: ('dis', 最小値, 最大値, 間隔)
            - cat: category. ex: ('cat', (文字列１, 文字列２, 文字列３, ))

        if you set constant value, write as follows:
            - 'parameter name': value
        """
        self.method_names = {
            'int': 'suggest_int',
            'uni': 'suggest_uniform',
            'log': 'suggest_loguniform',
            'dis': 'suggest_discrete_uniform',
            'cat': 'suggest_categorical'
        }
        self.model_names = list(models)
        self.model_params = {
            model_name: {
                key: (self.method_names.get(val[0]),
                      ('{}_{}'.format(model_name, key), *val[1:])
                      ) if type(val) is tuple else val
                for key, val in self.params.get(model_name).items()}
            for model_name in self.model_names
        }

    def __call__(self, trial):
        X = self.X
        y = self.y
        direction = self.direction

        best_score = self.best_score

        model_name = trial.suggest_categorical('model', self.model_names)
        params = {
            key: getattr(trial, val[0])(*val[1]) if type(val) is tuple else val
            for key, val in self.model_params.get(model_name).items()
        }
        model = self.models.get(model_name)(**params)

        cv_result = cross_validate(model, X, y,
                                   scoring=self.scoring,
                                   cv=self.cv,
                                   return_estimator=False)

        score = cv_result['test_score'].mean()

        # Log score
        if ((score > best_score) and (direction == 'maximize'))\
                or ((score < best_score) and (direction != 'maximize')):
            self.best_score = score
            self.best_model = model

        return score
