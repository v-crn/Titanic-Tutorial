from vmlkit.model_selection.feature_selector import optimize_features
from vmlkit.model_selection.tuneupper import tuneup


"""
Optimization
"""


def optimize(models, params, X, y,
             max_n_features=None,
             scoring='roc_auc',
             direction='maximize',
             n_trials_select=20,
             n_trials_tune=20,
             timeout=None, n_jobs=-1,
             path_model_init=None,
             path_model_params_init=None,
             path_log_tuneup_init=None,
             path_model=None,
             path_model_params=None,
             path_log_tuneup=None,
             path_features_opt=None,
             path_log_opt_features=None):
    """
    1. Initial tuneup
    """
    best_model_init = tuneup(models=models, params=params,
                             X=X, y=y,
                             direction=direction,
                             scoring=scoring,
                             n_trials=n_trials_tune,
                             timeout=timeout,
                             n_jobs=n_jobs,
                             path_model=path_model_init,
                             path_model_params=path_model_params_init,
                             path_log_tuneup=path_log_tuneup_init)

    """
    2. Optimize features number
    """
    selected_features = optimize_features(
        model=best_model_init, X=X, y=y,
        max_n_features=max_n_features,
        n_trials=n_trials_select,
        direction=direction, scoring=scoring,
        timeout=timeout,
        path_features_opt=path_features_opt,
        path_log_opt_features=path_log_opt_features)

    X_slc = X[selected_features]

    """
    3. Re-tuneup
    """
    best_model = tuneup(models=models, params=params,
                        X=X_slc, y=y,
                        direction=direction,
                        scoring=scoring,
                        n_trials=n_trials_tune,
                        timeout=timeout,
                        n_jobs=n_jobs,
                        path_model=path_model,
                        path_model_params=path_model_params,
                        path_log_tuneup=path_log_tuneup)

    return best_model
