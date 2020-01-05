import numpy as np
from imblearn.under_sampling import RandomUnderSampler


def under_sample(X, y, minority_ratio='auto', random_state=None):
    """
    minority_ratio = minority / majority

        float is only available for binary classification.
        An error is raised for multi-class classification.
    """
    if X.ndim < 2:
        # RandomUnderSampler.fit_resample() leads to error if X is 1d-array
        X = np.vstack(X)

    rus = RandomUnderSampler(
        sampling_strategy=minority_ratio, random_state=random_state)
    X_resampled, y_resampled = rus.fit_sample(X, y)
    return X_resampled, y_resampled
