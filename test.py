import config as c
from vmlkit import utility as utl

import joblib
import pandas as pd


def main():
    """
    1. Loading the model
    """
    # Mutable Paths
    path_trial_folder_ulid = utl.get_path_trial_folder_ulid(False)
    print('Test:', path_trial_folder_ulid)

    path_selected_features = path_trial_folder_ulid + c.SELECTED_FEATURES
    path_model = path_trial_folder_ulid + c.MODEL_FOR_TEST
    path_submit = path_trial_folder_ulid + c.SUBMIT

    # Loading
    model = joblib.load(path_model)
    train = joblib.load(c.PATH_TRAIN_PRP)

    y = train[c.TARGET_COLUMN]
    X = utl.except_for(train, c.TARGET_COLUMN)
    X_test = joblib.load(c.PATH_TEST_PRP)

    if c.USE_SELECTED_FEATURES:
        selected_features = list(pd.read_csv(path_selected_features))
        X = X[selected_features]
        X_test = X_test[selected_features]

    """
    2. Predict
    """
    model.fit(X, y)

    y_pred = model.predict_proba(X_test) if c.USE_PREDICT_PROBA\
        else model.predict(X_test)

    df_prediction = pd.DataFrame({
        c.TEST_ID: X_test[c.TEST_ID],
        c.TARGET_COLUMN: y_pred
    })
    print('Prediction\n', df_prediction.head())

    with open(path_submit, 'w', encoding='utf-8-sig') as f:
        df_prediction.to_csv(f, index=False)

# !kaggle competitions submit -c titanic -f ML/Kaggle/Titanic/trial_models/01DXQSYFD4W2NX064GHV38N53Z/submit.csv -m "LogisticRegression"


if __name__ == '__main__':
    with utl.timer('Test'):
        main()
