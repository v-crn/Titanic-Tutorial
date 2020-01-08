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
    path_selected_features = path_trial_folder_ulid + 'selected_features.csv'
    path_model = path_trial_folder_ulid + 'model_opt.joblib'
    path_submit = path_trial_folder_ulid + 'submit.csv'

    # Loading
    model = joblib.load(path_model)

    train_prp = joblib.load(c.PATH_TRAIN_PRP)
    y_prp = train_prp[c.TARGET_COLUMN]
    X_prp = utl.except_for(train_prp, c.TARGET_COLUMN)
    selected_features = list(pd.read_csv(path_selected_features))

    X_prp_slc = X_prp[selected_features]

    X_test_prp = joblib.load(c.PATH_TEST_PRP)
    X_test_prp_slc = X_test_prp[selected_features]

    """
    2. Predict
    """
    model.fit(X_prp_slc, y_prp)

    y_pred = model.predict(X_test_prp_slc)

    df_prediction = pd.DataFrame({
        c.TEST_ID: X_test_prp[c.TEST_ID],
        c.TARGET_COLUMN: y_pred
    })
    print('Prediction\n', df_prediction.head())

    with open(path_submit, 'w', encoding='utf-8-sig') as f:
        df_prediction.to_csv(f, index=False)

# !kaggle competitions submit -c titanic -f ML/Kaggle/Titanic/trial_models/01DXQSYFD4W2NX064GHV38N53Z/submit.csv -m "LogisticRegression"


if __name__ == '__main__':
    with utl.timer('Test'):
        main()
