from config import (PATH_TRAIN_PRP,
                    PATH_TEST_PRP,
                    TARGET_COLUMN,
                    TEST_ID,
                    PATH_TRIAL_FOLDER)
from vmlkit import utility as utl
import pandas as pd


"""
3. Test
"""


def main():
    """
    1. Loading the model
    """
    # ULID Path
    ULID = '01DXRFYZCKPE15V12VKPDG3TCD'

    # Mutable Paths
    PATH_TRIAL_FOLDER_ULID = PATH_TRIAL_FOLDER + ULID + '/'
    PATH_FEATURES_OPT = PATH_TRIAL_FOLDER_ULID + 'optimized_features.csv'
    PATH_MODEL = PATH_TRIAL_FOLDER_ULID + 'model.pkl'
    PATH_SUBMIT = PATH_TRIAL_FOLDER_ULID + 'submit.csv'

    # Loading
    model = utl.load(PATH_MODEL)

    train_prp = utl.load(PATH_TRAIN_PRP)
    y_prp = train_prp[TARGET_COLUMN]
    X_prp = utl.except_for(train_prp, TARGET_COLUMN)
    selected_features = utl.load(PATH_FEATURES_OPT, return_list=True)
    X_prp_slc = X_prp[selected_features]

    X_test_prp = utl.load(PATH_TEST_PRP)
    X_test_prp_slc = X_test_prp[selected_features]

    """
    2. Predict
    """
    model.fit(X_prp_slc, y_prp)

    y_pred = model.predict(X_test_prp_slc)

    df_prediction = pd.DataFrame({
        TEST_ID: X_test_prp[TEST_ID],
        TARGET_COLUMN: y_pred
    })

    with open(PATH_SUBMIT, 'w', encoding='utf-8-sig') as f:
        df_prediction.to_csv(f, index=False)

# !kaggle competitions submit -c titanic -f ML/Kaggle/Titanic/trial_models/01DXQSYFD4W2NX064GHV38N53Z/submit.csv -m "LogisticRegression"


if __name__ == '__main__':
    with utl.timer('Test'):
        main()
