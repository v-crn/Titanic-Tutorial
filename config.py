# import xgboost as xgb
# from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import ExtraTreesClassifier
# from sklearn.linear_model import RidgeClassifier
# from sklearn.neighbors import KNeighborsClassifier

"""
Pahts
"""
PATH_ROOT = './ML/Kaggle/Titanic/'
PATH_DATA_FOLDER = PATH_ROOT + 'data/'
PATH_TRIAL_FOLDER = PATH_ROOT + 'trials/'
PATH_LOG_FOLDER = PATH_ROOT + 'logs/'
PATH_CACHE_FOLDER = PATH_ROOT + '.chache/'
PATH_ULID = PATH_CACHE_FOLDER + 'ulid.txt'

PATH_TRAIN = PATH_DATA_FOLDER + 'train.csv'
PATH_TEST = PATH_DATA_FOLDER + 'test.csv'


"""
Important
"""
# Target
TARGET_COLUMN = 'Survived'

# Trial folder
CREATE_NEW = True
USE_SELECTED_FEATURES = False

# Feature Selection
RATIO_MAX_N_FEATURES = 0.8
N_TRIALS_SELECT = 5

# Optimization
N_TRIALS_TUNE = 5
TIMEOUT = None
STUDY_NAME_OPT_FIEATURES = None
PATH_OPTUNA_STORAGE_OPT_FIEATURES = None

# CV
CV = 'StratifiedKFold'
N_CV_SPLITS = 5
TEST_SIZE_RATIO = 0.2
SEED = 42

# Test
USE_PREDICT_PROBA = False
TRIAL_FOLDER_NAME_FOR_TEST = None
PATH_TRIAL_FOLDER_FOR_TEST\
    = PATH_TRIAL_FOLDER + TRIAL_FOLDER_NAME_FOR_TEST + '/'
MODEL_FILE_FOR_TEST = 'model.joblib'
SELECTED_FEATURES_FILE = 'selected_features.csv'
SUBMIT_FILE = 'submit.csv'

"""
Preprocess
"""
THRESH_NAN_RATIO_PER_COL = 0.5
THRESH_CORR = 0.95
ALT_NUM = 'mean'
ALT_CAT = 'mode'
CAT_ENCODER = 'one-hot'

# 【Category Encoderについて】
# 多クラス分類においてカテゴリー変数の数値変換の手法には様々なものがある。
# 決定木系モデルでは'ordinal' encoder を使うのが一般的。
# 線形モデルでは順序が影響しないように one - hot encoder が使われることが多い。

PATH_TRAIN_PRP = PATH_DATA_FOLDER + 'train_ppr_' + CAT_ENCODER + '.joblib'
PATH_TEST_PRP = PATH_DATA_FOLDER + 'test_ppr_' + CAT_ENCODER + '.joblib'
PATH_PROFILE_REPORT_TRAIN = PATH_DATA_FOLDER + 'profile_report_train.html'
PATH_PROFILE_REPORT_TRAIN_PRP\
    = PATH_DATA_FOLDER + 'profile_report_train_prp_' + CAT_ENCODER + '.html'

PATH_PROFILE_REPORT_TEST = PATH_DATA_FOLDER + 'profile_report_test.html'
PATH_PROFILE_REPORT_TEST_PRP\
    = PATH_DATA_FOLDER + 'profile_report_test_prp_' + CAT_ENCODER + '.html'


"""
Model config
"""
MODEL_NAME = 'LogisticRegression'
MODEL = '{0}.joblib'.format(MODEL_NAME)

MODEL_LIST = {
    # 'xgb': xgb.XGBClassifier,
    'LogisticRegression': LogisticRegression
    # 'BalancedBaggingClassifier': BalancedBaggingClassifier
}

# Under Sampling
IS_UNDER_SAMPLING = False

MODEL_PARAMS = {
    # 'xgb': {
    #     'objective': 'binary:logistic',
    #     'eval_metric': 'auc',
    # },
    'LogisticRegression': {
        'C': 1.0,
        'random_state': SEED,
        'max_iter': 100,
        'penalty': 'l1',
        'n_jobs': -1,
        'solver': 'liblinear',
        'class_weight': 'balanced',
        # 'class_weight': {0: 1, 1: 2},
    },
    # 'RandomForest': {
    #     'max_depth': 8,
    #     'min_sample_split': 2,
    #     'n_estimator': 200,
    #     'random_state': SEED,
    #     'class_weight': {0:1, 1:2},
    # }
    # 'BalancedBaggingClassifier': {
    #     'base_estimator': base_model,
    #     'n_estimators': 10,
    #     'n_jobs': -1,
    #     'sampling_strategy': 'auto',
    #     'random_state': SEED,
    # }
}


"""
Evaluation
"""
# Scoring
SCORING = 'roc_auc'
DIRECTION = 'maximize'


"""
Features
"""
# 使わないことがわかっている特徴量
DISCARDED_FEATURES = ['Name']

# 提出するデータのID（予測値と対になる列）など変換を行わない特徴量
TEST_ID = 'PassengerId'
EXCLUSIVE_FEATURES = [TEST_ID, TARGET_COLUMN]
