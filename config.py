"""
Constants
"""
# Data Setting
"""
Category Encoder

多クラス分類においてカテゴリー変数の数値変換の手法には様々なものがある。
決定木系モデルでは'ordinal' encoder を使うのが一般的。
線形モデルでは順序が影響しないように one-hot encoder が使われることが多い。
"""
CAT_ENCODER = 'ordinal'

TARGET_COLUMN = 'Survived'

SCORING = 'roc_auc'
DIRECTION = 'maximize'

# 前処理の前に使わないことがわかっている特徴量
DISCARDED_FEATURES = ['Name']

# 提出するデータのID（予測値と対になる列）など変換を行わない特徴量
TEST_ID = 'PassengerId'
EXCLUSIVE_FEATURES = [TEST_ID, TARGET_COLUMN]

# Under Sampling
IS_UNDER_SAMPLING = True

# Feature Selection
RATIO_MAX_N_FEATURES = 1.0
N_TRIALS_SELECT = 20

# Optimization
N_TRIALS_TUNE = 30
TIMEOUT = None

# CV
N_CV_SPLITS = 5
TEST_SIZE_RATIO = 0.2
RANDOM_STATE = 42

# Path
PATH_ROOT = './ML/Kaggle/Titanic/'
PATH_DATA_FOLDER = PATH_ROOT + 'data/'
PATH_ULID = PATH_ROOT + 'ulid.txt'

PATH_TRAIN = PATH_DATA_FOLDER + 'train.csv'
PATH_TEST = PATH_DATA_FOLDER + 'test.csv'

PATH_TRAIN_PRP = PATH_DATA_FOLDER + 'train_ppr_' + CAT_ENCODER + '.joblib'
# PATH_TRAIN_PRP = PATH_DATA_FOLDER + 'train_prp.joblib'
PATH_TEST_PRP = PATH_DATA_FOLDER + 'test_ppr_' + CAT_ENCODER + '.joblib'

PATH_PROFILE_REPORT = PATH_DATA_FOLDER + 'profile_report.html'
PATH_PROFILE_REPORT_PRP\
    = PATH_DATA_FOLDER + 'profile_report_prp_' + CAT_ENCODER + '.html'

PATH_TRIAL_FOLDER = PATH_ROOT + 'trials/'
