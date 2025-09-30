from scipy.stats import uniform, randint

# constants

RANDOM_STATE = 42 # Seed fixa
K_FOLDS = 5 # Stratified K-Fold
TARGET_COL = "Potability"
ARTIFACTS_DIR = 'artifacts/'
REPORTS_DIR = 'reports/'
FIGURES_DIR = 'figures/'
DATA_PATH = 'data/raw/water.csv'

# HistGradientBoosting Hyperparameter Grid for RandomizedSearchCV
HGBC_PARAM_DIST = {
    'clf__max_iter': randint(100, 300),
    'clf__learning_rate': uniform(0.01, 0.2), 
    'clf__max_leaf_nodes': randint(10, 50),
    'clf__l2_regularization': uniform(0.0, 1.0)
}