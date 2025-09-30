from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.config import RANDOM_STATE

def get_pipeline():
    
    # Median Imputation
    imputer = SimpleImputer(strategy="median")
    
    # HistGradientBoostingClassifier
    hgbc = HistGradientBoostingClassifier(random_state=RANDOM_STATE)
    
    # Pipeline Assembly
    pipeline = Pipeline([
        ("imputer", imputer),
        ("clf", hgbc)
    ])
    
    return pipeline