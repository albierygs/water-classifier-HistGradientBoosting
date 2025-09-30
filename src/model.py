from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.config import RANDOM_STATE

def get_pipeline():
    """
    Returns the HistGradientBoostingClassifier pipeline, including
    the required median imputation preprocessing step.
    """
    # 1. Imputação por Mediana (Protocolo Mínimo)
    imputer = SimpleImputer(strategy="median")
    
    # 2. HistGradientBoostingClassifier (Modelo escolhido)
    hgbc = HistGradientBoostingClassifier(random_state=RANDOM_STATE)
    
    # 3. Montagem do Pipeline
    pipeline = Pipeline([
        ("imputer", imputer),
        ("clf", hgbc) # clf stands for classifier
    ])
    
    return pipeline