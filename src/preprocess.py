import pandas as pd
from sklearn.impute import SimpleImputer

def load_and_preprocess(file_path):
    df = pd.read_csv(file_path)
    
    # Definição do alvo
    TARGET_COL = "Potability"
    y = df[TARGET_COL].astype(int)
    X = df.drop(columns=[TARGET_COL])
    
    # Tratamento de Ausentes por Mediana
    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X),
        columns=X.columns,
        index=X.index
    )
    
    return X_imputed, y