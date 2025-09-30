import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, 
    precision_recall_curve, auc
)

def calculate_all_metrics(y_true, y_pred, y_proba):
    
    # PR-AUC (Area Under Precision-Recall Curve)
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall_vals, precision_vals)
    
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'F1_Macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'ROC_AUC': roc_auc_score(y_true, y_proba),
        'Precision_Macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'Recall_Macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'PR_AUC': pr_auc
    }
    return metrics

def save_artifact(obj, filename, directory):
    import pickle
    import os
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, filename), 'wb') as f:
        pickle.dump(obj, f)
