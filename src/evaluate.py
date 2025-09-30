import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, precision_recall_curve
)
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
import pickle
import os
import argparse
import sys

from src.config import DATA_PATH, FIGURES_DIR, RANDOM_STATE, REPORTS_DIR
from src.utils import calculate_all_metrics

# Import preprocessing
from src.preprocess import load_and_preprocess

def parse_args():
    """Handles command-line arguments for CLI execution."""
    parser = argparse.ArgumentParser(description="Evaluate the trained model and generate figures.")
    parser.add_argument('--model', type=str, default='artifacts/best.pkl', help='Path to the trained model artifact.')
    parser.add_argument('--data', type=str, default=DATA_PATH, help='Path to the water.csv dataset.')
    parser.add_argument('--out', type=str, default=FIGURES_DIR, help='Output directory for figures.')
    return parser.parse_args()

def plot_and_save_figures(model, X_test, y_test, y_proba, metrics, feature_names, fig_dir):
    """Generates and saves the required figures."""
    os.makedirs(fig_dir, exist_ok=True)
    
    # 1. Confusion Matrix
    y_pred = (y_proba > 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    plt.figure(figsize=(6, 6))
    disp.plot(values_format='d', ax=plt.gca())
    plt.title('Confusion Matrix - Best HGBC Model')
    plt.savefig(f'{fig_dir}/confusion_matrix.png')
    plt.close()
    print("✅ Confusion Matrix saved.")

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {metrics["ROC_AUC"]:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig(f'{fig_dir}/roc_curve.png')
    plt.close()
    print("✅ ROC Curve saved.")

    # 3. Precision-Recall Curve
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)
    plt.figure(figsize=(6, 6))
    plt.plot(recall_vals, precision_vals, label=f'PR curve (AUC = {metrics["PR_AUC"]:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.savefig(f'{fig_dir}/pr_curve.png')
    plt.close()
    print("✅ Precision-Recall Curve saved.")
    
    # 4. Feature Importance (Permutation)
    # Use the imputer from the pipeline on X_test to ensure no NaNs for permutation importance
    X_test_imputed = pd.DataFrame(
        model['imputer'].transform(X_test), 
        columns=X_test.columns,
        index=X_test.index
    )
    
    # Calculate Permutation Importance
    r = permutation_importance(
        model['clf'], X_test_imputed, y_test, 
        n_repeats=30, random_state=RANDOM_STATE, n_jobs=-1, scoring='roc_auc'
    )
    
    feature_importances = pd.Series(
        r.importances_mean, 
        index=feature_names
    ).sort_values(ascending=False)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    feature_importances.plot.bar()
    plt.title('Permutation Feature Importance (ROC-AUC)')
    plt.ylabel('Importance (Mean decrease in ROC-AUC)')
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/permutation_importance.png')
    plt.close()
    print("✅ Permutation Importance saved.")


def load_and_combine_results(test_metrics, cv_results_path):
    """
    Loads CV results and combines them with Test metrics into a single DataFrame
    where metrics are the index.
    """
    
    # 1. Construir o DataFrame de Métricas de Teste de forma robusta
    # A chave do dicionário (ex: 'ROC_AUC') se torna o índice.
    # 'Test_Value' se torna o nome da coluna de dados.
    metrics_combined = pd.DataFrame.from_dict(
        test_metrics, orient='index', columns=['Test_Value']
    )
    
    # 2. Carregar e extrair resultados de CV
    cv_df = pd.read_csv(cv_results_path)
    
    # Encontrar o melhor modelo (rank_test_score = 1)
    best_cv = cv_df[cv_df['rank_test_score'] == 1].iloc[0]
    
    cv_mean = best_cv['mean_test_score']
    cv_std = best_cv['std_test_score']
    
    # 3. Adicionar colunas de CV ao DataFrame combinado
    # Inicializa as novas colunas com NaN (para métricas que não têm CV)
    metrics_combined['CV_Mean'] = float('nan')
    metrics_combined['CV_Std'] = float('nan')
    
    # 4. Preencher os valores de CV na linha ROC_AUC (métrica otimizada)
    # Garante que as chaves sejam compatíveis
    METRIC_NAME = 'ROC_AUC'
    
    if METRIC_NAME in metrics_combined.index:
        metrics_combined.loc[METRIC_NAME, 'CV_Mean'] = cv_mean
        metrics_combined.loc[METRIC_NAME, 'CV_Std'] = cv_std
    
    # 5. Salvar o novo DataFrame
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    # Usar index=True para escrever os nomes das métricas na primeira coluna
    metrics_combined.to_csv(
        f'{REPORTS_DIR}/results.csv', 
        header=True, 
        index=True, 
        index_label='' # Garante que o nome do índice (primeira célula vazia) seja vazio
    )
    print("✅ Relatório final de métricas (Teste + CV) salvo com sucesso.")
    return metrics_combined


def run_evaluation(args):
    """Loads the model, splits data, calculates metrics, and generates plots."""
    
    # 1. Load the best model
    try:
        with open(args.model, 'rb') as f:
            best_model = pickle.load(f)
        print("✅ Best model loaded successfully.")
    except FileNotFoundError:
        print(f"❌ Error: Model file not found at {args.model}. Execute 'python -m src.train_cv' first.")
        sys.exit(1)

    # 2. Load and split data (for an unseen test set)
    print("Loading data and splitting into Test set...")
    X_full, y_full = load_and_preprocess(args.data)
    
    # Ensure stratification is applied for the binary target
    strat = y_full if y_full.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=0.25, random_state=RANDOM_STATE, stratify=strat
    )
    
    # 3. Predict on the Test set
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    
    # 4. Calculate and Save Metrics
    metrics = calculate_all_metrics(y_test, y_pred, y_proba)
    metrics_df = pd.DataFrame(metrics, index=['Test_Value']).T
    os.makedirs(REPORTS_DIR, exist_ok=True)
    metrics_df.to_csv(f'{REPORTS_DIR}/results.csv', header=True)
    print("\n✅ Final Metrics (on Test Set) saved to reports/results.csv:")
    print(metrics_df.round(4))
    
    load_and_combine_results(metrics, f'{REPORTS_DIR}/cv_results.csv')

    # 5. Generate and Save Figures
    plot_and_save_figures(best_model, X_test, y_test, y_proba, metrics, X_full.columns, args.out)

if __name__ == '__main__':
    args = parse_args()
    run_evaluation(args)