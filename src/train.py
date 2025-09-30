import pandas as pd
import numpy as np
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from scipy.stats import uniform, randint
import pickle
import os
import argparse

from src.config import K_FOLDS, RANDOM_STATE, REPORTS_DIR, DATA_PATH, HGBC_PARAM_DIST, ARTIFACTS_DIR
from src.model import get_pipeline
from src.utils import save_artifact

from src.preprocess import load_and_preprocess

def parse_args():
    parser = argparse.ArgumentParser(description="Train and tune HistGradientBoostingClassifier model.")
    parser.add_argument('--data', type=str, default=DATA_PATH, help='Path to the water.csv dataset.')
    parser.add_argument('--out', type=str, default=REPORTS_DIR, help='Output directory for CV results.')
    parser.add_argument('--k', type=int, default=K_FOLDS, help='Number of folds for Stratified K-Fold.')
    parser.add_argument('--seed', type=int, default=RANDOM_STATE, help='Fixed random state/seed.')
    return parser.parse_args()

def run_training_and_cv(args):
    
    print("Loading and preprocessing data...")
    X, y = load_and_preprocess(args.data)
    
    # Define Stratified K-Fold Cross-Validation
    kf = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=args.seed)

    # Randomized Search setup
    rs = RandomizedSearchCV(
        get_pipeline(), 
        param_distributions=HGBC_PARAM_DIST, 
        n_iter=50,
        cv=kf,
        scoring='roc_auc',
        n_jobs=-1, 
        random_state=args.seed, 
        verbose=1
    )
    
    # Execute Search
    print(f"\nStarting Randomized Search with {args.k}-Fold Stratified CV...")
    rs.fit(X, y)
    
    best_model = rs.best_estimator_
    
    # Save Artifacts
    save_artifact(best_model, 'best.pkl', ARTIFACTS_DIR)
        
    print("\nTraining complete.")
    print(f"Best Hyperparameters: {rs.best_params_}")
    print(f"Best ROC-AUC on CV: {rs.best_score_:.4f}")
    
    # Save detailed CV results for reporting
    os.makedirs(args.out, exist_ok=True)
    results_df = pd.DataFrame(rs.cv_results_)
    # Calculate Mean ± STD of best score for the report
    best_index = rs.best_index_
    mean_score = results_df.loc[best_index, 'mean_test_score']
    std_score = results_df.loc[best_index, 'std_test_score']
    print(f"Best Model CV Score (Mean ± STD): {mean_score:.4f} ± {std_score:.4f}")
    
    # Save the full results of all iterations
    results_df.to_csv(f'{args.out}/cv_results.csv', index=False)
    print(f"Full CV results saved to {args.out}/cv_results.csv")

if __name__ == '__main__':
    args = parse_args()
    run_training_and_cv(args)