"""
main.py

Main script to run data analysis, model training and testing, and PINN training and testing.
"""

from data_analysis import DataAnalyzer
import os
import pandas as pd
from models import ModelTrainer
from PINN_model import train_pinn_full

DATA_DIR = ""
BUILDS = os.path.join(DATA_DIR, "builds.csv")
MACHINE_LOGS = os.path.join(DATA_DIR, "machine_logs.csv")
PARTS = os.path.join(DATA_DIR, "parts.csv")

def main():
    #Data Prep and Corr Analysis
    print("Loading and preparing data...")
    da = DataAnalyzer(BUILDS, MACHINE_LOGS, PARTS)
    da.load_data()
    X, y, y_col = da.prepare_features(debug=True)
    X_filtered, corr_target = da.plot_feature_correlation(X, y, threshold=0.1)

    print("Finding trends")
    da.analyze_trends(X, y, top_n_numeric=5)

    mask = ~y.isna()
    X = X.loc[mask]
    y = y.loc[mask]
    X_filtered = X_filtered.loc[mask]

    print("Calculating feature importances...")
    importances = da.feature_importance(X, y, top_n=10, plot=True)

    
    #Training classic ML models to predict density
    print("Training models...")
    mt = ModelTrainer(model_dir=os.path.join(DATA_DIR, "models"))
    best_model, best_metrics, metrics_df = mt.train_and_select(X, y)
    print("\nBest model metrics for all data:")
    print(best_metrics)
    best_model_2, best_metrics_2, metrics_df_2 = mt.train_and_select(X_filtered, y)
    print("\nBest model metrics for filtered data:")
    print(best_metrics_2)
    
    #Hyperparameters - can be tuned further
    epochs_pinn = 10000
    lambda_phys = 0.1
    learning_rate = 1e-3

    # Training Physics-Informed Neural Network (PINN) to predict density
    print("Training Physics-Informed Neural Network (PINN)...")
    trained_model, pinn_metrics, pinn_preds, pinn_datasets = train_pinn_full(X, y, epochs=epochs_pinn,  lr=learning_rate, lambda_phys=lambda_phys, test_size=0.2, filtered = False)
    print("PINN training completed.")

    trained_model_filt, pinn_metrics_filt, pinn_preds_filt, pinn_datasets_filt = train_pinn_full(X_filtered, y, epochs=epochs_pinn,  lr=learning_rate, lambda_phys=lambda_phys, test_size=0.2, filtered = True)
    print("PINN training on filtered data completed.")
    

if __name__ == '__main__':
    main()