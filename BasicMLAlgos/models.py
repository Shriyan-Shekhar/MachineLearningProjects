"""
models.py

Contains ModelTrainer class that trains multiple regression models to predict part density.
Saves the best model to disk with joblib.
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

import xgboost as xgb
import lightgbm as lgb

class ModelTrainer:
    def __init__(self, model_dir="models"):
        """
        Initialize ModelTrainer with directory to save models.
        
        Parameters:
            model_dir (str): Directory to save trained models.
        """
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.models = {}
        self.scores = {}

    def _evaluate(self, model, X_train, X_test, y_train, y_test):
        """
        Fit model, predict on test set, and return evaluation metrics.

        Parameters:
            model: sklearn regression model
            X_train: Training features
            X_test: Testing features
            y_train: Training targets
            y_test: Testing targets
            Returns: dict of evaluation metrics and predictions
        """
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        return {'rmse': rmse, 'mae': mae, 'r2': r2}, preds

    def train_and_select(self, X, y, random_state=42, test_size=0.2):
        """
        Train multiple models and select the best model.
        
        Parameters:
            X: Features DataFrame
            y: Target Series
            random_state (int): Random seed for reproducibility
            test_size (float): Proportion of data to use as test set
            Returns: best_model, best_metrics, metrics_df
            """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        #Take cols for scaling
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns

        scaler = StandardScaler()
        X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

        #All the predefined models for training and testing
        candidates = {
            'ridge': Ridge(alpha=1.0, random_state=random_state),
            'elasticnet': ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=random_state),
            'rf': RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=random_state),
            'extratrees': ExtraTreesRegressor(n_estimators=100, n_jobs=-1, random_state=random_state),
            'gbr': GradientBoostingRegressor(n_estimators=100, random_state=random_state),
            'xgb': xgb.XGBRegressor(n_estimators=100, n_jobs=-1, random_state=random_state, verbosity=0),
            'lgbm': lgb.LGBMRegressor(n_estimators=100, n_jobs=-1, random_state=random_state, verbosity=-1),
            'svr': SVR(),
            'knn': KNeighborsRegressor(n_neighbors=5, n_jobs=-1)
        }

        #Comparing results of models with test dataset
        results = []
        for name, model in candidates.items():
            metrics, preds = self._evaluate(model, X_train, X_test, y_train, y_test)
            results.append({'model': name, **metrics})
            self.models[name] = model

        metrics_df = pd.DataFrame(results).sort_values('rmse')
        best_row = metrics_df.iloc[0]
        best_name = best_row['model']
        best_model = self.models[best_name]
        best_metrics = best_row.to_dict()

        # Save best model
        model_path = os.path.join(self.model_dir, f"best_model_{best_name}.joblib")
        joblib.dump(best_model, model_path)
        print(f"Saved best model ({best_name}) to {model_path}")

        # Save metrics CSV
        metrics_df.to_csv(os.path.join(self.model_dir, "metrics_summary.csv"), index=False)

        return best_model, best_metrics, metrics_df