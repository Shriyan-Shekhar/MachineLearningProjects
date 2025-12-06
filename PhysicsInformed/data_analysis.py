"""
data_analysis.py

Contains a DataAnalyzer class that loads the three CSVs, performs exploratory analysis,
feature engineering, and returns trainable feature matrices.

"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

class DataAnalyzer:
    def __init__(self, builds_path, machine_logs_path, parts_path):
        """Initialize with paths to the three CSV files."""
        self.builds_path = builds_path
        self.machine_logs_path = machine_logs_path
        self.parts_path = parts_path
        self.builds = None
        self.machine_logs = None
        self.parts = None
        self.encoders = {}
        self.imputers = {}

    def load_data(self):
        """Load CSV files into pandas DataFrames."""
        self.builds = pd.read_csv(self.builds_path, parse_dates=True, low_memory=False)
        self.machine_logs = pd.read_csv(self.machine_logs_path, parse_dates=True, low_memory=False)
        self.parts = pd.read_csv(self.parts_path, parse_dates=True, low_memory=False)

        # Try to parse common datetime columns if present
        for df in [self.builds, self.machine_logs, self.parts]:
            for col in ['start_time', 'end_time', 'timestamp']:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')

    def aggregate_machine_logs(self):
        """
        Aggregate machine logs per build:
        Match records where machine_id matches and timestamp lies between build start/end time.
        Returns a per-build aggregated DataFrame.

        Parameters:
            None
        """
        ml = self.machine_logs.copy()

        required_build_cols = ['build_id', 'machine', 'start_time', 'end_time']
        required_log_cols = ['machine_id', 'timestamp']

        if not all(c in self.builds.columns for c in required_build_cols):
            print("Missing build columns required for machine-log matching.")
            return pd.DataFrame()

        if not all(c in ml.columns for c in required_log_cols):
            print("Missing machine-log columns required for timestamp matching.")
            return pd.DataFrame()

        ml_numeric_cols = ml.select_dtypes(include=['number']).columns.tolist()
        if not ml_numeric_cols:
            print("No numeric machine-log columns to aggregate.")
            return pd.DataFrame()

        results = []
        builds_small = self.builds[required_build_cols].dropna().copy()

        for _, build in builds_small.iterrows():
            mask = (
                (ml['machine_id'] == build['machine']) &
                (ml['timestamp'] >= build['start_time']) &
                (ml['timestamp'] <= build['end_time'])
            )
            subset = ml.loc[mask]

            if subset.empty:
                continue

            stats = subset[ml_numeric_cols].agg(['mean', 'std', 'min', 'max', 'median']).T
            stats = stats.stack()
            stats.index = [f"{col}_{stat}" for col, stat in stats.index]


            row = {'build_id': build['build_id']}
            row.update(stats.to_dict())

            results.append(row)

        #print (pd.DataFrame(results).head())
        #print (pd.DataFrame(results).columns)
        return pd.DataFrame(results)



    def prepare_features(self, debug=False):
        """
        Returns X (DataFrame) and y (Series) ready for modelling.
        Steps:
        - load data if not already loaded
        - aggregate machine logs per build
        - merge builds and parts tables
        - feature engineer: durations, positional info, simple geometry encodings
        - encode categorical variables and impute missing values

        Parameters:
            debug (bool): If True, prints shapes and columns of X and y
        """
        if self.builds is None or self.machine_logs is None or self.parts is None:
            self.load_data()

        ml_agg = self.aggregate_machine_logs()

        # Merge builds with parts
        df = self.parts.copy()
        if 'build_id' in df.columns and 'build_id' in self.builds.columns:
            df = df.merge(self.builds, on='build_id', how='left', suffixes=('','_build'))

        if not ml_agg.empty and 'build_id' in ml_agg.columns:
            df = df.merge(ml_agg, on='build_id', how='left')

        # Target: density
        target_cols = [c for c in df.columns if 'relative_density' in c.lower()]

        if len(target_cols) == 0:
            raise ValueError("No density column found in parts.csv. Columns: " + ", ".join(df.columns))
        
        y_col = target_cols[0]
        y = df[y_col].astype(float).copy()

        X = pd.DataFrame(index=df.index)
        #print (df.head())

        # Add numeric part features
        part_numeric_cols = [
            'volume_mm3', 'x_position_mm', 'y_position_mm', 'z_position_mm',
            'hatch_spacing_mm', 'layer_thickness_um', 'avg_scan_speed_mm_s',
            'avg_laser_power_w'
        ]

        for c in part_numeric_cols:
            if c in df.columns:
                X[c] = pd.to_numeric(df[c], errors='coerce')

        # Add Build duration Feature
        if 'start_time' in df.columns and 'end_time' in df.columns:
            X['build_duration_seconds'] = (
                pd.to_datetime(df['end_time']) -
                pd.to_datetime(df['start_time'])
            ).dt.total_seconds()

        # Add Categorical features
        categorical = []
        for c in ['machine', 'powder_batch', 'part_number', 'geometry']:
            if c in df.columns:
                categorical.append(c)
                X[c] = df[c]

        # Add Volumetric Energy Density (VED) = Power / (Speed * Hatch * Thickness)
        if all(c in X.columns for c in
            ['avg_laser_power_w', 'avg_scan_speed_mm_s', 'hatch_spacing_mm', 'layer_thickness_um']):
            X['VED'] = (
                X['avg_laser_power_w'] /
                (X['avg_scan_speed_mm_s'] *
                X['hatch_spacing_mm'] *
                X['layer_thickness_um'])
            )

        # Add Machine log features (aggregated stats)
        ml_cols = [c for c in df.columns
                if any(s in c for s in ['_mean', '_std', '_max', '_median'])]
        for c in ml_cols:
            X[c] = pd.to_numeric(df[c], errors='coerce')

        # Encode categoricals by One Hot or Frequency Encoding - mainly one hot
        self.encoders = {}
        for c in categorical:
            nunique = X[c].nunique(dropna=False)
            if nunique <= 10:
                ohe = pd.get_dummies(X[c].astype(str), prefix=c, dummy_na=True) #Same as using OneHotEncoder
                X = pd.concat([X.drop(columns=[c]), ohe], axis=1)
                self.encoders[c] = ('onehot', ohe.columns.tolist())
            else:
                # Frequency encoding
                freq = X[c].fillna('##NA##').value_counts(normalize=True).to_dict()
                X[c + '_freq'] = X[c].fillna('##NA##').map(freq)
                X = X.drop(columns=[c])
                self.encoders[c] = ('freq', None)

        constant_cols = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
        if constant_cols:
            X = X.drop(columns=constant_cols)

        bool_cols = X.select_dtypes(include='bool').columns
        X[bool_cols] = X[bool_cols].astype(int)

        # Take out missing numeric values
        self.imputers = {}
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) > 0:
            imp = SimpleImputer(strategy='median')
            X[num_cols] = imp.fit_transform(X[num_cols])
            self.imputers['num'] = imp

        # Drop remaining non-numeric columns (shouldn’t be any at this point)
        non_num = X.select_dtypes(include=['object']).columns.tolist()
        if non_num:
            X = X.drop(columns=non_num)

        if debug:
            print("X shape:", X.shape, "y shape:", y.shape)

        return X, y, y_col
    
    def plot_feature_correlation(self, X, y, threshold=0.4, figsize=(12,8)):
        """
        Plots a heatmap of features strongly correlated with the target and removes weakly correlated features.
        
        Parameters:
            X (DataFrame): Feature matrix
            y (Series): Target variable (relative_density)
            threshold (float): Absolute correlation threshold. Features with |corr| < threshold will be removed
            figsize (tuple): Figure size for heatmap
            debug (bool): If True, prints removed columns and correlations
        
        Returns:
            X_filtered (DataFrame): Features strongly correlated with target
            corr_target (Series): Correlation of remaining features with target
        """
        # Combine features and target
        df_corr = X.select_dtypes(include=[np.number]).copy()
        df_corr['target'] = y
        # Compute correlations
        corr_matrix = df_corr.corr()
        corr_target = corr_matrix['target'].drop('target')  # correlations with target only

        # Select features above threshold
        strong_features = corr_target[abs(corr_target) >= threshold].index.tolist()
        weak_features = corr_target[abs(corr_target) < threshold].index.tolist()

        # Filter X
        X_filtered = X[strong_features].copy()

        # Heatmap of correlations with target
        plt.figure(figsize=figsize)
        sns.heatmap(df_corr[strong_features + ['target']].corr(), annot=True, cmap='coolwarm', center=0)
        plt.title("Feature Correlations with Target (|corr| >= {:.2f})".format(threshold))
        plt.show()

        return X_filtered, corr_target[strong_features]
    
    def analyze_trends(self, X, y, top_n_numeric=5, figsize=(12,6)):
        """
        Analyze key trends in the dataset:
        - Visualizes categorical distributions vs target
        - Visualizes top numeric features vs target
        - Prints aggregate statistics for machine logs and builds
        
        Parameters:
            X (DataFrame): Feature matrix
            y (Series): Target variable (relative_density)
            top_n_numeric (int): Number of top numeric features to visualize
            figsize (tuple): Figure size for plots
        """
        # Categorical Feature Trends
        categorical_cols = [c for c in X.columns if "_freq" in c or c.startswith("geometry") or c.startswith("powder_batch")]
        
        for col in categorical_cols:
            plt.figure(figsize=figsize)
            if col in X.columns:
                sns.boxplot(x=X[col].astype(str), y=y)
                plt.title(f"Target vs {col}")
                plt.xticks(rotation=45)
                plt.show()

        # Numeric Feature Trends
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        # Compute correlations with target
        corr_target = pd.concat([X[numeric_cols], y], axis=1).corr()['relative_density'].drop('relative_density')
        top_features = corr_target.abs().sort_values(ascending=False).head(top_n_numeric).index.tolist()

        for feature in top_features:
            if feature == 'machine_M001' or feature == 'machine_M002' or feature == 'machine_M000' or feature == 'hatch_spacing_mm': 
                continue
            plt.figure(figsize=figsize)
            sns.scatterplot(x=X[feature], y=y)
            plt.title(f"{feature} vs Relative Density")
            plt.show()

        # Aggregate Statistics for Machine Logs / Builds and demonstrate trends
        ml_cols = [c for c in X.columns if any(s in c for s in ['_mean'])]
        if ml_cols:
            summary_stats = X[ml_cols].describe().T[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]
            print(summary_stats)
        else:
            print("No aggregated machine log columns found for analysis.")

    
    def feature_importance(self, X, y, top_n=20, plot=True, figsize=(10,6), random_state=42):
        """
        Computes and visualizes feature importance using Random Forest.
        
        Parameters:
            X (DataFrame): Feature matrix (numeric only)
            y (Series): Target variable (relative_density)
            top_n (int): Number of top features to show
            plot (bool): Whether to plot the feature importance
            figsize (tuple): Figure size for the plot
            random_state (int): Random seed for reproducibility
            
        Returns:
            importance_df (DataFrame): Features ranked by importance
        """
        # All cols should be numeric but check
        X_num = X.select_dtypes(include=[np.number])

        # Remove Machine as a feature
        exclude_features = ['machine_M002', 'machine_M001', 'machine_M000']
        X_num = X_num.drop(columns=[f for f in exclude_features if f in X_num.columns])

        # Using Random Forest for importance
        rf = RandomForestRegressor(n_estimators=500, random_state=random_state)
        rf.fit(X_num, y)

        # Feature importances
        importances = pd.Series(rf.feature_importances_, index=X_num.columns)
        importance_df = importances.sort_values(ascending=False).head(top_n).reset_index()
        importance_df.columns = ['feature', 'importance']

        # Plotting Display
        if plot:
            plt.figure(figsize=figsize)
            plt.barh(importance_df['feature'][::-1], importance_df['importance'][::-1], color='skyblue')
            plt.xlabel("Importance")
            plt.title(f"Top {top_n} Feature Importances (Random Forest)")
            plt.show()

        return importance_df