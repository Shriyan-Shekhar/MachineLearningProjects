# Density Prediction
Explored a Physics Informed Neural Network on a Materials Science Dataset. Main goal was to predict and not conduct data analysis.

## Running Instruction
Run `main.py` and ensure that all imports are resolved (download all relevant libraries in latest version on a virtual environment)

## Data Merging
The `DataAnalyzer` class handles loading and merging these datasets:

- **Builds and Parts:** Joined on `build_id` to associate part-level measurements with build-level metadata.
- **Machine Logs:** Aggregated per build by matching `machine_id` and filtering logs between `start_time` and `end_time`. I also identified that it can be merged based on start_time and machine_id. Numeric features such as sensor readings are summarized using mean, median, standard deviation, min, and max values.

This merging process results in a comprehensive per-part feature matrix containing both part-specific attributes and build/machine-level information.

## Feature Engineering
Several engineered features are added to improve model performance. The relevant features used in the final prepared dataset were:

- **Part Dimensions & Process Parameters:** Includes volume, positions (`x`, `y`, `z`), hatch spacing, layer thickness, average laser power, and scan speed.
- **Build Duration:** Calculated as the difference between build `end_time` and `start_time`.
- **Volumetric Energy Density (VED):** Computed as  
  `VED = avg_laser_power_w / (avg_scan_speed_mm_s * hatch_spacing_mm * layer_thickness_um)`  
  capturing the energy input per unit volume.
- **Machine Log Aggregates:** Statistical summaries (mean, std, min, max, median) for numeric machine sensor readings.
- **Categorical Encoding:**  
  - One-hot encoding for low-cardinality features (e.g., geometry, powder batch, machine).  
  - Frequency encoding for high-cardinality features.
- **Missing Value Imputation:** Numeric columns are imputed with median values.
- **Final Cleanup:** Non-informative or constant columns are removed to ensure model-ready feature matrices.

## Trend Identification
### Trend Analysis
Exploratory data analysis identifies patterns and correlations with the target variable (`relative_density`):


- **Correlation Analysis:** A correlation matrix is computed between numeric features and the target. Features with strong positive or negative correlations are highlighted, and weakly correlated features can be filtered out to simplify the data. 
- **Categorical Trends:** Boxplots visualize distributions of relative density across powder batches, machine IDs, and geometry types.
- **Numeric Trends:** Scatter plots and correlation analysis highlight features most strongly associated with density variations.
- **Machine Log Insights:** Summary statistics for aggregated sensor readings reveal operational conditions that influence part quality.
- **Feature Importance:** Random Forest models rank features by predictive power, helping identify the key contributors to relative density.

These analyses support informed feature selection. The visualizations can be easily found once you run `main.py`. The plots will be visible along with a table for machine log insights and a plot for feature importance.

## Machine Learning Models

### Models Used
The `ModelTrainer` class trains multiple regression models to predict part relative density. Target variable is once again `relative_density`. The following models are included:

- **Linear Models:** Ridge Regression, ElasticNet  
- **Tree-based Ensembles:** Random Forest, Extra Trees, Gradient Boosting  
- **Boosting Algorithms:** XGBoost, LightGBM  
- **Other Regressors:** Support Vector Regressor (SVR), K-Nearest Neighbors (KNN)  

All numeric features are standardized before training. Models are evaluated on a hold-out test set using **RMSE**, **MAE**, and **R²** metrics.

### Results
The performance of each model is summarized in `metrics_summary.csv` under the `models` directory.  
The best-performing model is automatically saved to disk for future inference.  

Key highlights:  
- Extra Trees model performed best on all data
- Light GBM model performed best on filtered data from correlation matrix

For full evaluation details, including metrics and saved model files, check the `models` folder in the project repository.

## Physics-Informed Neural Network (PINN)

### Overview
The `DensityPINN` model leverages a Physics-Informed Neural Network (PINN) approach to predict part relative density while enforcing physical constraints. The model combines:

- **Supervised Learning:** Predicting density from part, build, and machine features.  
- **Physics Constraints:** Gradients with respect to Volumetric Energy Density (VED) and oxygen content are penalized to ensure physically realistic trends (e.g., density should increase with VED and decrease with oxygen content).

The PINN loss function is a combination of standard mean squared error (MSE) and physics-based penalties.

### Results
- Predictions and metrics are saved as `model_predictions.csv` (all data) and `model_filtered_predictions.csv` (filtered data).  
- The filtered dataset consistently outperforms the full dataset across all metrics:
  - Lower **RMSE** and **MAE**
  - Higher **R²**
- Scatter plots of predicted vs true density show that predictions on the filtered dataset are closer to the 45° line, indicating fewer outliers. The full dataset exhibits more deviations from the ground truth. The MAE was below 1, which suggests that the model predicts well as the relative density values are exceeding 80, so the model results are about 1-2% inaccurate. R^2 value is not as close to 1 due to outliers in predictions (and in data effectively).

### Improvements to model
- **Physics-Informed Loss:** The Loss function can be improved with more knowledge of this domain. 
- **Overfitting and Underfitting:** Regularization techniques, early stopping, dropout layers, or cross-validation can help mitigate overfitting. Conversely, increasing network capacity or using more features may reduce underfitting.  
- **Other Models:** Time-series or sequence-based deep learning models such as RNNs, LSTMs, and GRUs could be explored, especially to capture temporal dependencies in machine logs or layer-wise build dynamics.  
- **Feature Engineering:** Incorporating more physics-based features or transformations (e.g., normalized energy input, part orientation effects) can further improve predictive accuracy. 
- **Hyperparamaters:**  Can be tuned further to improve performance such as changing phys lambda value or number of epochs (increase training time)


