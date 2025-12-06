"""
PINN_model.py

Contains implementation of Physics-Informed Neural Network (PINN) to predict part/relative density.
Includes training and testing functions. Algorithm enforces physics constraints via custom loss function.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler

class DensityPINN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        """
        Simple Feedforward Neural Network for Density Prediction.
        
        Parameters:
            input_dim (int): Number of input features.
            hidden_dim (int): Number of neurons in hidden layers.
        """
        super(DensityPINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)


def pinn_loss(model, X, y_true, lambda_physics=1.0, filtered=False):
    """
    Physics-informed loss enforcing sensible metallurgy constraints.
    
    Parameters:
        model: PINN model
        X: Input features (torch tensor)
        y_true: True density values (torch tensor)
        lambda_physics: Weight for physics loss component
        filtered: Boolean indicating if filtered data is used (affects feature indices)
        Returns: total loss, data loss, physics loss
    """
    X.requires_grad_(True)
    y_pred = model(X)
    mse = nn.MSELoss()(y_pred, y_true)

    grads = torch.autograd.grad(
        y_pred, X, grad_outputs=torch.ones_like(y_pred),
        create_graph=True
    )[0]

    i_ved = 9
    i_oxy = 10

    if filtered:
        i_ved = 3
        i_oxy = 4

    d_density_dVED = grads[:, i_ved]
    d_density_dOxy = grads[:, i_oxy]

    penalty_ved = torch.relu(-d_density_dVED).mean()    
    penalty_oxygen = torch.relu(d_density_dOxy).mean()   

    physics_loss = penalty_ved + penalty_oxygen

    return mse + lambda_physics * physics_loss, mse.item(), physics_loss.item()


def test_pinn(model, X_test, y_test, filtered=False):
    """
    Evaluate only supervised metrics (no physics loss).
    
    Parameters:
        model: Trained PINN model
        X_test: Test features DataFrame
        y_test: Test target Series
        filtered: Boolean indicating if filtered data is used (affects saving path)
        Returns: predictions, metrics dict
    """
    model.eval()

    X_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    with torch.no_grad():
        preds = model(X_tensor).cpu().numpy().flatten()

    y_true = y_test.to_numpy().flatten()

    rmse = np.sqrt(mean_squared_error(y_true, preds))
    mae = mean_absolute_error(y_true, preds)
    r2 = r2_score(y_true, preds)
    metrics = {"rmse": rmse, "mae": mae, "r2": r2}
    save_path = "./models"

    if save_path is not None:
        # Save predictions
        if filtered:
            save_path += "_filtered"
        df_results = pd.DataFrame({
            "y_true": y_true,
            "y_pred": preds
        })
        df_results.to_csv(f"{save_path}_predictions.csv", index=False)

        # Append metrics as a single row at the end
        with open(f"{save_path}_predictions.csv", "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([])  
            metrics_row = [f"{k}: {v}" for k, v in metrics.items()]
            writer.writerow(metrics_row)

        print(f"Saved predictions and metrics to {save_path}_predictions.csv")

    #Plot of results
    #Closer to line indicates that y_pred is closer to ground truth

    plt.figure(figsize=(8,6))
    sns.scatterplot(x=y_true, y=preds)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')  # 45° line
    plt.xlabel("True Values (y_true)")
    plt.ylabel("Predicted Values (y_pred)")
    plt.title("Predictions vs True Values")
    plt.grid(True)
    plt.show()

    return preds, metrics


def train_pinn_full(X, y, epochs=800, lr=1e-3, lambda_phys=0.2, test_size=0.2, random_state=42, filtered=False):
    """
    Fully contained pipeline for training + testing a PINN. Added filtered depending on data used.

    Parameters:
        X: Features DataFrame
        y: Target Series
        epochs (int): Number of training epochs
        lr (float): Learning rate
        lambda_phys (float): Weight for physics loss component
        test_size (float): Proportion of data to use as test set
        random_state (int): Random seed for reproducibility
        filtered (bool): Whether to use filtered data (affects feature selection)
        Returns: model, metrics, predictions, datasets
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = MinMaxScaler()
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
    

    X_train = X_train [numeric_cols]
    X_test = X_test [numeric_cols]

    model = DensityPINN(input_dim=X_train.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss, data_loss, phys_loss = pinn_loss(model, X_tensor, y_tensor, lambda_phys, filtered)
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch} | Total={loss:.4f} | Data={data_loss:.4f} | Phys={phys_loss:.4f}")

    preds, metrics = test_pinn(model, X_test, y_test, filtered=filtered)

    print("\n Final Test Results")
    print(f"RMSE: {metrics['rmse']:.5f}")
    print(f"MAE:  {metrics['mae']:.5f}")
    print(f"R²:   {metrics['r2']:.5f}\n")

    return model, metrics, preds, (X_train, X_test, y_train, y_test)
