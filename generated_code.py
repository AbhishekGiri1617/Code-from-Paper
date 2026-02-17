```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Monolithic stock-price forecasting pipeline.

Features
--------
* Load a CSV containing historical OHLCV data.
* Basic cleaning + a handful of technical indicators.
* Several regression models:
    - LinearRegression, DecisionTree, RandomForest, AdaBoost,
      GradientBoosting, XGBoost, KNeighbors, SVR
* LSTM implemented in PyTorch.
* Optional Optuna hyper-parameter tuning.
* Evaluation metrics: RMSE, MAE, MAPE, ME (bias).
* Tiny CLI built with Typer.

All code lives in this single file – no external package
structure is required.  Adjust the ``CONFIG`` dictionary below
to point at your data and to change hyper-parameters.
"""

# ----------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import pandas_ta as ta
import torch
import typer
import yaml
from loguru import logger
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor

# Optional hyper-parameter optimisation
try:
    import optuna
except Exception:  # pragma: no cover
    optuna = None

# ----------------------------------------------------------------------
# Global configuration (replace values as needed)
# ----------------------------------------------------------------------
CONFIG: Dict[str, Any] = {
    "data": {
        "csv_path": "stock_data.csv",          # path to your OHLCV CSV
        "date_col": "Date",
        "target": "Close",
        "train_split": 0.8,                    # proportion of rows for training
    },
    "features": {
        "lag_days": [1, 2, 3, 5, 10],
    },
    "model": {
        "type": "rf",                          # one of: linear, tree, rf, ada,
                                               # gboost, xgboost, knn, svr, lstm
        "forecast_horizon": 10,                # how many days ahead to predict
        "seq_len": 30,                         # only used for LSTM
        "input_dim": None,                     # will be inferred automatically
        "hidden_dim": 64,
        "num_layers": 2,
        "dropout": 0.2,
    },
    "training": {
        "batch_size": 64,
        "epochs": 30,
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "device": "cpu",                       # "cuda" if GPU is available
        "log_interval": 5,
        "tune": False,                         # set True to run Optuna tuning
        "n_trials": 30,
    },
    "evaluation": {
        "metrics": ["RMSE", "MAE", "MAPE", "ME"],
    },
    "logging": {
        "level": "INFO",
    },
}


# ----------------------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    """Make experiments reproducible."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str | None = None) -> Dict[str, Any]:
    """Load a YAML config file if supplied, otherwise fall back to ``CONFIG``."""
    if path is None:
        return CONFIG
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ----------------------------------------------------------------------
# Data loading & cleaning
# ----------------------------------------------------------------------
def load_stock(csv_path: Path, date_col: str = "Date") -> pd.DataFrame:
    logger.info(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path, parse_dates=[date_col])
    return df


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by date and forward-fill missing numeric values."""
    logger.info("Cleaning data – sorting, forward-filling")
    df = df.sort_values("Date").reset_index(drop=True)

    numeric = ["Open", "High", "Low", "Close", "Volume"]
    df[numeric] = df[numeric].ffill().bfill()
    return df


# ----------------------------------------------------------------------
# Feature engineering
# ----------------------------------------------------------------------
def add_lag_features(df: pd.DataFrame, lags: List[int]) -> pd.DataFrame:
    for lag in lags:
        df[f"Close_lag_{lag}d"] = df["Close"].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    # Simple moving averages
    df["MA_5"] = df["Close"].rolling(5).mean()
    df["MA_10"] = df["Close"].rolling(10).mean()

    # Bollinger Bands (20-day window)
    bb = ta.bbands(df["Close"], length=20, std=2)
    df["BB_upper"] = bb["BBU_20_2.0"]
    df["BB_middle"] = bb["BBM_20_2.0"]
    df["BB_lower"] = bb["BBL_20_2.0"]

    # Relative Strength Index
    df["RSI_14"] = ta.rsi(df["Close"], length=14)

    # Rate of change
    df["ROC_5"] = ta.roc(df["Close"], length=5)

    return df


def engineer_features(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    logger.info("Running feature engineering")
    df = add_lag_features(df, cfg["features"]["lag_days"])
    df = add_rolling_features(df)
    df = df.dropna().reset_index(drop=True)  # drop rows that contain NaNs after shifts
    return df


# ----------------------------------------------------------------------
# Model wrappers (sklearn unified API)
# ----------------------------------------------------------------------
class SklearnWrapper:
    """Thin wrapper that mimics the .fit/.predict API of PyTorch models."""

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.estimator.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.estimator.predict(X)


# ----------------------------------------------------------------------
# LSTM implementation (PyTorch)
# ----------------------------------------------------------------------
class LSTMRegressor(nn.Module):
    """Multi-step LSTM that predicts ``output_len`` future values."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_len: int = 10,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_dim, output_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, input_dim)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device)
        out, _ = self.lstm(x, (h0, c0))  # out: (batch, seq_len, hidden_dim)
        out = out[:, -1, :]  # last time step
        out = self.fc(out)   # (batch, output_len)
        return out


class LSTMTrainer:
    """Encapsulates training / inference for ``LSTMRegressor``."""

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.device = torch.device(cfg["training"]["device"])
        model_cfg = cfg["model"]
        self.model = LSTMRegressor(
            input_dim=model_cfg["input_dim"],
            hidden_dim=model_cfg["hidden_dim"],
            num_layers=model_cfg["num_layers"],
            dropout=model_cfg["dropout"],
            output_len=model_cfg["forecast_horizon"],
        ).to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg["training"]["lr"],
            weight_decay=cfg["training"]["weight_decay"],
        )

    def fit(self, loader: DataLoader) -> None:
        epochs = self.cfg["training"]["epochs"]
        log_int = self.cfg["training"]["log_interval"]
        self.model.train()
        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            for xb, yb in loader:
                xb = xb.to(self.device).float()
                yb = yb.to(self.device).float()
                self.optimizer.zero_grad()
                preds = self.model(xb)
                loss = self.criterion(preds, yb)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
            epoch_loss /= len(loader.dataset)
            if epoch % log_int == 0:
                logger.info(f"LSTM epoch {epoch}/{epochs} – loss {epoch_loss:.6f}")

    def predict(self, loader: DataLoader) -> np.ndarray:
        self.model.eval()
        all_preds = []
        with torch.no_grad():
            for xb in loader:
                xb = xb.to(self.device).float()
                out = self.model(xb)
                all_preds.append(out.cpu().numpy())
        return np.concatenate(all_preds, axis=0)


# ----------------------------------------------------------------------
# Training utilities
# ----------------------------------------------------------------------
MODEL_MAP = {
    "linear": LinearRegression,
    "tree": DecisionTreeRegressor,
    "rf": RandomForestRegressor,
    "ada": AdaBoostRegressor,
    "gboost": GradientBoostingRegressor,
    "xgboost": XGBRegressor,
    "knn": KNeighborsRegressor,
    "svr": SVR,
}


def time_series_split(df: pd.DataFrame, split_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological train / test split."""
    split_date = df["Date"].quantile(split_ratio)
    train = df[df["Date"] <= split_date].copy()
    test = df[df["Date"] > split_date].copy()
    logger.info(f"Train rows: {len(train)} | Test rows: {len(test)}")
    return train, test


def build_xy(
    df: pd.DataFrame,
    target: str,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    X = all columns except Date & target.
    y = matrix of ``horizon`` future target values.
    """
    feature_cols = [c for c in df.columns if c not in ("Date", target)]
    X = df[feature_cols].values.astype(np.float32)

    close = df[target].values
    y = np.stack(
        [close[i + 1 : i + 1 + horizon] for i in range(len(close) - horizon)],
        axis=0,
    )
    X = X[: len(y)]
    return X, y


def train_sklearn(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    cfg: Dict[str, Any],
) -> SklearnWrapper:
    """Instantiate (and optionally tune) a sklearn model, then fit."""
    ModelCls = MODEL_MAP[model_name]

    # ------------------ Hyper-parameter tuning (optional) ------------------
    if cfg["training"]["tune"] and optuna is not None:
        logger.info("Running Optuna tuning for {}", model_name)
        best_params = tune_hyperparameters(model_name, X, y, cfg)
    else:
        best_params = {}

    model = SklearnWrapper(ModelCls(**best_params))
    model.fit(X, y.ravel())
    return model


def tune_hyperparameters(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Very small Optuna search space – extend as needed."""
    if optuna is None:
        raise RuntimeError("Optuna is not installed – cannot tune.")

    def objective(trial):
        params = {}
        if name == "rf":
            params["n_estimators"] = trial.suggest_int("n_estimators", 50, 400)
            params["max_depth"] = trial.suggest_int("max_depth", 3, 20)
            params["min_samples_split"] = trial.suggest_int("min_samples_split", 2, 10)
        elif name == "xgboost":
            params["n_estimators"] = trial.suggest_int("n_estimators", 50, 400)
            params["learning_rate"] = trial.suggest_float(
                "learning_rate", 0.01, 0.3, log=True
            )
            params["max_depth"] = trial.suggest_int("max_depth", 3, 12)
        elif name == "knn":
            params["n_neighbors"] = trial.suggest_int("n_neighbors", 3, 20)
            params["weights"] = trial.suggest_categorical("weights", ["uniform", "distance"])
        else:
            # fallback – no hyper-params to tune
            return 0.0

        model = SklearnWrapper(MODEL_MAP[name](**params))
        tscv = TimeSeriesSplit(n_splits=3)
        rmses = []
        for tr_idx, val_idx in tscv.split(X):
            model.estimator.fit(X[tr_idx], y[tr_idx].ravel())
            preds = model.estimator.predict(X[val_idx])
            rmses.append(np.sqrt(mean_squared_error(y[val_idx].ravel(), preds)))
        return np.mean(rmses)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=cfg["training"]["n_trials"], show_progress_bar=True)
    logger.info("Best params for {}: {}", name, study.best_params)
    return study.best_params


def prepare_lstm_data(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
) -> Tuple[DataLoader, DataLoader, StandardScaler, List[str]]:
    """Create Torch DataLoaders for LSTM training & validation."""
    horizon = cfg["model"]["forecast_horizon"]
    seq_len = cfg["model"]["seq_len"]
    feature_cols = [c for c in df.columns if c not in ("Date", cfg["data"]["target"])]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols].values)

    # Build sliding windows
    X_seq, y_seq = [], []
    close = df[cfg["data"]["target"]].values
    for i in range(len(df) - seq_len - horizon + 1):
        X_seq.append(X_scaled[i : i + seq_len])
        y_seq.append(close[i + seq_len : i + seq_len + horizon])
    X_seq = np.stack(X_seq).astype(np.float32)
    y_seq = np.stack(y_seq).astype(np.float32)

    # Train-validation split (80/20 chronological)
    split_idx = int(0.8 * len(X_seq))
    X_tr, X_val = X_seq[:split_idx], X_seq[split_idx:]
    y_tr, y_val = y_seq[:split_idx], y_seq[split_idx:]

    train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        drop_last=False,
    )
    return train_loader, val_loader, scaler, feature_cols


# ----------------------------------------------------------------------
# Evaluation utilities
# ----------------------------------------------------------------------
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return mean_absolute_error(y_true, y_pred)


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0


def me(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Error (bias). Positive = over-prediction."""
    return np.mean(y_pred - y_true)


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizon: int,
) -> Dict[str, Dict[str, float]]:
    """Return metrics for each horizon step + overall"""
    metrics = {}
    for i in range(horizon):
        metrics[f"t+{i+1}"] = {
            "RMSE": rmse(y_true[:, i], y_pred[:, i]),
            "MAE": mae(y_true[:, i], y_pred[:, i]),
            "MAPE": mape(y_true[:, i], y_pred[:, i]),
            "ME": me(y_true[:, i], y_pred[:, i]),
        }

    # Overall metrics
    metrics["Overall"] = {
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "ME": me(y_true, y_pred),
    }
    return metrics


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main(
    csv_path: str = CONFIG["data"]["csv_path"],
    model_type: str = CONFIG["model"]["type"],
    forecast_horizon: int = CONFIG["model"]["forecast_horizon"],
    tune: bool = CONFIG["training"]["tune"],
) -> None:
    """Main entry point."""
    logger.info("Loading data")
    df = load_stock(csv_path)
    df = basic_clean(df)
    df = engineer_features(df, CONFIG)
    logger.info("Data loaded and preprocessed")

    if model_type == "lstm":
        train_loader, val_loader, scaler, feature_cols = prepare_lstm_data(df, CONFIG)
        model = LSTMTrainer(CONFIG)
        model.fit(train_loader)
        y_pred = model.predict(val_loader)
        y_true = df[CONFIG["data"]["target"]].values[-len(y_pred):]
        y_true = y_true.reshape(-1, forecast_horizon)
    else:
        train, test = time_series_split(df, CONFIG["data"]["train_split"])
        X_train, y_train = build_xy(train, CONFIG["data"]["target"], forecast_horizon)
        X_test, y_test = build_xy(test, CONFIG["data"]["target"], forecast_horizon)
        model = train