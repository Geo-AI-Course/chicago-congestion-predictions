"""Model training and evaluation.

Usage:
    python src/model.py
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

FEATURE_PATH = "data/features.parquet"
MODEL_PATH = "data/model.joblib"
TARGET = "congestion_score"
DROP_COLS = ["segment_id", "avg_volume", "max_volume", TARGET]


def load_data(path: str = FEATURE_PATH):
    df = pd.read_parquet(path)
    labeled = df[df[TARGET] > 0].copy()
    print(f"Training on {len(labeled)} labeled segments (of {len(df)} total)")
    feature_cols = [c for c in labeled.columns if c not in DROP_COLS]
    X = labeled[feature_cols]
    y = labeled[TARGET]
    return X, y


def evaluate(y_true, y_pred, label=""):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{label}  RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}")
    return {"rmse": rmse, "mae": mae, "r2": r2}


def plot_feature_importance(model, feature_names: list, out: str = "data/feature_importance.png"):
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importances)), importances[idx])
    plt.xticks(range(len(importances)),
               [feature_names[i] for i in idx], rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"Feature importance plot saved to {out}")


if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Baseline: predict training mean
    baseline_pred = np.full(len(y_test), y_train.mean())
    evaluate(y_test, baseline_pred, label="Baseline")

    # Random Forest — tune n_estimators and max_depth via 5-fold CV
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 10, None],
    }
    grid = GridSearchCV(
        RandomForestRegressor(n_jobs=-1, random_state=42),
        param_grid, cv=5, scoring="r2", n_jobs=-1, refit=True,
    )
    grid.fit(X_train, y_train)
    print(f"Best params: {grid.best_params_}  CV R²={grid.best_score_:.4f}")
    rf = grid.best_estimator_
    evaluate(y_test, rf.predict(X_test), label="RandomForest")

    plot_feature_importance(rf, list(X.columns))
    joblib.dump(rf, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
