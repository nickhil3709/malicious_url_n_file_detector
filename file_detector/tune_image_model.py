import pandas as pd
import numpy as np
import os
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

DATA_DIR = "../data/csv_features"
MODEL_DIR = "../models"

def load_dataset():
    print(f"[INFO] Loading dataset from: {DATA_DIR}")
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train_train.csv"))
    val_df = pd.read_csv(os.path.join(DATA_DIR, "val_val.csv"))

    df = pd.concat([train_df, val_df], ignore_index=True)
    print(f"[INFO] Loaded dataset shape: {df.shape}")
    return df

def tune_xgboost(X, y):
    print("[INFO] Starting XGBoost hyperparameter tuning...")

    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        tree_method="hist",
        use_label_encoder=False
    )

    # Define parameter grid
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.7, 1.0],
        "colsample_bytree": [0.7, 1.0],
        "min_child_weight": [1, 3, 5]
    }

    sample_weights = compute_sample_weight(class_weight="balanced", y=y)

    grid_search = GridSearchCV(
        xgb,
        param_grid,
        scoring="roc_auc",
        cv=3,
        n_jobs=-1,
        verbose=2
    )

    grid_search.fit(X, y, sample_weight=sample_weights)

    print("[RESULT] Best Parameters:", grid_search.best_params_)
    print("[RESULT] Best ROC AUC:", grid_search.best_score_)

    return grid_search.best_estimator_

if __name__ == "__main__":
    df = load_dataset()
    X = df.drop(columns=["label"])
    y = df["label"]

    best_model = tune_xgboost(X, y)

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "stego_xgb_tuned.joblib")
    joblib.dump(best_model, model_path)
    print(f"[SAVED] Tuned model -> {model_path}")
