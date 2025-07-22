import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
import lightgbm as lgb
import joblib

DATA_DIR = "../data/csv_features"
MODEL_DIR = "../models"
os.makedirs(MODEL_DIR, exist_ok=True)

def load_datasets():
    train_path = os.path.join(DATA_DIR, "train_train.csv")
    val_path = os.path.join(DATA_DIR, "val_val.csv")
    test_path = os.path.join(DATA_DIR, "test_test.csv")

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    print(f"[INFO] Loaded train: {df_train.shape}, val: {df_val.shape}, test: {df_test.shape}")
    return df_train, df_val, df_test

def prepare_data(df):
    X = df.drop(columns=['label'])
    y = df['label']
    return X, y

def evaluate_model(model, X, y, dataset_name="DATA"):
    y_pred = model.predict(X)
    y_pred_binary = (y_pred >= 0.5).astype(int)
    acc = accuracy_score(y, y_pred_binary)
    roc_auc = roc_auc_score(y, y_pred)
    print(f"\n=== {dataset_name} RESULTS ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC AUC:  {roc_auc:.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y, y_pred_binary))
    print("Classification Report:\n", classification_report(y, y_pred_binary))
    return acc, roc_auc

df_train, df_val, df_test = load_datasets()
X_train, y_train = prepare_data(df_train)
X_val, y_val = prepare_data(df_val)
X_test, y_test = prepare_data(df_test)

train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val)

params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 64,
    'max_depth': -1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'is_unbalance': True  # handles class imbalance
}

print("[INFO] Training LightGBM...")
callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=True)]

model = lgb.train(
    params,
    train_data,
    valid_sets=[train_data, val_data],
    num_boost_round=500,
    callbacks=callbacks
)

print("\n[INFO] Evaluating on TRAIN, VAL, TEST...")
evaluate_model(model, X_train, y_train, "TRAIN")
evaluate_model(model, X_val, y_val, "VAL")
evaluate_model(model, X_test, y_test, "TEST")

model_path = os.path.join(MODEL_DIR, "stego_lgbm_model.txt")
model.save_model(model_path)
print(f"[SAVED] LightGBM model -> {model_path}")
