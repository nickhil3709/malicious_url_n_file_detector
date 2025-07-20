import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# Load enhanced feature dataset
df = pd.read_csv("../data/feature_pe_data.csv")

# Define feature columns (including new entropy features)
FEATURE_COLUMNS = [
    "file_size",
    "has_digital_signature",
    "num_libraries",
    "lib_entropy",
    "num_functions",
    "func_entropy",
    "num_suspicious_apis"
]

X = df[FEATURE_COLUMNS]
y = df['label']

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print("[*] Training RandomForest on PE dataset with enhanced features...")

# Train RandomForest
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save model & scaler
joblib.dump((model, scaler, FEATURE_COLUMNS), "../models/pe_rf_model.joblib")
print("✅ Model saved as ../models/pe_rf_model.joblib")
