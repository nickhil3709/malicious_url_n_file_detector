import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import sys

DATA_PATH = "../data/PDFMalware2022.csv"
MODEL_OUT = "pdf_rf_model.joblib"
RANDOM_STATE = 42

# -----------------------
# Load
# -----------------------
df = pd.read_csv(DATA_PATH)
print(f"[INFO] Loaded dataset shape: {df.shape}")
print("[INFO] Columns:", list(df.columns))

# -----------------------
# Basic Cleaning
# -----------------------
# Drop rows with missing label
if "Class" not in df.columns:
    print("[ERROR] 'Class' column not found.", file=sys.stderr)
    sys.exit(1)

before = df.shape[0]
df = df.dropna(subset=["Class"])
print(f"[INFO] Dropped {before - df.shape[0]} rows with NaN Class.")

# Standardize label values (strip + title case)
df["Class"] = df["Class"].astype(str).str.strip().str.title()

# Keep only Benign / Malicious
valid_classes = {"Benign": 0, "Malicious": 1}
df = df[df["Class"].isin(valid_classes.keys())].copy()
print(f"[INFO] Remaining rows after filtering valid classes: {df.shape[0]}")

# -----------------------
# Encode the 'Class' label
# -----------------------
y = df["Class"].map(valid_classes).astype(int)

# -----------------------
# Encode 'text' column (multi-valued)
# -----------------------
def map_text_column(val):
    v = str(val).strip().lower()
    if v == "yes":
        return 1.0
    if v == "no":
        return 0.0
    if v == "unclear":
        return 0.5
    # Numeric-like? attempt convert
    try:
        return float(v)
    except ValueError:
        return 0.0

if "text" in df.columns:
    df["text"] = df["text"].apply(map_text_column)

# -----------------------
# Identify possible Yes/No flag columns
# (We will map any column whose unique *string* values are subset of {Yes, No}
# OR mixture of those plus NaNs.)
# -----------------------
possible_flag_cols = []
for col in df.columns:
    if col == "Class":
        continue
    if df[col].dtype == object:
        unique_vals = set(str(x).strip().lower() for x in df[col].dropna().unique())
        # Accept minor variations (e.g. 'yes', 'no', '0', '1')
        if unique_vals.issubset({"yes", "no", "y", "n", "1", "0", ""}):
            possible_flag_cols.append(col)

# Explicit list (from dataset preview) – will override detection if present
explicit_yes_no = [
    "isEncrypted", "JS", "Javascript", "AA", "OpenAction", "Acroform",
    "JBIG2Decode", "RichMedia", "launch", "EmbeddedFile", "encrypt"
]
# Merge (keep duplicates unique)
flag_cols = sorted(set(possible_flag_cols + [c for c in explicit_yes_no if c in df.columns]))

def map_yes_no(val):
    v = str(val).strip().lower()
    if v in {"yes", "y", "1"}:
        return 1
    if v in {"no", "n", "0"}:
        return 0
    if v == "" or v == "nan":
        return 0
    # Fallback: try numeric
    try:
        return int(float(v))
    except:
        return 0

for col in flag_cols:
    df[col] = df[col].apply(map_yes_no)

print(f"[INFO] Mapped Yes/No style columns: {flag_cols}")

# -----------------------
# Header column (PDF magic)
# -----------------------
if "header" in df.columns:
    df["header"] = df["header"].apply(
        lambda x: 1 if isinstance(x, str) and x.startswith("%PDF") else 0
    )

# -----------------------
# Drop unused identifier columns
# -----------------------
drop_cols = [c for c in ["Fine name", "File name", "Filename"] if c in df.columns]
if drop_cols:
    print(f"[INFO] Dropping identifier columns: {drop_cols}")
    df = df.drop(columns=drop_cols)

# -----------------------
# After encoding 'text' & flags, convert remaining object columns
# -----------------------
remaining_object_cols = df.select_dtypes(include=["object"]).columns.tolist()
remaining_object_cols = [c for c in remaining_object_cols if c != "Class"]

if remaining_object_cols:
    print(f"[INFO] Attempting numeric coercion for columns: {remaining_object_cols}")
    for col in remaining_object_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Replace residual NaNs with 0
df = df.fillna(0)

# -----------------------
# Final feature matrix
# -----------------------
feature_cols = [c for c in df.columns if c != "Class"]
X = df[feature_cols]

# Sanity check: ensure all numeric
non_numeric_final = X.select_dtypes(exclude=[np.number]).columns.tolist()
if non_numeric_final:
    print("[ERROR] Non-numeric columns remain:", non_numeric_final, file=sys.stderr)
    # Show sample values to debug
    for c in non_numeric_final:
        print(f"  Column '{c}' sample unique values:", X[c].unique()[:10], file=sys.stderr)
    sys.exit(1)

print(f"[INFO] Final feature count: {len(feature_cols)}")
print("[INFO] Sample dtypes:\n", X.dtypes.head())

# -----------------------
# Train/Test Split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"[INFO] Train size: {X_train.shape}, Test size: {X_test.shape}")

# -----------------------
# Scale (optional for RF, but keeps parity w/ other models)
# -----------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------
# Train Model
# -----------------------
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=16,
    random_state=RANDOM_STATE,
    class_weight="balanced_subsample",
    n_jobs=-1
)
model.fit(X_train_scaled, y_train)

# -----------------------
# Evaluation
# -----------------------
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"\n[RESULT] Accuracy: {acc:.4f}")
print("\n[RESULT] Confusion Matrix:\n", cm)
print("\n[RESULT] Classification Report:\n", classification_report(y_test, y_pred, digits=4))

# Optional: simple threshold tuning suggestion
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_test, y_prob)
print(f"[RESULT] ROC AUC: {auc:.4f}")

# -----------------------
# Save
# -----------------------
joblib.dump((model, scaler, feature_cols), MODEL_OUT)
print(f"\n[SAVED] Model + scaler + feature columns -> {MODEL_OUT}")
