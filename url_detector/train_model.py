import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import joblib
import re

# ✅ Step 1: Load feature dataset
df = pd.read_csv("../data/feature_data.csv")

# Drop unnecessary columns
if 'type' not in df.columns:
    raise ValueError("Dataset must have a 'type' column as target")
y = df['type']
X = df[['url', 'url_length', 'domain_length', 'num_digits', 'num_dots', 'is_https', 'has_suspicious_keywords']]

# ✅ Normalize URLs
def normalize_url(url_series):
    return url_series.apply(lambda u: re.sub(r'^(?:https?:\/\/)?(?:www\.)?', '', str(u).lower()))

# ✅ Apply normalization before TF-IDF
text_preprocessor = FunctionTransformer(normalize_url)

# Define columns
text_features = 'url'
numeric_features = ['url_length', 'domain_length', 'num_digits', 'num_dots', 'is_https', 'has_suspicious_keywords']

# ✅ ColumnTransformer with preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('text', Pipeline([
            ('normalize', text_preprocessor),
            ('tfidf', TfidfVectorizer(token_pattern=r'[a-zA-Z0-9]+', max_features=2000, ngram_range=(1, 2)))
        ]), 'url'),
        ('num', StandardScaler(), numeric_features)
    ]
)

# ✅ Full pipeline with SMOTE
model = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
])

# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ✅ Train model
model.fit(X_train, y_train)

# ✅ Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ✅ Save the model
joblib.dump(model, "../models/hybrid_tfidf_model.joblib")
print("\n✅ Model saved as hybrid_tfidf_model.joblib")
