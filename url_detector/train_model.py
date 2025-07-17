import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
df = pd.read_csv("../data/feature_data.csv")

# OPTIONAL: speed up experimentation on very large datasets
# df = df.sample(100_000, random_state=42)
# Target & Features
y = df['type']
X = df[['url', 'url_length', 'domain_length', 'num_digits', 'num_dots', 'is_https', 'has_suspicious_keywords']]

text_feature = 'url'
numeric_features = ['url_length', 'domain_length', 'num_digits', 'num_dots', 'is_https', 'has_suspicious_keywords']

# Preprocessor: TF-IDF for URL text + scale numeric features
# NOTE: with_mean=False is required when numeric + sparse tf-idf are combined.
preprocessor = ColumnTransformer(
    transformers=[
        ('text', TfidfVectorizer(
            token_pattern=r'[a-zA-Z0-9]+',
            max_features=5000,
            ngram_range=(1, 2),
            lowercase=True
        ), text_feature),
        ('num', StandardScaler(with_mean=False), numeric_features),
    ],
    sparse_threshold=0.3  # keep sparse when mostly text
)

# Model pipeline
# LogisticRegression (saga) handles large sparse high-dim input well.
# class_weight='balanced' helps cope with any remaining imbalance.
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(
        solver='saga',
        max_iter=1000,
        class_weight='balanced',
        n_jobs=-1  # parallel where applicable
    ))
])

# Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "../models/tfidf_hybrid_model.joblib")
print("\n✅ Model saved as tfidf_hybrid_model.joblib")
