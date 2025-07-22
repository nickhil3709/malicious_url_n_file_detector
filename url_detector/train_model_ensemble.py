import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Load dataset
df = pd.read_csv("../data/feature_data.csv")

# Target and Features
y = df['type']
X = df[['url', 'url_length', 'domain_length', 'num_digits', 'num_dots', 'is_https', 'has_suspicious_keywords']]

# 2. Define text and numeric features
text_features = 'url'
numeric_features = ['url_length', 'domain_length', 'num_digits', 'num_dots', 'is_https', 'has_suspicious_keywords']

# 3. Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('text', TfidfVectorizer(token_pattern=r'[a-zA-Z0-9]+', max_features=5000, ngram_range=(1,2)), text_features),
        ('num', StandardScaler(), numeric_features)
    ]
)

# 4. Base models
rf = RandomForestClassifier(n_estimators=150, random_state=42)
lr = LogisticRegression(max_iter=1000, solver='saga')

# 5. Voting Classifier (soft voting to use probabilities)
ensemble_model = VotingClassifier(
    estimators=[
        ('rf', rf),
        ('lr', lr)
    ],
    voting='soft'
)

# 6. Pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', ensemble_model)
])

# 7. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 8. Fit model
print("Training the ensemble model...")
model.fit(X_train, y_train)

# 9. Evaluate
y_pred = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n✅ Classification Report:\n", classification_report(y_test, y_pred))

# 10. Save model
joblib.dump(model, "../models/ensemble_tfidf_model.joblib")
print("\n✅ Ensemble model saved as ensemble_tfidf_model.joblib")
