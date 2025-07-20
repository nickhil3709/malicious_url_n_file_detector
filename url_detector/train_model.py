import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ✅ Load dataset
df = pd.read_csv("../data/feature_data.csv")

# ✅ Target & Features
y = df['type']  # Multi-class: benign, phishing, malware, defacement
X = df[['url', 'url_length', 'domain_length', 'num_digits', 'num_dots', 'is_https', 'has_suspicious_keywords']]

# ✅ Define feature groups
text_feature = 'url'
numeric_features = ['url_length', 'domain_length', 'num_digits', 'num_dots', 'is_https', 'has_suspicious_keywords']

# ✅ Preprocessor: TF-IDF for URL + StandardScaler for numeric
preprocessor = ColumnTransformer(
    transformers=[
        ('text', TfidfVectorizer(token_pattern=r'[a-zA-Z0-9]+', max_features=2000), 'url'),
        ('num', StandardScaler(), numeric_features)
    ]
)

# ✅ Build pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42,n_jobs=-1, class_weight='balanced'))
])

# ✅ Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ✅ Train model
print("Training the model... (this may take a while)")
model.fit(X_train, y_train)

# ✅ Evaluate
y_pred = model.predict(X_test)
print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n✅ Classification Report:\n", classification_report(y_test, y_pred))

# ✅ Save model
joblib.dump(model, "../models/multi_class_tfidf_rf.joblib")
print("\n✅ Model saved as multi_class_tfidf_rf.joblib")
