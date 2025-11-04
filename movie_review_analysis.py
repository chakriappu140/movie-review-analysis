import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier # Import the advanced model

# --- 1. Load and Prepare Data ---
try:
    df = pd.read_csv('IMDB Dataset.csv')
    print("IMDB Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: IMDB Dataset.csv not found.")
    exit()

# --- 2. Encode Target & Split Data ---
X = df['review']
y_raw = df['sentiment']

le = LabelEncoder()
y = le.fit_transform(y_raw) # y is now 0 or 1
# le.classes_ will show you ['negative', 'positive']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Data split and encoded complete.")

# --- 3. Baseline Model: Logistic Regression ---
pipeline_logreg = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=50000)),
    ('model', LogisticRegression(max_iter=1000, random_state=42))
])
print("\nTraining Baseline Logistic Regression...")
pipeline_logreg.fit(X_train, y_train)
y_pred_logreg = pipeline_logreg.predict(X_test)
print("--- Logistic Regression Results ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_logreg):.4f}")

# --- 4. New Model: Tuned LightGBM (LGBM) ---
print("\nStarting Hyperparameter Tuning for LightGBM...")
# Create the pipeline
pipeline_lgbm = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('model', LGBMClassifier(random_state=42, n_jobs=-1))
])

# Define the parameter grid to search
# We'll tune the vectorizer and the model at the same time
param_dist_lgbm = {
    'tfidf__max_features': [20000, 40000, 60000],          # Vocabulary size
    'tfidf__ngram_range': [(1, 1), (1, 2)],                # Include bi-grams (two-word phrases)
    'model__n_estimators': [100, 200],                     # Number of trees
    'model__learning_rate': [0.05, 0.1]                    # Step size
}

# Use RandomizedSearchCV (faster than GridSearchCV)
random_search_lgbm = RandomizedSearchCV(
    estimator=pipeline_lgbm,
    param_distributions=param_dist_lgbm,
    n_iter=5, # 5 random combinations is fast but effective
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

random_search_lgbm.fit(X_train, y_train)

# Get the best tuned pipeline
best_lgbm_pipeline = random_search_lgbm.best_estimator_
print("\n--- Tuned LightGBM Results ---")
print(f"Best CV Accuracy: {random_search_lgbm.best_score_:.4f}")
print("Best Parameters Found:")
print(random_search_lgbm.best_params_)

# 5. Final Evaluation of Best Model on Test Set
y_pred_lgbm_tuned = best_lgbm_pipeline.predict(X_test)
print("\n--- Tuned LightGBM Test Set Performance ---")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred_lgbm_tuned):.4f}")
print(classification_report(y_test, y_pred_lgbm_tuned, target_names=['Negative (0)', 'Positive (1)']))

# --- 6. Save the BEST model for deployment ---
if accuracy_score(y_test, y_pred_lgbm_tuned) > accuracy_score(y_test, y_pred_logreg):
    print("\nTuned LGBM is the best model. Saving...")
    with open('movie_review_model.pkl', 'wb') as f:
        pickle.dump(best_lgbm_pipeline, f)
else:
    print("\nLogistic Regression is the best model. Saving...")
    with open('movie_review_model.pkl', 'wb') as f:
        pickle.dump(pipeline_logreg, f)

print("Best model saved to 'movie_review_model.pkl'")