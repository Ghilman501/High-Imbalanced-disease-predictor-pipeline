import pandas as pd
import numpy as np
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

def create_imbalanced_data(X, y, target_minority_ratio=0.1):
    df = pd.DataFrame(X)
    df['target'] = y
    majority = df[df['target'] == 0]
    minority = df[df['target'] == 1]
    
    n_majority = len(majority)
    n_minority_needed = int((target_minority_ratio * n_majority) / (1 - target_minority_ratio))
    minority_sampled = minority.sample(n=n_minority_needed, random_state=42)
    
    imbalanced_df = pd.concat([majority, minority_sampled]).sample(frac=1, random_state=42)
    return imbalanced_df.drop('target', axis=1).values, imbalanced_df['target'].values

# Load and prepare data
data = load_breast_cancer()
X, y = create_imbalanced_data(data.data, data.target, target_minority_ratio=0.1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Build pipeline
pipeline = ImbPipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42))
])

param_grid = {
    'classifier__n_estimators': [100],
    'classifier__max_depth': [5, 10]
}

# Train and tune
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(pipeline, param_grid, scoring='recall', cv=cv, n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f"Training complete. Best CV Recall: {grid_search.best_score_:.4f}")

# Serialize the trained pipeline
joblib.dump(best_model, 'model.joblib')
print("Model saved to ../backend/model.joblib")