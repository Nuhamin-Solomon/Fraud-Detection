import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib

# -------------------------------
# Data Loading
# -------------------------------
def load_data(fraud_path, credit_path):
    """
    Load processed fraud and credit card datasets.
    """
    try:
        fraud = pd.read_csv(fraud_path)
        credit = pd.read_csv(credit_path)
        print("Data loaded successfully!")
        return fraud, credit
    except FileNotFoundError as e:
        print(f"Error: {e}")
        raise

# -------------------------------
# Feature / Target Separation
# -------------------------------
def separate_features_target(df, target_col):
    """
    Separate features and target variable.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

# -------------------------------
# Train-Test Split
# -------------------------------
def stratified_split(X, y, test_size=0.2, random_state=42):
    """
    Stratified train-test split to preserve class distribution.
    """
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

# -------------------------------
# SMOTE Oversampling
# -------------------------------
def apply_smote(X_train, y_train, random_state=42):
    """
    Apply SMOTE to balance classes on the training set only.
    """
    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    return X_res, y_res

# -------------------------------
# Logistic Regression Training
# -------------------------------
def train_logistic_regression(X_train, y_train, max_iter=1000):
    """
    Train a logistic regression model.
    """
    lr = LogisticRegression(max_iter=max_iter)
    lr.fit(X_train, y_train)
    return lr

# -------------------------------
# Random Forest Training
# -------------------------------
def train_random_forest(X_train, y_train, n_estimators=200, max_depth=10, random_state=42):
    """
    Train a Random Forest model.
    """
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, n_jobs=-1)
    rf.fit(X_train, y_train)
    return rf

# -------------------------------
# Model Evaluation
# -------------------------------
def evaluate_model(model, X_test, y_test):
    """
    Evaluate a model using AUC-PR, F1-score, and confusion matrix.
    """
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        # fallback if model does not support predict_proba
        y_prob = y_pred
    metrics = {
        "AUC-PR": average_precision_score(y_test, y_prob),
        "F1": f1_score(y_test, y_pred),
        "Confusion Matrix": confusion_matrix(y_test, y_pred)
    }
    return metrics

# -------------------------------
# Cross-Validation
# -------------------------------
def cross_validate_model(model, X, y, scoring="average_precision", cv_splits=5):
    """
    Perform Stratified K-Fold cross-validation and return mean and std of metric.
    """
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, scoring=scoring, cv=skf)
    return scores.mean(), scores.std()

# -------------------------------
# Hyperparameter Tuning: Random Forest
# -------------------------------
def grid_search_rf(X_train, y_train, param_grid=None, cv_splits=5, scoring="average_precision"):
    """
    Perform GridSearchCV on Random Forest.
    """
    if param_grid is None:
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, None]
        }
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    grid = GridSearchCV(rf, param_grid, cv=skf, scoring=scoring, n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_
