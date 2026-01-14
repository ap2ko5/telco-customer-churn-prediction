"""ML models for customer churn prediction."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
import pickle

def train_random_forest(X, y, n_estimators=100, random_state=42):
    """Train Random Forest classifier for churn prediction."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=15,
        random_state=random_state,
        n_jobs=-1,
        class_weight='balanced'
    )
    model.fit(X, y)
    return model

def train_logistic_regression(X, y, random_state=42):
    """Train Logistic Regression classifier for churn prediction."""
    model = LogisticRegression(
        random_state=random_state,
        max_iter=1000,
        class_weight='balanced'
    )
    model.fit(X, y)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance with multiple metrics."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    return metrics, y_pred, y_pred_proba

def save_model(model, filepath):
    """Save trained model to disk."""
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filepath}")

def load_model(filepath):
    """Load trained model from disk."""
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    return model
