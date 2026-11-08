"""shared helpers used by both the training nb & streamlit app."""
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

POSITIVE_CLASS = 1  # 1 = session ended in a purchase
METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


def compute_metrics(y_true, y_pred, y_proba):
    """return the 6 metrics for a binary classifier.

    y_proba is the predicted probability of the positive class.
    precision, recall & F1 are reported for the positive class (purchase).
    """
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba),
        "Precision": precision_score(y_true, y_pred, pos_label=POSITIVE_CLASS, zero_division=0),
        "Recall": recall_score(y_true, y_pred, pos_label=POSITIVE_CLASS, zero_division=0),
        "F1": f1_score(y_true, y_pred, pos_label=POSITIVE_CLASS, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def evaluate_pipeline(pipe, X, y):
    """predict with a fitted pipeline & return (metrics dict, predictions)."""
    y_pred = pipe.predict(X)
    pos_idx = list(pipe.classes_).index(POSITIVE_CLASS)
    y_proba = pipe.predict_proba(X)[:, pos_idx]
    return compute_metrics(y, y_pred, y_proba), y_pred


MODEL_NAMES = [
    "Logistic Regression",
    "Decision Tree",
    "KNN",
    "Naive Bayes",
    "Support Vector Machine",
]


def model_path(model_dir, name):
    """File path for a saved model, e.g. model/decision_tree.joblib."""
    return f"{model_dir}/{name.lower().replace(' ', '_')}.joblib"
