"""Streamlit app: evaluate the saved classifiers on uploaded test data
(Online Shoppers Purchasing Intention dataset)."""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from ml_utils import METRIC_NAMES, MODEL_NAMES, evaluate_pipeline, model_path
# ---------- Paths (relative to this file, so it works locally and on Streamlit Cloud)
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model"
SAMPLE_CSV = BASE_DIR / "test_data.csv"
CLASS_LABELS = ["No purchase", "Purchase"]

st.set_page_config(
    page_title="Online Shopper Purchase Prediction",
    page_icon="🛒",
    layout="wide",
)


# ---------- Cached loaders (run once, then reused on every interaction)
@st.cache_data
def load_model_info():
    return json.loads((MODEL_DIR / "model_info.json").read_text())


@st.cache_resource
def load_model(name):
    return joblib.load(model_path(MODEL_DIR, name))


info = load_model_info()
FEATURES = info["feature_columns"]
TARGET = info["target"]

# ---------- Header
st.title("🛒 Online Shopper Purchase Prediction")
st.caption("Machine Learning, Assignment 2 · BITS Pilani WILP · Nandini Maheshwari")
st.write(
    "Upload test data to evaluate five pre-trained classifiers that predict whether "
    "an online shopping session ends in a purchase. Models are loaded from disk; "
    "nothing is retrained on the uploaded data."
)

# ---------- Sidebar
with st.sidebar:
    st.header("About")
    st.markdown(
        "**Dataset:** Online Shoppers Purchasing Intention (UCI #468, via Kaggle). "
        f"{len(FEATURES)} features, binary target `{TARGET}`.\n\n"
        "**Models:** Logistic Regression, Decision Tree, KNN, "
        "Gaussian Naive Bayes, SVM (RBF, calibrated).\n\n"
        "**Trained with:** scikit-learn "
        f"{info['sklearn_version']} on {info['train_rows']:,} sessions."
    )
    st.download_button(
        "⬇️ Download sample test data",
        data=SAMPLE_CSV.read_bytes(),
        file_name="test_data.csv",
        mime="text/csv",
        help="The held-out 20% test split used for the results in the README.",
    )

# ---------- 1. Upload
st.subheader("1. Upload test data")
uploaded = st.file_uploader(
    "CSV with all feature columns and the target column",
    type="csv",
)
if uploaded is None:
    st.info("Upload a CSV to begin, or download the sample from the sidebar.")
    st.stop()

df = pd.read_csv(uploaded)

# ---------- Validate columns and target
missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {', '.join(missing)}")
    st.stop()

X = df[FEATURES]
y = df[TARGET]
if y.dtype == bool:  # the raw dataset stores Revenue as True/False
    y = y.astype(int)
if not set(y.unique()) <= {0, 1}:
    st.error(f"`{TARGET}` must contain only 0/1 or True/False values.")
    st.stop()
if y.nunique() < 2:
    st.error("The file needs both purchase and non-purchase rows to compute AUC and MCC.")
    st.stop()

# ---------- Preview
c1, c2, c3 = st.columns(3)
c1.metric("Rows", f"{len(df):,}")
c2.metric("Features used", len(FEATURES))
c3.metric("Purchase rate", f"{y.mean():.1%}")
with st.expander("Preview first 10 rows"):
    st.dataframe(df.head(10))

st.divider()

# ---------- 2. Choose a model
st.subheader("2. Choose a model")
model_name = st.selectbox(
    "Model",
    MODEL_NAMES,
    index=MODEL_NAMES.index("Decision Tree"),  # best overall on our test split
    help="All five models were trained on the same 80% training split and saved to disk.",
)

with st.spinner(f"Loading {model_name} and predicting…"):
    model = load_model(model_name)
    metrics, y_pred = evaluate_pipeline(model, X, y)

# ---------- 3. Evaluation metrics
st.subheader(f"3. Evaluation metrics: {model_name}")
metric_cols = st.columns(len(METRIC_NAMES))
for col, name in zip(metric_cols, METRIC_NAMES):
    col.metric(name, f"{metrics[name]:.4f}")

baseline = 1 - y.mean()
st.caption(
    f"Positive class = purchase. Precision, Recall and F1 refer to purchases; "
    f"AUC uses predicted probabilities. Baseline accuracy on this file "
    f"(always predicting 'no purchase'): {baseline:.4f}."
)

with st.expander("How to read these metrics"):
    st.markdown(
        "- **Accuracy:** share of all sessions classified correctly. Misleading here, "
        "because about 84% of sessions are non-purchases.\n"
        "- **AUC:** how well the model ranks buyers above non-buyers (0.5 = random).\n"
        "- **Precision:** of the sessions predicted as purchases, how many really were.\n"
        "- **Recall:** of the real purchases, how many the model caught.\n"
        "- **F1:** balance of precision and recall.\n"
        "- **MCC:** correlation between predictions and truth (−1 to 1), "
        "reliable even with imbalanced classes."
    )

st.divider()


# ---------- 4. Confusion matrix and classification report
st.subheader(f"4. Confusion matrix and classification report: {model_name}")
left, right = st.columns(2)

with left:
    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="BuGn", cbar=False,
        xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
    plt.close(fig)

    tn, fp, fn, tp = cm.ravel()
    st.caption(
        f"Caught {tp} of {tp + fn} buyers ({tp / (tp + fn):.0%}), "
        f"missed {fn}, with {fp} false alarms."
    )

with right:
    report = classification_report(
        y, y_pred, labels=[0, 1], target_names=CLASS_LABELS,
        output_dict=True, zero_division=0,
    )
    report.pop("accuracy", None)  # already shown in the metrics row above
    report_df = pd.DataFrame(report).T.round(4)
    report_df["support"] = report_df["support"].astype(int)
    st.dataframe(report_df)

    predictions_csv = df.assign(Predicted=y_pred).to_csv(index=False).encode()
    st.download_button(
        "⬇️ Download predictions (CSV)",
        data=predictions_csv,
        file_name=f"predictions_{model_name.lower().replace(' ', '_')}.csv",
        mime="text/csv",
    )

st.divider()

# ---------- 5. Compare all models on this file
st.subheader("5. Compare all models on this file")
if st.toggle("Evaluate all five models on the uploaded data"):
    with st.spinner("Evaluating all models…"):
        all_results = {
            name: evaluate_pipeline(load_model(name), X, y)[0] for name in MODEL_NAMES
        }
    comparison = pd.DataFrame(all_results).T[METRIC_NAMES].round(4)
    comparison.index.name = "Model"
    st.dataframe(comparison)

    best = comparison["MCC"].idxmax()
    st.success(f"Best on MCC for this file: **{best}** ({comparison.loc[best, 'MCC']:.4f})")
    st.bar_chart(comparison[["F1", "MCC"]], stack=False)

st.caption("Models trained offline in model_training.ipynb; this app only loads and evaluates them.")