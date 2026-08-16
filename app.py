import sys
from pathlib import Path
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
sys.path.insert(0, str(MODEL_DIR))

st.set_page_config(page_title="ML Classification Assignment 2", page_icon="🧠", layout="wide")
st.title("🧠 ML Classification Model Comparison")
st.caption("UCI Breast Cancer Wisconsin (Diagnostic) — ML Assignment 2")

@st.cache_data
def get_dataset():
    data = load_breast_cancer(as_frame=True)
    X = data.data.copy()
    y = (data.target == 0).astype(int)  # malignant = 1
    return X, y

X, y = get_dataset()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

@st.cache_resource
def build_models():
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=5000, random_state=42))
        ]),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "kNN": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=7))
        ]),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=42, n_jobs=-1
        )
    }

models = build_models()
for model in models.values():
    model.fit(X_train, y_train)

with st.sidebar:
    st.header("Model Selection")
    selected_model = st.selectbox("Choose a model", list(models.keys()))
    uploaded_file = st.file_uploader(
        "Upload test CSV",
        type=["csv"],
        help="Upload the supplied test_data.csv. It must contain the 30 feature columns."
    )
    search_clicked = st.button(
        "🔍 Submit",
        type="primary",
        use_container_width=True
    )

default_test = X_test.copy()
default_test["Diagnosis"] = y_test.map({0: "Benign", 1: "Malignant"})
# Use uploaded file only after Submit button is clicked
if search_clicked:

    if uploaded_file is None:
        st.warning("Please upload a CSV file before clicking Submit.")
        st.stop()

    df = pd.read_csv(uploaded_file)

else:
    # Before submitting, show default test data
    df = default_test.copy()
    
# df = pd.read_csv(uploaded_file) if uploaded_file else default_test

required_features = list(X.columns)
missing = [c for c in required_features if c not in df.columns]

if missing:
    st.error("Required feature columns are missing.")
    st.write(missing)
    st.stop()

X_input = df[required_features]
model = models[selected_model]
pred = model.predict(X_input)
prob = model.predict_proba(X_input)[:, 1]

st.subheader("Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)

st.markdown(f"### Predictions — {selected_model}")
prediction_df = pd.DataFrame({
    "Prediction": pd.Series(pred).map({0: "Benign", 1: "Malignant"}),
    "Malignant Probability": prob
})
st.dataframe(prediction_df, use_container_width=True)

if "Diagnosis" in df.columns:
    y_true = df["Diagnosis"].map({"Benign": 0, "Malignant": 1})

    if y_true.notna().all():
        y_true = y_true.astype(int)

        metrics = {
            "Accuracy": accuracy_score(y_true, pred),
            "AUC": roc_auc_score(y_true, prob),
            "Precision": precision_score(y_true, pred, zero_division=0),
            "Recall": recall_score(y_true, pred, zero_division=0),
            "F1 Score": f1_score(y_true, pred, zero_division=0),
            "MCC": matthews_corrcoef(y_true, pred)
        }

        st.markdown("### Evaluation Metrics")
        cols = st.columns(6)
        for col, (name, value) in zip(cols, metrics.items()):
            col.metric(name, f"{value:.4f}")

        st.markdown("### Confusion Matrix")
        cm = confusion_matrix(y_true, pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(1.8, 1.5))
        ax.imshow(cm)
        ax.set_xticks([0, 1], ["Benign", "Malignant"])
        ax.set_yticks([0, 1], ["Benign", "Malignant"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center")
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("### Classification Report")
        report = classification_report(
            y_true, pred,
            target_names=["Benign", "Malignant"],
            zero_division=0,
            output_dict=True
        )
        st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

st.subheader("Comparison of All Five Models")
comparison = []
for name, mdl in models.items():
    p = mdl.predict(X_test)
    pr = mdl.predict_proba(X_test)[:, 1]
    comparison.append({
        "ML Model Name": name,
        "Accuracy": accuracy_score(y_test, p),
        "AUC": roc_auc_score(y_test, pr),
        "Precision": precision_score(y_test, p, zero_division=0),
        "Recall": recall_score(y_test, p, zero_division=0),
        "F1": f1_score(y_test, p, zero_division=0),
        "MCC": matthews_corrcoef(y_test, p)
    })

st.dataframe(pd.DataFrame(comparison).style.format({
    "Accuracy": "{:.4f}", "AUC": "{:.4f}", "Precision": "{:.4f}",
    "Recall": "{:.4f}", "F1": "{:.4f}", "MCC": "{:.4f}"
}), use_container_width=True)

# ---------------------------------------------------------
# Model Performance Observations
# ---------------------------------------------------------

st.subheader("Model Performance Observations")

observations = {
    "Logistic Regression":
        "Performs very well on this dataset with strong accuracy, precision, recall, and F1-score. It is effective because the dataset is relatively well separated.",

    "Decision Tree":
        "Provides good classification performance and is easy to interpret. However, a single decision tree can be more sensitive to the training data and may have slightly lower generalization performance.",

    "kNN":
        "Provides good classification performance after feature scaling. Its performance depends on the choice of k and the distance between samples.",

    "Naive Bayes":
        "Provides competitive performance and is computationally efficient. Its performance may be affected by the assumption that features are conditionally independent.",

    "Random Forest (Ensemble)":
        "Provides strong and stable performance by combining multiple decision trees. The ensemble approach generally reduces overfitting and improves generalization."
}

observation_table = []

for model_name in [
    "Logistic Regression",
    "Decision Tree",
    "kNN",
    "Naive Bayes",
    "Random Forest (Ensemble)"
]:
    observation_table.append({
        "ML Model Name": model_name,
        "Observation about model performance": observations[model_name]
    })

observation_df = pd.DataFrame(observation_table)


# Use HTML table inside a horizontally scrollable container
html_table = observation_df.to_html(
    index=False,
    escape=False
)

html_table = html_table.replace(
    "<th>",
    '<th style="text-align: center; padding: 10px; white-space: nowrap;">'
)

# Style table cells
html_table = html_table.replace(
    "<td>",
    '<td style="padding: 10px; vertical-align: top;">'
)

st.markdown(
    f"""
    <div style="
        overflow-x: auto;
        width: 100%;
        border: 1px solid #ddd;
        border-radius: 5px;
    ">
        <div style="min-width: 1100px;">
            {html_table}
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# Overall Winner
# ---------------------------------------------------------

st.subheader("Overall Winner")

comparison_df = pd.DataFrame(comparison)

# Find model with highest Recall
winner_row = comparison_df.loc[
    comparison_df["Recall"].idxmax()
]

winner = winner_row["ML Model Name"]
winner_recall = winner_row["Recall"]

st.success(
    f"🏆 {winner} is the overall winner for this dataset "
    f"based on the highest Recall ({winner_recall:.4f})."
)

st.write(
    "Recall is used as the primary metric for selecting the overall winner."
)

st.info("Positive class = Malignant. All five models use the same stratified 20% test split (random_state=42).")
