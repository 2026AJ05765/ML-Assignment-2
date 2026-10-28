
import streamlit as st
import pandas as pd
import joblib
import os

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

st.set_page_config(page_title="ML Classifier Comparison", layout="wide")

st.title("Machine Learning Classification App")
st.write("Breast Cancer Wisconsin Diagnostic Dataset")
st.write("Select a trained model and upload test data to evaluate its performance.")

model_files = {
    "Logistic Regression": "models/logistic_regression.joblib",
    "Decision Tree": "models/decision_tree.joblib",
    "KNN": "models/knn.joblib",
    "Naive Bayes": "models/naive_bayes.joblib",
    "Support Vector Machine": "models/support_vector_machine.joblib"
}

selected_model = st.selectbox("Select a classification model", list(model_files.keys()))

uploaded_file = st.file_uploader("Upload test CSV", type=["csv"])

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)

    if "target" not in test_df.columns:
        st.error("CSV must contain a target column for evaluation.")
        st.stop()

    X_test_app = test_df.drop(columns=["target"])
    y_test_app = test_df["target"]

    if not os.path.exists(model_files[selected_model]):
        st.error("Model file not found. Make sure the models folder is included.")
        st.stop()

    model = joblib.load(model_files[selected_model])

    if selected_model in [
        "Logistic Regression",
        "KNN",
        "Support Vector Machine"
    ]:
        scaler = joblib.load("models/scaler.joblib")
        X_test_app = scaler.transform(X_test_app)

    predictions = model.predict(X_test_app)

    st.subheader("Evaluation Metrics")

    metrics = {
        "Accuracy": accuracy_score(y_test_app, predictions),
        "Precision": precision_score(y_test_app, predictions, zero_division=0),
        "Recall": recall_score(y_test_app, predictions, zero_division=0),
        "F1 Score": f1_score(y_test_app, predictions, zero_division=0),
        "MCC": matthews_corrcoef(y_test_app, predictions)
    }

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test_app)[:, 1]
        metrics["AUC"] = roc_auc_score(y_test_app, probabilities)

    st.dataframe(
        pd.DataFrame([metrics]).round(4),
        use_container_width=True
    )

    st.subheader("Confusion Matrix")
    st.dataframe(
        pd.DataFrame(
            confusion_matrix(y_test_app, predictions),
            index=["Actual 0", "Actual 1"],
            columns=["Predicted 0", "Predicted 1"]
        )
    )

    st.subheader("Classification Report")
    st.text(classification_report(y_test_app, predictions, zero_division=0))
