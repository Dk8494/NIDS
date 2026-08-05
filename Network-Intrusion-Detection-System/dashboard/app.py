"""Streamlit dashboard for AI-powered network intrusion detection."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import IntrusionPredictor
from src.train_model import train_all_models


st.set_page_config(page_title="AI-Powered NIDS", layout="wide")
st.title("AI-Powered Network Intrusion Detection System")
st.caption("Upload network traffic data and run real-time intrusion detection using the trained best model.")

models_dir = PROJECT_ROOT / "models"
data_dir = PROJECT_ROOT / "data"
metadata_path = models_dir / "best_model_metadata.json"

with st.sidebar:
    st.header("Model Management")
    if st.button("Train / Retrain Models"):
        with st.spinner("Training models. This may take a few minutes..."):
            train_all_models(data_dir=data_dir, models_dir=models_dir)
        st.success("Model training complete.")

if not metadata_path.exists():
    st.warning("No trained model found. Click 'Train / Retrain Models' in the sidebar to generate artifacts.")
    st.stop()

predictor = IntrusionPredictor(models_dir=models_dir)
st.success(f"Loaded best model: {predictor.best_model_name}")

uploaded_file = st.file_uploader("Upload CSV network traffic data", type=["csv"])

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
    st.subheader("Input Preview")
    st.dataframe(input_df.head(20), use_container_width=True)

    if st.button("Run Intrusion Detection"):
        predictions_df = predictor.predict_dataframe(input_df)

        st.subheader("Prediction Results")
        st.dataframe(predictions_df, use_container_width=True)

        csv_bytes = predictions_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Predictions CSV",
            data=csv_bytes,
            file_name="intrusion_predictions.csv",
            mime="text/csv",
        )

        st.subheader("Attack Statistics")
        counts = predictions_df["prediction_label"].value_counts().rename_axis("class").reset_index(name="count")

        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(data=counts, x="class", y="count", palette="Set2", ax=ax)
            ax.set_title("Predicted Class Distribution")
            ax.set_xlabel("Class")
            ax.set_ylabel("Count")
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.pie(counts["count"], labels=counts["class"], autopct="%1.1f%%", startangle=90)
            ax.set_title("Attack vs Normal Ratio")
            st.pyplot(fig)
else:
    st.info("Upload a CSV file to begin intrusion detection.")
