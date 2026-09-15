# AI-Powered Network Intrusion Detection System

An end-to-end machine learning and deep learning project for detecting network intrusions using the NSL-KDD dataset. The system trains multiple models, compares their performance, selects the best candidate, and exposes the final model through a Streamlit dashboard for batch prediction.

## Overview

This project includes:

- Data preprocessing and feature engineering for NSL-KDD
- Training of multiple intrusion detection models
- Evaluation with standard classification metrics
- Best-model selection based on F1 score
- Saved model artifacts and metrics for deployment
- A Streamlit app for uploading CSV data and generating predictions

## Project Structure

```text
Network-Intrusion-Detection-System/
├── data/
│   ├── KDDTrain+.txt
│   └── KDDTest+.txt
├── dashboard/
│   └── app.py
├── models/
│   ├── best_model.joblib
│   ├── best_model_metadata.json
│   ├── deep_neural_network.keras
│   ├── logistic_regression.joblib
│   ├── metrics_summary.csv
│   ├── preprocessor.joblib
│   ├── random_forest.joblib
│   └── xgboost.joblib
├── notebooks/
├── scripts/
│   └── setup.sh
├── src/
│   ├── evaluate_model.py
│   ├── feature_engineering.py
│   ├── predict.py
│   ├── preprocessing.py
│   └── train_model.py
├── .gitignore
├── main.py
├── README.md
├── requirements.txt
└── setup.py
```

## Dataset and Task

The project uses the NSL-KDD dataset, a cleaned and improved version of KDD'99 designed for intrusion detection research.

- Input features: 41 network traffic attributes
- Label style: binary classification
- Normal traffic is mapped to 0
- Attack traffic is mapped to 1
- The pipeline automatically loads train and test data from the data folder when available

## Models Included

The training pipeline evaluates and saves the following models:

- Logistic Regression
- Random Forest
- XGBoost
- Deep Neural Network (TensorFlow/Keras)

Each model is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion matrix plots

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd Network-Intrusion-Detection-System
```

### 2. Set up the environment

```bash
bash scripts/setup.sh
source venv/bin/activate
```

This script creates a virtual environment and installs the dependencies from requirements.txt.

## Quick Start

### Train all models

```bash
python main.py
```

This runs the full workflow and generates artifacts inside the models folder.

### Start the dashboard

```bash
streamlit run dashboard/app.py
```

Then upload a CSV file containing network traffic records and click Run Intrusion Detection.

## Prediction API

You can also run predictions directly from Python:

```python
from src.predict import predict_from_csv

predict_from_csv(
    "input.csv",
    "predictions.csv",
    models_dir="models",
)
```

## Output Artifacts

After training, the project stores the following files in the models directory:

- metrics_summary.csv
- model_comparison_f1.png
- best_model_metadata.json
- confusion matrices for each trained model
- best_model.joblib for the selected classical model
- deep_neural_network.keras for the selected neural model
- preprocessor.joblib for feature transformation

## Dashboard Features

The Streamlit app supports:

- Uploading network traffic data in CSV format
- Running intrusion detection with the best trained model
- Viewing row-level predictions and probabilities
- Downloading prediction results as CSV
- Visualizing attack distribution with charts

## Notes

- The code assumes the dataset files are present in the data folder.
- If the dataset is missing, the system should be updated to download it automatically in a future extension.
- The best model is selected using the highest F1 score.

## Future Improvements

- Add multiclass attack classification
- Support real-time packet monitoring
- Add explainability tools such as SHAP
- Add automated tests and CI workflows
- Package the project for deployment as an API or service

