# Customer Churn Prediction System

An end-to-end Machine Learning project to predict customer churn using advanced preprocessing, PCA, ensemble learning, and model optimization techniques.

This project includes:
- Data ingestion
- Data preprocessing
- Feature engineering
- PCA dimensionality reduction
- Model training (Logistic, Random Forest, XGBoost, LightGBM)
- Stacking ensemble
- Model evaluation
- Model saving
- Logging and exception handling
- Ready for deployment

---

## Project Overview

Customer churn prediction helps telecom companies identify customers who are likely to leave the service.

This system:
- Processes raw data
- Performs feature engineering
- Applies preprocessing pipelines
- Uses PCA for dimensionality reduction
- Trains multiple ML models
- Selects the best model
- Saves model artifacts for deployment

---

## 🛠 Tech Stack

- Python 3.10+
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- LightGBM
- Matplotlib
- Seaborn

---

## 📂 Project Structure
customer_churn_project/
│
├── data/                        # Raw & processed datasets
│   ├── raw/
│   │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   │
│   └── processed/
│       └── processed_telco_data.csv
│
├── artifacts/                   #  Saved Model artifacts and plots  
│   ├── models/
│   │   ├── best_model.pkl
│   │   ├── preprocessor.pkl
│   │   ├── pca.pkl
│   │   └── metrics.json
│   │
│   └── plots/
│       ├── confusion_matrix.png
│       ├── roc_curve.png
│       └── feature_importance.png
│
├── src/                         # Data ingestion & preprocessing
│   ├── __init__.py
│   ├── data_ingestion.py
│   ├── data_analysis.py
│   └── data_transform.py
│
├── model/                        # Model training & evaluation
│   ├── __init__.py
│   ├── model_trainer.py
│   ├── model_evaluation.py
│   ├── model_selector.py
│   └── model_saver.py
│
├── pipelines/                    # End to end model pipeline
│   ├── __init__.py
│   └── training_pipeline.py
│
├── utils/                        # Utility functions
│   ├── __init__.py
│   ├── logger.py
│   └── exception.py
│
├── config.yaml                    # Configuration file
├── requirements.txt               # Dependencies file
├── README.md                      # Read me file
└── app.py   (optional for deployment)
## 📁 File Details

### 🔹 data_ingestion.py
- Reads raw dataset
- Saves processed copy
- Returns DataFrame

### 🔹 data_transform.py
- Feature engineering
- Handling missing values
- Encoding categorical variables
- Scaling numeric features
- PCA dimensionality reduction
- Train-test split

### 🔹 model_trainer.py
- Hyperparameter tuning
- Logistic Regression (GridSearch)
- XGBoost (RandomizedSearch)
- Random Forest
- LightGBM
- Stacking Ensemble

### 🔹 model_evaluation.py
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix
- ROC Curve
- Threshold optimization

### 🔹 model_selector.py
- Compares all models
- Selects best model based on ROC-AUC

### 🔹 model_saver.py
- Saves:
  - Best model (.pkl)
  - Preprocessor
  - PCA object
  - Metrics JSON

### 🔹 training_pipeline.py
- Runs complete end-to-end pipeline

---

## 📥 Installation Guide

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/ankushpahal-12/Customer-Churn-Prediction-Retention-Analytics-System
cd customer_churn_project
```
### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```
Activate:
Windows

```bash
venv\Scripts\activate
```
Mac/Linux

```bash
source venv/bin/activate
```
### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

▶️ How To Run The Project

From project root folder:

```bash
python -m pipelines.training_pipeline
---

This will:

    Load data
    Transform data
    Apply PCA
    Train models
    Evaluate models
    Select best model
    Save artifacts

📊 Model Performance (Final)

    Balanced Tuned Model:
    Accuracy ≈ 79%
    Precision ≈ 59%
    Recall ≈ 68%
    F1 Score ≈ 0.63
    ROC-AUC ≈ 0.846
    The ROC-AUC remains stable across multiple tuning experiments, indicating strong model ranking capability.

📈 Evaluation Metrics Explained

    Accuracy – Overall correct predictions
    Precision – Correct churn predictions
    Recall – Ability to detect churn customers
    F1-score – Balance between precision and recall
    ROC-AUC – Model ranking strength

🔍 Key ML Concepts Used

Feature Engineering
    OneHotEncoding
    Scaling
    PCA
    Hyperparameter Tuning
    Class Imbalance Handling
    Ensemble Learning
    Stacking
    Threshold Optimization
👨‍💻 Author

Ankush
B.Tech CSE (AI & ML Specialization)


