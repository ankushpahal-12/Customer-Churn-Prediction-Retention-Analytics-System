# Customer Churn Prediction & Retention Analytics System

An end-to-end **Machine Learning production-grade pipeline** that predicts telecom customer churn using advanced preprocessing, dimensionality reduction (PCA), ensemble learning, and model optimization techniques.

Built with modular architecture, logging, exception handling, and deployment-ready artifacts.

---

## Problem Statement

Customer churn prediction helps telecom companies identify customers who are likely to discontinue their services.

Early identification allows businesses to:

- Improve retention strategies  
- Reduce revenue loss  
- Optimize marketing spend  
- Increase customer lifetime value  

This system automates the entire ML workflow — from raw data ingestion to model saving.

---

# 🛠 Tech Stack

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python"/>
  <img src="https://img.shields.io/badge/Pandas-Data%20Processing-black?logo=pandas"/>
  <img src="https://img.shields.io/badge/NumPy-Numerical%20Computing-blue?logo=numpy"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-ML%20Framework-orange?logo=scikit-learn"/>
  <img src="https://img.shields.io/badge/XGBoost-Gradient%20Boosting-red"/>
  <img src="https://img.shields.io/badge/LightGBM-Boosting-green"/>
  <img src="https://img.shields.io/badge/Matplotlib-Visualization-blue"/>
  <img src="https://img.shields.io/badge/Seaborn-Statistical%20Plots-teal"/>
</p>

---

#  System Architecture

```
Raw Data → Data Ingestion → Preprocessing → Feature Engineering → PCA →
Model Training → Hyperparameter Tuning → Ensemble →
Evaluation → Model Selection → Artifact Saving → Deployment Ready
```

---

#  Project Structure

```
customer_churn_project/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── artifacts/
│   ├── models/
│   └── plots/
│
├── src/
│   ├── data_ingestion.py
│   ├── data_analysis.py
│   └── data_transform.py
│
├── model/
│   ├── model_trainer.py
│   ├── model_evaluation.py
│   ├── model_selector.py
│   └── model_saver.py
│
├── pipelines/
│   └── training_pipeline.py
│
├── utils/
│   ├── logger.py
│   └── exception.py
│
├── config.yaml
├── requirements.txt
├── README.md
└── app.py (optional deployment)
```

---

#  Key Features

✔ Modular ML pipeline  
✔ PCA dimensionality reduction  
✔ Hyperparameter tuning  
✔ Stacking ensemble learning  
✔ Class imbalance handling  
✔ Threshold optimization  
✔ Logging & custom exception handling  
✔ Model artifact saving  
✔ Deployment-ready structure  

---

#  Machine Learning Workflow

## 1️⃣ Data Ingestion
- Reads raw CSV dataset
- Saves processed copy
- Returns DataFrame

## 2️⃣ Data Transformation
- Missing value handling  
- Categorical encoding (OneHotEncoder)  
- Feature scaling  
- Train-Test split  
- PCA dimensionality reduction  

## 3️⃣ Model Training
Models implemented:

- Logistic Regression (GridSearchCV)
- Random Forest
- XGBoost (RandomizedSearchCV)
- LightGBM
- Stacking Ensemble

## 4️⃣ Model Evaluation
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix
- ROC Curve
- Threshold Optimization

## 5️⃣ Model Selection
Best model selected based on **ROC-AUC score**.

## 6️⃣ Artifact Saving
Saved objects:
- Best Model (`.pkl`)
- Preprocessor
- PCA transformer
- Metrics JSON
- Evaluation plots

---

# 📊 Final Model Performance

| Metric | Score |
|--------|--------|
| Accuracy | ≈ 79% |
| Precision | ≈ 59% |
| Recall | ≈ 68% |
| F1-Score | ≈ 0.63 |
| ROC-AUC | ≈ 0.846 |

ROC-AUC stability across experiments indicates strong ranking capability of the model.

---

#  Evaluation Metrics Explained

- **Accuracy** → Overall correctness  
- **Precision** → Correct churn predictions out of predicted churn  
- **Recall** → Ability to detect actual churn customers  
- **F1-score** → Balance between Precision & Recall  
- **ROC-AUC** → Model’s ability to rank churn vs non-churn  

---

#  Installation Guide

### 1️⃣ Clone Repository

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

---

# ▶️ Run Training Pipeline

```bash
python -m pipelines.training_pipeline
```

This will:

- Load data  
- Transform features  
- Apply PCA  
- Train models  
- Evaluate performance  
- Select best model  
- Save artifacts  

---

#  Deployment Ready

Artifacts saved inside:

```
artifacts/models/
```

These can be directly integrated into:
- Flask API
- FastAPI
- Streamlit
- Docker container
- Cloud deployment (AWS / GCP / Azure)

---

#  Future Improvements

- SHAP explainability integration  
- MLflow experiment tracking  
- CI/CD pipeline  
- Docker containerization  
- Real-time inference API  

---

#  Author

**Ankush**  
B.Tech CSE (AI & ML Specialization)

---

⭐ If you found this project helpful, consider giving it a star!
