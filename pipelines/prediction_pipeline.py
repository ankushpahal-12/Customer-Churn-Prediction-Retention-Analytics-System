import pickle
import numpy as np
import pandas as pd

class PredictionPipleline:
    def __init__(self):
        with open("artifacts/models/XGBoost_model.pkl", "rb") as f:
            self.model = pickle.load(f)
        
        with open("artifacts/models/preprocessor.pkl", "rb") as f:
            self.preprocessor = pickle.load(f)
        
        with open("artifacts/models/pca.pkl", "rb") as f:
            self.pca = pickle.load(f)
    
    def predict(self, input_data):
        # transform input dictionary to dataframe
        df = pd.DataFrame([input_data])
        
        # Feature Engineering (replicating logic from data_transform.py)
        
        # 1. tenure_group
        df["tenure_group"] = pd.cut(
            df["tenure"],
            bins=[0, 12, 24, 48, 72],
            labels=["0-1yr", "1-2yr", "2-4yr", "4-6yr"]
        )
        
        # 2. avg_monthly_cost
        df["avg_monthly_cost"] = (
            df["TotalCharges"] / (df["tenure"] + 1)
        )
        
        # 3. high_monthly_charge
        # Note: Median value should ideally be loaded from training artifacts.
        # Using 70.35 as standard median for Telco Churn dataset or calculated from data.
        MEDIAN_MONTHLY_CHARGES = 70.35 
        df["high_monthly_charge"] = (
            df["MonthlyCharges"] > MEDIAN_MONTHLY_CHARGES
        ).astype(int)
        
        # 4. long_term_customer
        df["long_term_customer"] = (
            df["tenure"] > 24
        ).astype(int)
        
        # 5. is_month_to_month
        df["is_month_to_month"] = (
            df["Contract"] == "Month-to-month"
        ).astype(int)
        
        # Preprocessing
        processed = self.preprocessor.transform(df)
        
        # PCA
        processed_pca = self.pca.transform(processed)
        
        # Prediction
        prediction = self.model.predict(processed_pca)
        
        probability = None
        if hasattr(self.model, "predict_proba"):
            probability = self.model.predict_proba(processed_pca)[0][1]
            
        return {
            "prediction": int(prediction[0]),
            "probability": float(probability) if probability is not None else None
        }