from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from pipelines.prediction_pipeline import PredictionPipleline
from fastapi.middleware.cors import CORSMiddleware
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("API_KEY")

app = FastAPI(title="Customer Churn Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = PredictionPipleline()

# Define the full schema expected by your ColumnTransformer
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

@app.get("/")
def home():
    return {"message": "Churn Prediction API Running 🚀"}

@app.post("/predict")
def predict(data: CustomerData, x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Convert Pydantic model to Dictionary
    input_dict = data.model_dump()
    
    # Pass the dictionary to the pipeline
    result = pipeline.predict(input_dict)
    return result

@app.get("/predict")
def predict_get():
    return {"message": "This is a POST endpoint. Please use POST method to send data for prediction."}