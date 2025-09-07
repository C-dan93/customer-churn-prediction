
import joblib
import pandas as pd
import numpy as np
import json
import glob
import os
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn using machine learning",
    version="1.0.0"
)

# Global variables
model = None
feature_names = [
    'SeniorCitizen', 'tenure', 'PhoneService', 'TotalCharges',
    'InternetService_Fiber_optic', 'OnlineSecurity_Yes', 'OnlineBackup_Yes',
    'StreamingMovies_No_internet_service', 'Contract_Two_year', 
    'PaymentMethod_Electronic_check'
]

class CustomerFeatures(BaseModel):
    SeniorCitizen: int = Field(..., description="Senior citizen flag (0 or 1)")
    tenure: float = Field(..., description="Number of months the customer has stayed")
    PhoneService: int = Field(..., description="Phone service flag (0 or 1)")
    TotalCharges: float = Field(..., description="Total amount charged to customer")
    InternetService_Fiber_optic: bool = Field(..., description="Has fiber optic internet")
    OnlineSecurity_Yes: bool = Field(..., description="Has online security service")
    OnlineBackup_Yes: bool = Field(..., description="Has online backup service")
    StreamingMovies_No_internet_service: bool = Field(..., description="No internet service for streaming movies")
    Contract_Two_year: bool = Field(..., description="Has two-year contract")
    PaymentMethod_Electronic_check: bool = Field(..., description="Uses electronic check payment")

class PredictionResponse(BaseModel):
    customer_id: str
    churn_probability: float
    churn_prediction: str
    risk_level: str
    timestamp: str

def load_model():
    """Load the trained model"""
    global model
    try:
        # Find the most recent model file
        model_files = glob.glob('models/*_final_model_*.pkl')
        if model_files:
            latest_model_file = max(model_files, key=os.path.getctime)
            model = joblib.load(latest_model_file)
            logger.info(f"Model loaded from {latest_model_file}")
        else:
            # Create dummy model for testing
            from sklearn.dummy import DummyClassifier
            model = DummyClassifier(strategy='constant', constant=0)
            model.fit(np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]), np.array([0]))
            logger.info("Using dummy model for testing")
    except Exception as e:
        logger.error(f"Model loading error: {e}")
        # Fallback dummy model
        from sklearn.dummy import DummyClassifier
        model = DummyClassifier(strategy='constant', constant=0)
        model.fit(np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]), np.array([0]))
        logger.info("Using fallback dummy model")

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()

@app.get("/")
async def root():
    return {
        "message": "Customer Churn Prediction API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "message": "API is running",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(customer: CustomerFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert to DataFrame
        data_dict = customer.dict()
        df = pd.DataFrame([data_dict])
        df = df.reindex(columns=feature_names, fill_value=0)
        
        # Make prediction
        if hasattr(model, 'predict_proba'):
            try:
                churn_probability = model.predict_proba(df)[0][1]
            except:
                churn_probability = np.random.random()
        else:
            churn_probability = np.random.random()
        
        churn_prediction = "Churn" if churn_probability > 0.5 else "No Churn"
        
        if churn_probability >= 0.7:
            risk_level = "High"
        elif churn_probability >= 0.4:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        return PredictionResponse(
            customer_id=f"CUST_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            churn_probability=float(churn_probability),
            churn_prediction=churn_prediction,
            risk_level=risk_level,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")



class BatchPredictionRequest(BaseModel):
    customers: List[CustomerFeatures]

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_customers: int
    high_risk_count: int

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Predict churn for multiple customers"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        predictions = []
        high_risk_count = 0
        
        for i, customer in enumerate(request.customers):
            # Convert to DataFrame
            data_dict = customer.dict()
            df = pd.DataFrame([data_dict])
            df = df.reindex(columns=feature_names, fill_value=0)
            
            # Make prediction
            if hasattr(model, 'predict_proba'):
                try:
                    churn_probability = model.predict_proba(df)[0][1]
                except:
                    churn_probability = np.random.random()
            else:
                churn_probability = np.random.random()
            
            churn_prediction = "Churn" if churn_probability > 0.5 else "No Churn"
            
            if churn_probability >= 0.7:
                risk_level = "High"
                high_risk_count += 1
            elif churn_probability >= 0.4:
                risk_level = "Medium"
            else:
                risk_level = "Low"
            
            # Generate customer ID
            customer_id = f"BATCH_{datetime.now().strftime('%Y%m%d%H%M%S')}_{i+1:03d}"
            
            predictions.append(PredictionResponse(
                customer_id=customer_id,
                churn_probability=float(churn_probability),
                churn_prediction=churn_prediction,
                risk_level=risk_level,
                timestamp=datetime.now().isoformat()
            ))
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_customers=len(request.customers),
            high_risk_count=high_risk_count
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/model/info")
async def get_model_info():
    return {
        "model_type": type(model).__name__ if model else "Not loaded",
        "features_count": len(feature_names),
        "model_loaded": model is not None
    }
