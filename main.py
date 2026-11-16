from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib


app = FastAPI()


# Load model and feature list
model = joblib.load(
     "campaign_response_rf_model_compressed.pkl"
)

features = joblib.load(
    "campaign_features.pkl"
)


class CustomerData(BaseModel):
    Total_Sales: float
    Unique_Products: int
    Number_of_Invoices: int
    nps: float
    n_comp: int
    n_communications: int
    loyalty: int


@app.get("/")
def home():
    return {
        "message": "Campaign Response Prediction API",
        "endpoint": "/predict"
    }


@app.post("/predict")
def predict(data: CustomerData):

    input_data = pd.DataFrame([{
        "Total_Sales": data.Total_Sales,
        "Unique_Products": data.Unique_Products,
        "Number_of_Invoices": data.Number_of_Invoices,
        "nps": data.nps,
        "n_comp": data.n_comp,
        "n_communications": data.n_communications,
        "loyalty": data.loyalty
    }])

    # Keep the exact feature order used for training
    input_data = input_data[features]

    probability = model.predict_proba(
        input_data
    )[0][1]

    prediction = model.predict(
        input_data
    )[0]

    return {
        "predicted_probability":
        round(float(probability), 4),

        "predicted_class":
        int(prediction)
    }
