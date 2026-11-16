from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import os


app = FastAPI()


# Load model and feature list
model = joblib.load(
    "campaign_response_rf_model_compressed.pkl"
)

features = joblib.load(
    "campaign_features.pkl"
)


# Create folder for uploaded input files
INPUT_FOLDER = "input_data"

os.makedirs(
    INPUT_FOLDER,
    exist_ok=True
)


class CustomerData(BaseModel):

    Total_Sales: float = Field(ge=0)

    Unique_Products: int = Field(ge=0)

    Number_of_Invoices: int = Field(ge=0)

    nps: float = Field(
        ge=0,
        le=10
    )

    n_comp: int = Field(ge=0)

    n_communications: int = Field(ge=0)

    loyalty: int = Field(
        ge=0,
        le=1
    )


@app.get("/")
def home():

    return {
        "message":
        "Campaign Response Prediction API",

        "individual_prediction":
        "/predict",

        "batch_prediction":
        "/predict-file"
    }


@app.post("/predict")
def predict(data: CustomerData):

    input_data = pd.DataFrame([{
        "Total_Sales":
        data.Total_Sales,

        "Unique_Products":
        data.Unique_Products,

        "Number_of_Invoices":
        data.Number_of_Invoices,

        "nps":
        data.nps,

        "n_comp":
        data.n_comp,

        "n_communications":
        data.n_communications,

        "loyalty":
        data.loyalty
    }])

    input_data = input_data[
        features
    ]

    probability = model.predict_proba(
        input_data
    )[0][1]

    prediction = model.predict(
        input_data
    )[0]

    return {
        "predicted_probability":
        round(
            float(probability),
            4
        ),

        "predicted_class":
        int(prediction)
    }


@app.post("/predict-file")
async def predict_file(
    file: UploadFile = File(...)
):

    # Only allow CSV files
    if not file.filename.endswith(".csv"):

        raise HTTPException(
            status_code=400,
            detail="Only CSV files are accepted."
        )

    file_path = os.path.join(
        INPUT_FOLDER,
        file.filename
    )

    # Save uploaded file
    contents = await file.read()

    with open(
        file_path,
        "wb"
    ) as f:

        f.write(contents)

    # Load CSV
    try:

        data = pd.read_csv(
            file_path
        )

    except Exception:

        raise HTTPException(
            status_code=400,
            detail="Unable to read the CSV file."
        )


    # Check required columns
    missing_columns = [
        column
        for column in features
        if column not in data.columns
    ]

    if missing_columns:

        raise HTTPException(
            status_code=400,
            detail=
            f"Missing columns: {missing_columns}"
        )


    # Keep required features
    input_data = data[
        features
    ]


    # Check invalid values
    if (
        (input_data["nps"] < 0).any()
        or
        (input_data["nps"] > 10).any()
    ):

        raise HTTPException(
            status_code=400,
            detail=
            "NPS values must be between 0 and 10."
        )


    if (
        ~input_data["loyalty"]
        .isin([0, 1])
    ).any():

        raise HTTPException(
            status_code=400,
            detail=
            "Loyalty values must be 0 or 1."
        )


    # Generate predictions
    probabilities = (
        model.predict_proba(
            input_data
        )[:, 1]
    )

    predictions = model.predict(
        input_data
    )


    # Add results
    data[
        "Predicted_Probability"
    ] = probabilities.round(4)

    data[
        "Predicted_Class"
    ] = predictions


    return {
        "records_processed":
        len(data),

        "predictions":
        data.to_dict(
            orient="records"
        )
    }
