# Campaign Response Prediction Project

This project develops and deploys a machine learning solution to predict whether a customer is likely to respond to a marketing campaign.

## Project Overview

The project was completed in several phases covering data preparation, exploratory analysis, statistical modelling, machine learning, sentiment analysis, and deployment.

### Phase 1 – Data Preparation and Feature Engineering
- Loaded Online Retail Sales and Campaign Response datasets
- Checked missing values, duplicates, and unexpected values
- Created customer-level features including:
  - Total Sales
  - Unique Products Purchased
  - Number of Invoices
  - Number of Purchase Days
- Merged transaction features with campaign response data
- Created categorical versions of selected numerical variables
- Calculated the overall campaign response rate

### Phase 2 – Exploratory Data Analysis
- Compared NPS, competitor count, and communication frequency by campaign response
- Created response-rate tables for loyalty and NPS categories
- Compared transaction features between responders and non-responders
- Analysed monthly sales trends
- Identified top-selling products

### Phase 3 – Logistic Regression
- Split the data into 80% training and 20% testing datasets
- Developed a Binary Logistic Regression model
- Checked statistical significance of variables
- Checked multicollinearity using VIF
- Evaluated performance using ROC curves, AUC, confusion matrix, sensitivity, specificity, and balanced accuracy
- Tested a second model using recoded categorical variables

### Phase 4 – Machine Learning Models
The following models were developed and compared:

- Naive Bayes
- Random Forest
- Support Vector Machine (SVM)
- XGBoost

The models were compared using Test AUC.

Random Forest achieved the highest Test AUC and was selected as the final machine learning model.

### Phase 5 – Sentiment Analysis
Customer feedback was analysed using NLP-based sentiment analysis.

Reviews were classified as:

- Positive
- Neutral
- Negative

Common positive and negative themes were also identified to understand key customer satisfaction drivers and pain points.

### Phase 6 – Model Deployment
The final Random Forest model was saved using Joblib and deployed as a REST API using FastAPI.

The API provides real-time campaign response predictions and returns:

- Predicted probability
- Predicted response class

An Excel front-end was created to allow users to enter customer information and submit prediction requests to the deployed API.

## Model Input Variables

The deployed model uses the following variables:

- Total Sales
- Unique Products
- Number of Invoices
- NPS
- Number of Competitors
- Number of Communications
- Loyalty

## API Endpoint

### Prediction

`POST /predict`

Example request:

```json
{
  "Total_Sales": 100,
  "Unique_Products": 7,
  "Number_of_Invoices": 4,
  "nps": 8,
  "n_comp": 6,
  "n_communications": 8,
  "loyalty": 1
}
```

Example response:

```json
{
  "predicted_probability": 0.84,
  "predicted_class": 1
}
```

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Statsmodels
- XGBoost
- NLTK / VADER
- FastAPI
- Joblib
- Render
- GitHub
- Microsoft Excel
- VBA

## Repository Files

- `main.py` – FastAPI application
- `campaign_response_rf_model_compressed.pkl` – trained Random Forest model
- `campaign_features.pkl` – model feature list
- `requirements.txt` – Python dependencies
- Excel interface – used to submit prediction requests and display results

## Deployment

The FastAPI application is deployed on Render and can be tested using the automatically generated Swagger documentation at:

`/docs`

## Final Model

Random Forest was selected as the final machine learning method because it produced the highest AUC on the test dataset compared with Naive Bayes, SVM, and XGBoost.
