from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import joblib


model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

app = FastAPI()

class DataSchema(BaseModel):
    no_of_dependents: int
    education: str
    self_employed: str
    income_annum: int
    loan_amount: int
    loan_term: int
    cibil_score: int
    residential_assets_value: int
    commercial_assets_value: int
    luxury_assets_value: int
    bank_asset_value: int


@app.post('/predict')
async def predict(data: DataSchema):
    data_dict = dict(data)

    education = data_dict.pop("education")
    education_binary = [1 if education == "Not Graduate" else 0]

    employed = data_dict.pop("self_employed")
    employed_binary = [1 if employed == "Yes" else 0]

    features = list(data_dict.values()) + education_binary + employed_binary
    scaled_data = scaler.transform([features])
    prediction = model.predict(scaled_data)[0]
    return {"class": str(prediction)}



if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
