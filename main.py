from fastapi import FastAPI
import uvicorn
import joblib
from pydantic import BaseModel


model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

exam_app = FastAPI()

class PropertySchema(BaseModel):
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


@exam_app.post('/predict/')
async def predict(data: PropertySchema):
    data_dict = data.dict()

    education_val = data_dict.pop("education")
    self_emp_val = data_dict.pop("self_employed")

    education_ohe = [1 if education_val == "Not Graduate" else 0]
    self_emp_ohe = [1 if self_emp_val == "Yes" else 0]

    features = list(data_dict.values()) + education_ohe + self_emp_ohe

    scaled_data = scaler.transform([features])

    pred = model.predict(scaled_data)[0]

    return {"class": str(pred)}



if __name__ == 'main':
    uvicorn.run(exam_app, host='localhost', port=8000)
