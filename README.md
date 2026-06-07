# Loan Approval Risk Scoring API

> Predicts loan approval probability based on applicant's financial profile,
> helping lenders make faster and more consistent credit decisions.

[![Python](https://img.shields.io/badge/Python-3.11-blue)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688)]()
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green)]()
[![Accuracy](https://img.shields.io/badge/Accuracy-92%25-brightgreen)]()

---

## Business Problem

Manual loan review is slow, inconsistent, and costly — a single analyst
processes tens of applications per day while approval bias remains a
regulatory risk. This model automates the initial screening of applicants
using objective financial indicators, reducing review time and standardizing
decisions across portfolios.

---

## Demo

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "no_of_dependents": 2,
       "education": "Graduate",
       "self_employed": "No",
       "income_annum": 900000,
       "loan_amount": 5000000,
       "loan_term": 12,
       "cibil_score": 750,
       "residential_assets_value": 3000000,
       "commercial_assets_value": 1000000,
       "luxury_assets_value": 500000,
       "bank_asset_value": 800000
     }'
```

Response:
```json
{"Answer:": "Approved"}
```

---

## Results

| Metric    | Score        |
|-----------|--------------|
| Accuracy  | ~92%         |
| F1-score  | —            |
| ROC-AUC   | —            |

Best model: Logistic Regression  
Baseline (majority class): Accuracy ≈ 67%  
↑ +25% improvement vs baseline

> F1 and ROC-AUC are planned for the next iteration.

---

## Dataset

- Source: Loan approval dataset (Kaggle)
- Size: 4 269 records
- Features: 11 (numeric + 2 categorical, label-encoded)
- Class balance: moderately imbalanced — not explicitly handled (planned: SMOTE)

---

## Approach

1. **Data loading & inspection** — shape, dtypes, column names audit
2. **Cleaning** — strip whitespace from column names and string values, drop `loan_id`
3. **EDA** — pairplot, correlation heatmap, unique value inspection
4. **Encoding** — label encoding for `education`, `self_employed`, `loan_status`
5. **Splitting** — 80/20 train/test, `random_state=42`
6. **Scaling** — `StandardScaler` fit on train, applied to test
7. **Training** — `LogisticRegression` (default params)
8. **Evaluation** — accuracy score on test set
9. **Serialization** — model and scaler saved via `joblib`
10. **Deployment** — FastAPI REST API with Pydantic schema validation

---

## Key Challenges & Solutions

**Dirty input data**  
Column names and string values contained leading/trailing whitespace →
applied `.str.lstrip()` / `.str.strip()` globally →
eliminated silent label encoding failures.

**Categorical features in a numeric pipeline**  
`education` and `self_employed` are strings; StandardScaler requires floats →
manual binary encoding before scaling →
clean pipeline with no dependency on `pd.get_dummies` or ColumnTransformer.

**Class imbalance (planned fix)**  
Rejected applications outnumber approved ones in many real portfolios →
current model trained without resampling →
next step: SMOTE or `class_weight='balanced'` to improve minority-class recall.

---

## Tech Stack

| Category   | Tools                              |
|------------|------------------------------------|
| Language   | Python 3.11                        |
| ML         | scikit-learn (LogisticRegression)  |
| API        | FastAPI, Uvicorn, Pydantic         |
| Data       | pandas, NumPy                      |
| Viz        | seaborn, matplotlib                |
| Deployment | joblib (model serialization)       |

---

## How to Run

```bash
# 1. Clone & install
git clone https://github.com/your-username/loan-risk-api.git
cd loan-risk-api
pip install -r requirements.txt
```

```bash
# 2. Train the model
jupyter notebook loan_model.ipynb
# or: python train.py
```

```bash
# 3. Start the API
python main.py
# → http://127.0.0.1:8000/docs
```

---

## Business Impact

- ↓ ~70% review time per application vs manual screening (estimated)
- ↑ decision consistency — removes analyst-to-analyst variability
- ↓ regulatory risk from subjective approval criteria
- ↑ throughput — API handles hundreds of requests/min vs ~20 manual/day
- ↑ auditability — every prediction logged with input features (planned)

---
