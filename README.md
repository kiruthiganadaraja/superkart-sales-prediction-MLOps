# superkart-sales-prediction-MLOps
End-to-end ML system that forecasts retail sales revenue for SuperKart using XGBoost — with a Flask REST API backend and Streamlit UI, fully containerized with Docker and deployed on Hugging Face Spaces.
# SuperKart Sales Prediction — End-to-End ML Deployment

> Forecasting quarterly retail sales revenue using ensemble ML models, deployed as a Flask REST API (Docker) and Streamlit UI on Hugging Face Spaces.

---

## Project Overview

SuperKart is a retail chain operating supermarkets and food marts across Tier 1, 2, and 3 cities. To optimize inventory management and inform regional sales strategy, this project builds and deploys a production-grade machine learning system that predicts per-product, per-store sales revenue for the upcoming quarter.

The solution covers the full ML lifecycle — from data validation and exploratory analysis through model training, hyperparameter tuning, serialization, and cloud deployment via Hugging Face Spaces.

---

## Folder Structure

```
superkart-sales-prediction/
│
├── notebook/
│   └── SuperKart_Model_Deployment.ipynb       # Full Colab notebook
│
├── backend/
│   ├── app.py                                 # Flask REST API
│   ├── best_xgb_model.pkl                     # Serialized XGBoost model
│   ├── requirements.txt                       # Backend dependencies
│   └── Dockerfile                             # Docker config for backend
│
├── frontend/
│   ├── app.py                                 # Streamlit UI
│   ├── requirements.txt                       # Frontend dependencies
│   └── Dockerfile                             # Docker config for frontend
│
└── README.md
```

---

## Problem Statement

Sales forecasting is critical for retail operations — enabling smarter procurement, territory planning, and regional inventory decisions. SuperKart partnered with a data science team to build a scalable forecasting solution that predicts `Product_Store_Sales_Total` (quarterly revenue per product per store) and integrates directly into operational decision-making systems.

---

## Dataset

The dataset contains **8,763 records** across **12 features** covering product attributes and store characteristics.

| Feature | Description |
|---|---|
| `Product_Id` | Unique product identifier (dropped — high cardinality) |
| `Product_Weight` | Product weight in kg |
| `Product_Sugar_Content` | Sugar level: Low Sugar / Regular / No Sugar |
| `Product_Allocated_Area` | Ratio of shelf space allocated to the product |
| `Product_Type` | Category (e.g., Dairy, Snack Foods, Meat, Seafood — 16 types) |
| `Product_MRP` | Maximum Retail Price |
| `Store_Id` | Unique store identifier (dropped — no predictive value) |
| `Store_Establishment_Year` | Year the store opened |
| `Store_Size` | Store size: Small / Medium / High |
| `Store_Location_City_Type` | City tier: Tier 1 / Tier 2 / Tier 3 |
| `Store_Type` | Store format: Supermarket Type 1/2, Departmental Store, Food Mart |
| `Product_Store_Sales_Total` | **Target variable** — total quarterly sales revenue |

---

## Data Quality & Preprocessing

### Validation Findings
- All **numeric columns** passed validation — no negative, zero, or out-of-range values detected across Product_Weight, Product_MRP, Product_Allocated_Area, Store_Establishment_Year, and Product_Store_Sales_Total.
- One **categorical data quality issue** identified: `Product_Sugar_Content` contained a junk value `reg` (1.23% of records) — standardized to `Regular`.

### Preprocessing Pipeline
1. Standardize `reg` → `Regular` in `Product_Sugar_Content`
2. Drop `Product_Id` (unique identifier, zero predictive signal)
3. Drop `Store_Id` (identifier — no predictive value)
4. One-Hot Encode categorical features: `Product_Sugar_Content`, `Product_Type`, `Store_Size`, `Store_Location_City_Type`, `Store_Type`
5. Pass numeric features through unchanged (tree-based models do not require scaling)
6. 80/20 train-test split (random_state=42)

---

## Exploratory Data Analysis — Key Insights

### Numerical Features
- **Product_MRP** (r ≈ 0.79) and **Product_Weight** (r ≈ 0.74) are the strongest predictors of sales — higher-priced and heavier products consistently generate higher revenue.
- **Product_Allocated_Area** shows negligible correlation — shelf space ratio has minimal direct impact on sales.
- **Store_Establishment_Year** has a weak negative correlation (r ≈ −0.19) — older stores tend to perform differently but the relationship is not strong.

### Store-Level Insights
- **OUT003** (Departmental Store, Tier 1) achieves the highest average sales (~5,000), while **OUT002** (Food Mart, Tier 3) records the lowest (~1,800).
- **Store Size**: Larger stores significantly outperform smaller ones — High > Medium > Small.
- **City Tier**: Tier 1 cities generate the highest mean sales (~5,000), followed by Tier 2 (~3,500) and Tier 3 (~1,800).
- **Store Type**: Departmental Stores lead in average sales, followed by Supermarket Type 1, Supermarket Type 2, and Food Mart.

### Product-Level Insights
- `Fruits and Vegetables` and `Snack Foods` dominate the dataset by volume.
- `Breakfast`, `Starchy Foods`, and `Seafood` show marginally higher mean sales despite lower volume.
- `Product_Id` — every product is unique (each appears exactly once), confirming it carries no predictive value.

---

## Model Building

### Models Evaluated
Two ensemble tree-based regressors were selected:

| Model | Rationale |
|---|---|
| Random Forest Regressor | Robust baseline for tabular data; handles non-linearity and outliers; resistant to overfitting via ensemble averaging |
| XGBoost Regressor | Efficient gradient boosting with built-in L1/L2 regularization; excels at learning complex feature interactions |

### Evaluation Metric
**RMSE (Root Mean Squared Error)** — primary metric, as it penalizes large prediction errors more heavily, which matters in retail sales forecasting where large deviations are costly.

MAE and R² are also reported for completeness.

---

## Model Performance

### Baseline Results

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Random Forest | 282.86 | 108.01 | 0.9299 |
| XGBoost | 290.39 | 122.56 | 0.9261 |

### After Hyperparameter Tuning (RandomizedSearchCV, 5-fold CV)

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Random Forest (Tuned) | 318.22 | 179.92 | 0.9113 |
| **XGBoost (Tuned)** | **287.53** | **120.01** | **0.9275** |

### Final Model: Tuned XGBoost Regressor

XGBoost tuning meaningfully improved performance — achieving the lowest RMSE, lowest MAE, and highest R² across all evaluated models. Random Forest tuning did not improve over its baseline, indicating the default configuration was already well-suited to this dataset's size and feature space.

**The tuned XGBoost pipeline is selected as the final production model.**

---

## Model Serialization

The final model pipeline is serialized using `joblib` for production deployment:

```python
import joblib

# Save
joblib.dump(xgb_best, "best_xgb_model.pkl")

# Load and verify
loaded_model = joblib.load("best_xgb_model.pkl")
# RMSE: 287.53 | MAE: 120.01 | R²: 0.9275 — confirmed consistent
```

---

## Deployment Architecture

The solution is deployed as a two-service architecture on Hugging Face Spaces:

```
[User] → [Streamlit UI (Docker Space)] → [Flask REST API (Docker Space)] → [XGBoost Model]
```

### Backend — Flask REST API

Hosted at: `https://Kiruthigan17-SalesPredictionBackend.hf.space`

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Health check — confirms API is running |
| `/predict` | POST | Single prediction from JSON payload |
| `/predict_batch` | POST | Batch prediction from uploaded CSV |

**Example Request:**
```json
POST /predict
{
  "Product_Weight": 12.5,
  "Product_Sugar_Content": "Low Sugar",
  "Product_Allocated_Area": 0.05,
  "Product_Type": "Snack Foods",
  "Product_MRP": 150.0,
  "Store_Establishment_Year": 2009,
  "Store_Size": "Medium",
  "Store_Location_City_Type": "Tier 2",
  "Store_Type": "Supermarket Type2"
}
```

**Example Response:**
```json
{ "predicted_sales": 3241.87 }
```

### Frontend — Streamlit UI

Hosted at: `https://kiruthigan17-salespredictionui.hf.space`

Interactive web application allowing users to input product and store details via dropdowns and number inputs, then receive real-time sales predictions from the Flask backend.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data Processing | Python, Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| ML Models | Scikit-learn (Random Forest), XGBoost |
| Pipeline & Encoding | Scikit-learn Pipeline, ColumnTransformer, OneHotEncoder |
| Hyperparameter Tuning | RandomizedSearchCV (5-fold CV) |
| Model Serialization | Joblib |
| Backend API | Flask, Gunicorn |
| Frontend UI | Streamlit |
| Containerization | Docker |
| Cloud Deployment | Hugging Face Spaces |
| Development Environment | Google Colab, Google Drive |

---

## Business Recommendations

1. **Pricing is the #1 lever** — Product_MRP drives the strongest correlation with sales. Pricing strategy should be reviewed store-by-store across city tiers before inventory planning.
2. **Prioritize Tier 1 / Departmental Store expansion** — these store types consistently generate 2–3x the revenue of Tier 3 Food Mart locations.
3. **Review low-performing stores** — OUT002 underperforms significantly compared to OUT003. Regional pricing or assortment strategy differences may explain the gap and should be investigated.
4. **Shelf space allocation needs rethinking** — Product_Allocated_Area shows near-zero correlation with sales, suggesting current shelf allocation practices are not driving incremental revenue.
5. **Standardize data entry** — the `reg` / `Regular` inconsistency in Product_Sugar_Content is a data governance gap. Implementing input validation at the point of data entry will prevent quality issues at scale.

---

## How to Run Locally

### Backend
```bash
cd backend
pip install -r requirements.txt
python app.py
# API runs at http://localhost:5000
```

### Frontend
```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
# UI runs at http://localhost:8501
```

### Run with Docker
```bash
# Backend
docker build -t superkart-backend ./backend
docker run -p 7860:7860 superkart-backend

# Frontend
docker build -t superkart-frontend ./frontend
docker run -p 7860:7860 superkart-frontend
```

---

## Live Demo

- **Streamlit UI:** https://kiruthigan17-salespredictionui.hf.space
- **Flask API:** https://Kiruthigan17-SalesPredictionBackend.hf.space

---
