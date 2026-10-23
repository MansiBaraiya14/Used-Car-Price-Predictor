# Used Car Price Predictor

Analyses Indian used-car listings (CarDekho dataset) and predicts a car's selling price with machine learning,
through an interactive Streamlit app.

## Setup
```bash
pip install -r requirements.txt
```
## Dataset
Save the Kaggle CarDekho CSV (e.g. `cardekho_dataset.csv`) in `data/raw/` (any `.csv` there is detected),
or upload it in the app sidebar. Expected columns: `car_name, brand, model, vehicle_age, km_driven,
seller_type, fuel_type, transmission_type, mileage, engine, max_power, seats, selling_price` (plus an unnamed index column, which is dropped).
If no file is found, the app uses a clearly labelled synthetic sample.

## Run and test
```bash
python -m streamlit run app.py
python -m pytest
```
## What it does
- **Predict price:** pick brand/model, adjust details, get an estimate (₹ and lakh) with an indicative range.
- **Explore data:** price vs age and km, brand/fuel/transmission comparisons, correlations.
- **Model performance:** Ridge baseline vs Random Forest, scored on unseen data with R², MAE, RMSE; predicted-vs-actual; feature importance.
- **Data & limitations:** data preview, cleaning report, caveats.

## Method
Cleaning (duplicates, invalid values) → one-hot encoding of categories and scaling of numbers → models trained on
`log(price)` (prices are skewed) → 80/20 train/test split → best model chosen by test R².

## Structure
`app.py` UI · `src/data.py` loading/cleaning/synthetic data · `src/model.py` training and evaluation · `tests/`

## Limitations
Condition, accident history, city and negotiation are not in the data; unseen models give weaker estimates;
prices reflect the dataset's period; correlations and feature importance are not causal.
