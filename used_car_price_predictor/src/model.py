"""Model training and evaluation. Target is log-transformed to handle skewed prices."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUM = ["vehicle_age", "km_driven", "mileage", "engine", "max_power", "seats"]
CAT = ["brand", "model", "seller_type", "fuel_type", "transmission"]
FEATURES = NUM + CAT


def build_pipeline(estimator) -> Pipeline:
    prep = ColumnTransformer([
        ("num", StandardScaler(), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
    ])
    return Pipeline([("prep", prep), ("model", estimator)])


def predict_price(pipe: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Predicted price in INR (model works on log1p scale)."""
    return np.expm1(pipe.predict(X[FEATURES]))


def price_range(pipe: Pipeline, X: pd.DataFrame, low: int = 10, high: int = 90) -> tuple[float, float] | None:
    """Spread of individual trees (RandomForest only); an indicative range, not a guarantee."""
    rf = pipe.named_steps["model"]
    if not hasattr(rf, "estimators_"):
        return None
    Z = pipe.named_steps["prep"].transform(X[FEATURES])
    preds = np.expm1(np.array([t.predict(Z) for t in rf.estimators_]))[:, 0]
    return float(np.percentile(preds, low)), float(np.percentile(preds, high))


def _metrics(y, p) -> dict:
    return {"R²": r2_score(y, p), "MAE (₹)": mean_absolute_error(y, p),
            "RMSE (₹)": float(np.sqrt(mean_squared_error(y, p)))}


def feature_importance(pipe: Pipeline) -> pd.DataFrame:
    rf = pipe.named_steps["model"]
    if not hasattr(rf, "feature_importances_"):
        return pd.DataFrame(columns=["feature", "importance"])
    names = [n.split("__", 1)[1] for n in pipe.named_steps["prep"].get_feature_names_out()]
    imp = dict.fromkeys(FEATURES, 0.0)
    for name, v in zip(names, rf.feature_importances_):
        owner = name if name in imp else next(f for f in CAT if name.startswith(f + "_"))
        imp[owner] += float(v)
    return pd.DataFrame({"feature": list(imp), "importance": list(imp.values())}).sort_values("importance")


def train_and_evaluate(df: pd.DataFrame, seed: int = 42, n_trees: int = 120) -> dict:
    """Train Ridge (baseline) and RandomForest; keep the best by test R²."""
    X, y = df[FEATURES], df["selling_price"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
    models = {
        "Ridge (baseline)": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=n_trees, min_samples_leaf=2, n_jobs=-1, random_state=seed),
    }
    rows, fitted = [], {}
    for name, est in models.items():
        pipe = build_pipeline(est).fit(Xtr, np.log1p(ytr))
        pred = predict_price(pipe, Xte)
        rows.append({"Model": name, **_metrics(yte, pred)})
        fitted[name] = (pipe, pred)
    results = pd.DataFrame(rows).set_index("Model")
    best = results["R²"].idxmax()
    pipe, pred = fitted[best]
    test = Xte.assign(actual=yte.values, predicted=pred)
    return {"best": best, "pipeline": pipe, "results": results, "test": test,
            "importance": feature_importance(pipe), "n_train": len(Xtr), "n_test": len(Xte)}
