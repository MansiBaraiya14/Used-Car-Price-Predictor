import pandas as pd
import pytest

from src.data import DataError, clean, generate_synthetic, normalize_columns
from src.model import FEATURES, predict_price, price_range, train_and_evaluate


def test_normalize_drops_index_and_renames_transmission():
    raw = pd.DataFrame(columns=["Unnamed: 0", "Transmission_Type", " Car Name "])
    assert list(normalize_columns(raw).columns) == ["transmission", "car_name"]


def test_clean_synthetic_and_derived_column():
    df, rep = clean(generate_synthetic(500))
    assert rep["rows_out"] == len(df) > 0
    assert (df["price_lakh"] == df["selling_price"] / 1e5).all()


def test_missing_column_raises():
    with pytest.raises(DataError):
        clean(generate_synthetic(50).drop(columns=["engine"]))


def test_duplicates_and_invalid_rows_removed():
    raw = generate_synthetic(100)
    bad = raw.iloc[[0]].copy()
    bad["selling_price"] = -5
    bad["km_driven"] = "abc"
    dirty = pd.concat([raw, raw.iloc[[1]], bad], ignore_index=True)
    df, rep = clean(dirty)
    assert rep["duplicates_removed"] >= 1 and rep["invalid_rows_dropped"] >= 1
    assert (df["selling_price"] > 0).all()


def test_text_values_normalized():
    raw = generate_synthetic(60)
    raw["fuel_type"] = " petrol "
    df, _ = clean(raw)
    assert set(df["fuel_type"]) == {"Petrol"}


@pytest.fixture(scope="module")
def trained():
    df, _ = clean(generate_synthetic(1500))
    return df, train_and_evaluate(df, n_trees=40)


def test_model_quality_and_prediction(trained):
    df, art = trained
    assert art["results"]["R²"].max() > 0.8
    row = df.iloc[[0]]
    pred = predict_price(art["pipeline"], row)[0]
    assert pred > 0


def test_older_cars_predicted_cheaper(trained):
    df, art = trained
    row = df.iloc[[0]].copy()
    young, old = row.assign(vehicle_age=2), row.assign(vehicle_age=14)
    assert predict_price(art["pipeline"], young)[0] > predict_price(art["pipeline"], old)[0]


def test_unseen_model_does_not_crash(trained):
    df, art = trained
    row = df.iloc[[0]].assign(brand="Nonexistent", model="Nothing")
    assert predict_price(art["pipeline"], row)[0] > 0


def test_price_range_ordered(trained):
    df, art = trained
    rng = price_range(art["pipeline"], df.iloc[[0]])
    if art["best"] == "Random Forest":
        assert rng[0] <= rng[1]
    assert set(FEATURES) <= set(df.columns)
