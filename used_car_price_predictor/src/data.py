"""Loading, validating, cleaning the CarDekho used-car dataset (plus a synthetic fallback)."""
from __future__ import annotations

import re
from pathlib import Path
from typing import BinaryIO

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
REQUIRED = ["brand", "model", "vehicle_age", "km_driven", "seller_type", "fuel_type", "transmission",
            "mileage", "engine", "max_power", "seats", "selling_price"]
NUMERIC = ["vehicle_age", "km_driven", "mileage", "engine", "max_power", "seats", "selling_price"]


class DataError(Exception):
    """Raised when the dataset cannot be read or has missing required columns."""


def find_dataset(raw_dir: Path = RAW_DIR) -> Path | None:
    """Return the first CSV in ``raw_dir`` (e.g. cardekho_dataset.csv), or None."""
    files = sorted(raw_dir.glob("*.csv"))
    return files[0] if files else None


def load_csv(source: str | Path | BinaryIO) -> pd.DataFrame:
    for enc in ("utf-8", "latin-1"):
        try:
            if hasattr(source, "seek"):
                source.seek(0)
            df = pd.read_csv(source, encoding=enc)
            if df.empty:
                raise DataError("The CSV file has no rows.")
            return df
        except UnicodeDecodeError:
            continue
        except (pd.errors.ParserError, pd.errors.EmptyDataError, FileNotFoundError) as exc:
            raise DataError(f"Could not read CSV: {exc}") from exc
    raise DataError("Could not decode the CSV file.")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_") for c in out.columns]
    out = out.loc[:, [c for c in out.columns if c and not c.startswith("unnamed")]]  # drop index column
    return out.rename(columns={"transmission_type": "transmission"})


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean the raw frame. Returns (clean_df, report).

    Assumptions: prices are in INR; exact duplicates are removed; rows with missing/invalid
    required values or non-positive price/engine/power/seats are dropped; outliers are kept.
    """
    df = normalize_columns(raw)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise DataError(f"Missing required columns: {', '.join(missing)}")
    report: dict = {"rows_in": len(df)}
    n = len(df)
    df = df.drop_duplicates()
    report["duplicates_removed"] = n - len(df)

    for c in ("brand", "model"):
        df[c] = df[c].astype("string").str.strip()
    for c in ("seller_type", "fuel_type", "transmission"):
        df[c] = df[c].astype("string").str.strip().str.title()
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    n = len(df)
    df = df.dropna(subset=REQUIRED)
    valid = ((df.selling_price > 0) & (df.vehicle_age >= 0) & (df.km_driven >= 0) & (df.engine > 0)
             & (df.max_power > 0) & (df.seats > 0) & (df.mileage >= 0))
    df = df[valid]
    report["invalid_rows_dropped"] = n - len(df)
    df = df.assign(price_lakh=df["selling_price"] / 1e5).reset_index(drop=True)
    report["rows_out"] = len(df)
    return df, report


def generate_synthetic(n: int = 3000, seed: int = 42) -> pd.DataFrame:
    """Fictional CarDekho-like data (original column names) for demoing the app."""
    rng = np.random.default_rng(seed)
    cat = {  # brand: [(model, engine cc, power bhp, new price INR)]
        "Maruti": [("Alto", 796, 46, 350000), ("Swift", 1197, 83, 600000), ("Baleno", 1197, 83, 700000)],
        "Hyundai": [("i20", 1197, 82, 750000), ("Creta", 1497, 113, 1300000), ("Venue", 998, 118, 900000)],
        "Honda": [("City", 1498, 118, 1100000), ("Amaze", 1198, 88, 700000)],
        "Toyota": [("Innova", 2494, 100, 1800000), ("Fortuner", 2755, 175, 3200000)],
        "Mercedes-Benz": [("C-Class", 1950, 194, 5500000)],
    }
    rows = []
    for _ in range(n):
        brand = rng.choice(list(cat))
        mdl, eng, pw, new = cat[brand][rng.integers(len(cat[brand]))]
        age, fuel = int(rng.integers(1, 15)), rng.choice(["Petrol", "Diesel", "Cng"], p=[.6, .35, .05])
        trans, seller = rng.choice(["Manual", "Automatic"], p=[.75, .25]), rng.choice(["Individual", "Dealer"], p=[.6, .4])
        km = int(max(1000, rng.normal(11000 * age, 8000)))
        price = (new * np.exp(-0.11 * age) * (1 - min(km, 300000) / 700000) * (1.12 if fuel == "Diesel" else 1)
                 * (1.15 if trans == "Automatic" else 1) * (1.05 if seller == "Dealer" else 1) * rng.lognormal(0, .08))
        rows.append({"car_name": f"{brand} {mdl}", "brand": brand, "model": mdl, "vehicle_age": age, "km_driven": km,
                     "seller_type": seller, "fuel_type": fuel, "transmission_type": trans,
                     "mileage": round(float(rng.uniform(14, 26)), 2), "engine": eng,
                     "max_power": round(pw * float(rng.uniform(.97, 1.03)), 2),
                     "seats": 7 if mdl in ("Innova", "Fortuner") else 5, "selling_price": int(price)})
    return pd.DataFrame(rows)
