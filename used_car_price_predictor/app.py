"""Used Car Price Predictor (Streamlit). Run: python -m streamlit run app.py"""
from __future__ import annotations

import io

import pandas as pd
import plotly.express as px
import streamlit as st

from src.data import DataError, clean, find_dataset, generate_synthetic, load_csv
from src.model import predict_price, price_range, train_and_evaluate

st.set_page_config(page_title="Used Car Price Predictor", page_icon="🚗", layout="wide")
COLORS = ["#264653", "#2a9d8f", "#e9c46a", "#e76f51", "#8d99ae"]
inr = lambda v: f"₹{v:,.0f}"
lakh = lambda v: f"₹{v / 1e5:,.2f} lakh"


@st.cache_data(show_spinner="Loading and cleaning data…")
def load(kind: str, payload) -> tuple[pd.DataFrame, dict]:
    if kind == "bytes":
        return clean(load_csv(io.BytesIO(payload)))
    if kind == "path":
        return clean(load_csv(payload))
    return clean(generate_synthetic())


@st.cache_resource(show_spinner="Training models (first run only)…")
def train(df: pd.DataFrame) -> dict:
    return train_and_evaluate(df)


def get_data() -> tuple[pd.DataFrame, dict, str]:
    up = st.sidebar.file_uploader("Upload car CSV (optional)", type="csv")
    local = find_dataset()
    tries = []
    if up is not None:
        tries.append((f"uploaded file ({up.name})", ("bytes", up.getvalue())))
    if local is not None:
        tries.append((f"local file ({local.name})", ("path", str(local))))
    for label, (kind, payload) in tries:
        try:
            df, rep = load(kind, payload)
            if len(df) < 200:
                raise DataError(f"only {len(df)} valid rows after cleaning (need at least 200)")
            return df, rep, label
        except DataError as exc:
            st.error(f"Could not use the {label}: {exc}")
    st.warning("Using **synthetic sample data** (fictional). Put the CarDekho CSV in `data/raw/` or upload it.")
    df, rep = load("synthetic", None)
    return df, rep, "synthetic sample data"


st.title("🚗 Used Car Price Predictor")
st.caption("Explore what drives used-car prices in India and estimate the price of a car with a machine-learning model.")
df, report, source = get_data()
st.sidebar.caption(f"Data source: {source}\n\n{len(df):,} cars after cleaning")
art = train(df)
pipe = art["pipeline"]

tab_pred, tab_exp, tab_model, tab_data = st.tabs(["Predict price", "Explore data", "Model performance", "Data & limitations"])

# ------------------------------------------------------------------ predict
with tab_pred:
    c1, c2 = st.columns(2)
    brand = c1.selectbox("Brand", sorted(df["brand"].unique()))
    sub = df[df["brand"] == brand]
    model = c2.selectbox("Model", sorted(sub["model"].unique()))
    ref = df[(df["brand"] == brand) & (df["model"] == model)]
    k = f"_{brand}_{model}"

    def pick(label: str, col: str, container) -> str:
        opts = sorted(df[col].unique())
        return container.selectbox(label, opts, index=opts.index(ref[col].mode()[0]), key=col + k)

    a, b, c = st.columns(3)
    age = a.number_input("Vehicle age (years)", 0, 30, int(min(max(ref.vehicle_age.median(), 0), 30)), key="age" + k)
    km = b.number_input("Kilometres driven", 0, 1_000_000, int(min(ref.km_driven.median(), 1_000_000)), step=1000, key="km" + k)
    seats = c.number_input("Seats", 2, 10, int(min(max(ref.seats.mode()[0], 2), 10)), key="seats" + k)
    d, e, f = st.columns(3)
    fuel, trans, seller = pick("Fuel type", "fuel_type", d), pick("Transmission", "transmission", e), pick("Seller type", "seller_type", f)
    g, h, i = st.columns(3)
    mileage = g.number_input("Mileage (km/l)", 0.0, 60.0, float(round(ref.mileage.median(), 1)), key="mil" + k)
    engine = h.number_input("Engine (cc)", 200, 8000, int(min(max(ref.engine.median(), 200), 8000)), key="eng" + k)
    power = i.number_input("Max power (bhp)", 20.0, 800.0, float(min(max(round(ref.max_power.median(), 1), 20), 800)), key="pw" + k)

    row = pd.DataFrame([dict(brand=brand, model=model, vehicle_age=age, km_driven=km, seller_type=seller,
                             fuel_type=fuel, transmission=trans, mileage=mileage, engine=engine, max_power=power, seats=seats)])
    price = float(predict_price(pipe, row)[0])
    st.divider()
    m1, m2 = st.columns(2)
    m1.metric("Estimated price", inr(price), help=f"Predicted by: {art['best']}")
    m2.metric("In lakh", lakh(price))
    rng = price_range(pipe, row)
    if rng:
        st.caption(f"Indicative range (10th–90th percentile of the model's trees): {inr(rng[0])} – {inr(rng[1])}")
    st.caption(f"Typical listed price for {brand} {model} in the data: {lakh(ref.selling_price.median())} "
               f"(median of {len(ref):,} listings). This is an estimate, not a valuation.")

# ------------------------------------------------------------------ explore
with tab_exp:
    sample = df.sample(min(len(df), 4000), random_state=1)
    c1, c2 = st.columns(2)
    fig = px.scatter(sample, x="vehicle_age", y="price_lakh", color="fuel_type", opacity=0.5, color_discrete_sequence=COLORS,
                     hover_data=["brand", "model"], labels={"vehicle_age": "Vehicle age (years)", "price_lakh": "Price (₹ lakh)", "fuel_type": "Fuel"},
                     title="Price falls as cars age")
    c1.plotly_chart(fig)
    fig = px.scatter(sample, x="km_driven", y="price_lakh", color="transmission", opacity=0.5, color_discrete_sequence=COLORS,
                     labels={"km_driven": "Kilometres driven", "price_lakh": "Price (₹ lakh)", "transmission": "Transmission"},
                     title="Price vs kilometres driven")
    c2.plotly_chart(fig)
    top = df.groupby("brand")["price_lakh"].agg(["median", "count"]).query("count >= 20").nlargest(15, "median").reset_index()
    c3, c4 = st.columns(2)
    c3.plotly_chart(px.bar(top.sort_values("median"), x="median", y="brand", orientation="h", color_discrete_sequence=[COLORS[0]],
                           hover_data=["count"], labels={"median": "Median price (₹ lakh)", "brand": ""},
                           title="Median price by brand (brands with 20+ cars)"))
    by = df.groupby(["fuel_type", "transmission"])["price_lakh"].median().reset_index()
    c4.plotly_chart(px.bar(by, x="fuel_type", y="price_lakh", color="transmission", barmode="group", color_discrete_sequence=COLORS,
                           labels={"price_lakh": "Median price (₹ lakh)", "fuel_type": "Fuel", "transmission": "Transmission"},
                           title="Median price by fuel and transmission"))
    corr = df[["selling_price", "vehicle_age", "km_driven", "mileage", "engine", "max_power", "seats"]].corr()
    st.plotly_chart(px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu", zmin=-1, zmax=1, title="Correlation between numeric columns"))
    st.caption("Correlation shows association, not cause: e.g. power and price rise together partly because bigger, newer car classes have both.")

# ------------------------------------------------------------------ model
with tab_model:
    st.markdown(f"Trained on **{art['n_train']:,}** cars and tested on **{art['n_test']:,}** unseen cars. "
                f"Best model: **{art['best']}**.")
    st.dataframe(art["results"].style.format({"R²": "{:.3f}", "MAE (₹)": "₹{:,.0f}", "RMSE (₹)": "₹{:,.0f}"}))
    st.caption("R² = share of price variation explained. MAE = average error in rupees. RMSE penalises large errors more.")
    t = art["test"].sample(min(len(art["test"]), 3000), random_state=1)
    fig = px.scatter(t, x="actual", y="predicted", opacity=0.4, color_discrete_sequence=[COLORS[1]],
                     labels={"actual": "Actual price (₹)", "predicted": "Predicted price (₹)"}, title="Predicted vs actual (test set)")
    mx = float(max(t.actual.max(), t.predicted.max()))
    fig.add_shape(type="line", x0=0, y0=0, x1=mx, y1=mx, line=dict(dash="dot", color="#333"))
    st.plotly_chart(fig)
    if len(art["importance"]):
        st.plotly_chart(px.bar(art["importance"], x="importance", y="feature", orientation="h", color_discrete_sequence=[COLORS[0]],
                               labels={"importance": "Importance", "feature": ""}, title="Which inputs matter most to the model"))

# ------------------------------------------------------------------ data
with tab_data:
    st.markdown(f"**Source:** {source}")
    st.dataframe(df.head(50))
    st.subheader("Cleaning report")
    st.json(report)
    st.markdown("""
**Limitations**
- Predictions come from past listings; real prices also depend on condition, accident history, location, service records and negotiation, none of which are in the data.
- The model only learns brands/models present in the data; unusual cars give less reliable estimates.
- Prices are in INR and reflect the dataset's time period; market changes make older data less accurate.
- Feature importance and correlations describe patterns, not causes.
- Synthetic fallback data is fictional and only for demonstrating the app.
""")
