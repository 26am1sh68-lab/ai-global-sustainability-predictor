import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from pathlib import Path

st.set_page_config(page_title="Global Sustainability Predictor", page_icon="🌍", layout="wide")

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "sample_indicators.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

data = load_data()
METRICS = {
    "Energy consumption": ("energy_kwh_per_capita", "kWh per capita"),
    "Water consumption": ("water_m3_per_capita", "m³ per capita"),
    "Carbon emissions": ("carbon_tonnes_per_capita", "tonnes CO₂ per capita"),
}

st.title("🌍 AI-Based Global Resource & Sustainability Predictor")
st.caption("Explore resource trends, forecast future indicators, and review an illustrative sustainability index.")

st.warning(
    "DEMO DATA: This starter project uses generated sample data for demonstration, "
    "not verified country statistics. Replace it with World Bank / UN / IEA data before "
    "making real-world claims."
)

with st.sidebar:
    st.header("Prediction settings")
    country = st.selectbox("Select country", sorted(data["country"].unique()), index=2)
    horizon = st.slider("Forecast horizon (years)", 1, 10, 5)
    st.markdown("---")
    st.write("**Model:** Scikit-learn Linear Regression")
    st.write("**History:** 2010–2024 (sample data)")

country_data = data[data.country == country].sort_values("year")
last_year = int(country_data.year.max())
future_years = np.arange(last_year + 1, last_year + horizon + 1)

forecast_parts = []
for label, (col, unit) in METRICS.items():
    model = LinearRegression()
    X = country_data[["year"]]
    y = country_data[col]
    model.fit(X, y)
    predicted = model.predict(pd.DataFrame({"year": future_years}))
    predicted = np.maximum(predicted, 0)
    forecast_parts.append(pd.DataFrame({
        "country": country, "year": future_years,
        "indicator": label, "value": predicted, "unit": unit
    }))

forecast = pd.concat(forecast_parts, ignore_index=True)

# A transparent, illustrative relative score based on the latest sample values.
# Lower per-capita resource use/emissions is treated as lower pressure for this demo only.
latest = data[data.year == last_year].copy()
score_components = []
for col in ["energy_kwh_per_capita", "water_m3_per_capita", "carbon_tonnes_per_capita"]:
    low, high = latest[col].min(), latest[col].max()
    if high == low:
        component = pd.Series(100.0, index=latest.index)
    else:
        component = 100 * (high - latest[col]) / (high - low)
    score_components.append(component)
latest["demo_score"] = sum(score_components) / len(score_components)
score = float(latest.loc[latest.country == country, "demo_score"].iloc[0])
band = "Optimal" if score >= 80 else ("Moderate" if score >= 50 else "Critical")

st.subheader(f"Overview — {country}")
c1, c2, c3, c4 = st.columns(4)
latest_row = country_data.iloc[-1]
c1.metric("Energy use", f'{latest_row.energy_kwh_per_capita:,.0f}', "kWh per capita")
c2.metric("Water use", f'{latest_row.water_m3_per_capita:,.2f}', "m³ per capita")
c3.metric("Carbon emissions", f'{latest_row.carbon_tonnes_per_capita:,.2f}', "tonnes CO₂ per capita")
c4.metric("Illustrative sustainability score", f"{score:.0f}/100", band)

st.caption(
    "Score is a simple relative demo indicator: the three latest-year per-capita values are "
    "min-max normalized across the six sample countries and averaged, with lower values scoring higher. "
    "It is not an official sustainability index and is not a measure of resource sufficiency."
)

tab1, tab2, tab3 = st.tabs(["📈 Historical trends", "🔮 Future predictions", "📊 Compare countries"])

with tab1:
    metric_label = st.selectbox("Choose indicator", list(METRICS.keys()), key="history_metric")
    col, unit = METRICS[metric_label]
    fig = px.line(country_data, x="year", y=col, markers=True,
                  title=f"{metric_label} — {country}", labels={col: unit, "year": "Year"})
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    metric_label = st.selectbox("Choose forecast indicator", list(METRICS.keys()), key="forecast_metric")
    col, unit = METRICS[metric_label]
    historical = country_data[["year", col]].copy()
    historical["Type"] = "Historical sample"
    historical = historical.rename(columns={col: "value"})
    projected = forecast[forecast.indicator == metric_label][["year", "value"]].copy()
    projected["Type"] = "Model forecast"
    chart_data = pd.concat([historical, projected], ignore_index=True)
    fig = px.line(chart_data, x="year", y="value", color="Type", markers=True,
                  title=f"{horizon}-year forecast — {country}", labels={"value": unit, "year": "Year"})
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(projected.round(2), use_container_width=True, hide_index=True)
    st.download_button("⬇️ Download forecast CSV", forecast.to_csv(index=False).encode("utf-8"),
                       file_name=f"{country.lower().replace(' ','_')}_forecast.csv", mime="text/csv")

with tab3:
    comparison = latest[["country", "energy_kwh_per_capita", "water_m3_per_capita",
                         "carbon_tonnes_per_capita", "demo_score"]].sort_values("demo_score", ascending=False)
    st.dataframe(comparison.round(2), use_container_width=True, hide_index=True)
    fig = px.bar(comparison, x="country", y="demo_score", color="demo_score",
                 title="Illustrative relative sustainability score",
                 labels={"demo_score": "Demo score (0–100)", "country": "Country"})
    st.plotly_chart(fig, use_container_width=True)

st.subheader("How the prediction works")
st.write(
    "For each indicator, the app trains a separate Linear Regression model using the selected "
    "country's sample observations from 2010–2024. The year is the input feature and the resource "
    "indicator is the target. The fitted model estimates values for the selected future years."
)
st.info(
    "This is a fast academic prototype. Linear regression is a baseline, not an advanced "
    "time-series system. Forecasts are extrapolations from synthetic data and should not be "
    "interpreted as real country forecasts."
)

with st.expander("Project scope and next improvements"):
    st.markdown("""
    - Replace generated data with sourced World Bank Open Data, UN Data, or IEA observations.
    - Add data validation, missing-value handling, normalization, and feature engineering.
    - Compare Linear Regression, Decision Tree, and Random Forest using a chronological holdout set.
    - Add an API layer (FastAPI) and a more advanced dashboard if time permits.
    - Evaluate MAE, RMSE, and R² on unseen years before reporting model performance.
    """)
st.caption("Academic prototype | Python · Streamlit · Pandas · Scikit-learn · Plotly")
