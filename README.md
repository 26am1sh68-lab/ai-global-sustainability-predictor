# AI-Based Global Resource & Sustainability Predictor

A beginner-friendly Streamlit prototype for exploring energy use, water use, carbon emissions, and simple forecasts.

## Run in VS Code

1. Install Python 3.10 or newer.
2. Open this folder in VS Code.
3. Open Terminal → New Terminal.
4. Create a virtual environment:

   Windows:
   ```powershell
   py -m venv .venv
   .venv\Scripts\activate
   ```

   macOS/Linux:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

5. Install packages:
   ```bash
   python -m pip install -r requirements.txt
   ```
6. Start the dashboard:
   ```bash
   streamlit run app.py
   ```
7. A browser tab should open at the local Streamlit address (usually http://localhost:8501).

## Important data note

`data/sample_indicators.csv` contains generated demonstration data, not verified official country statistics. Do not present it as World Bank/UN/IEA data. Replace it with sourced observations for a research-grade project.

## Model

The app trains one Scikit-learn LinearRegression model per indicator and country. It uses year as the input and the indicator value as the target, then extrapolates for the selected forecast horizon. Evaluate with a chronological holdout before reporting accuracy.

## Sustainability score

The displayed 0–100 score is an illustrative relative demo score. It min-max normalizes the latest-year energy, water, and carbon per-capita values across the six sample countries, then averages the three components (lower values receive higher scores). It is not an official sustainability index.

## Project alignment

The prototype covers the project goals of centralizing energy/water/carbon indicators, basic prediction, and interactive visualization. It intentionally omits production API, database, cloud deployment, and advanced LSTM/XGBoost models to meet a same-day build deadline.
