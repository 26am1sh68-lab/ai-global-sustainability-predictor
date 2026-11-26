# Architecture

- **Presentation layer:** Streamlit dashboard (`app.py`)
- **Data layer:** CSV sample dataset (`data/sample_indicators.csv`) loaded with Pandas
- **ML layer:** Scikit-learn Linear Regression, trained separately for each indicator
- **Visualization:** Plotly interactive charts
- **Export:** Forecast CSV download from the dashboard

Flow: CSV → Pandas → per-indicator regression → future estimates → Streamlit/Plotly dashboard.
