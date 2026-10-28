# Household Electricity Forecast & TNB Bill Estimator (Malaysia)

## Run
    pip install -r requirements.txt
    python build_artifacts.py      # trains the model, writes artifacts/ (re-run if your library versions differ)
    python test_tariff.py          # optional: checks the bill calculator against published worked examples
    streamlit run app.py

The 10-day outlook fetches live conditions from Open-Meteo, so it needs internet access.
Offline, it falls back to typical conditions automatically.

## Tabs
1. 10-Day Outlook - weather-app style cards: predicted kWh, range, indicative cost and conditions per day
2. Forecast - date-range forecast, what-if scenario, single-day check
3. Monthly Bill - TNB bill estimate with breakdown, 12-month projection, recommendations, savings what-if
4. Environmental Conditions - 10-day conditions, climate through the year, monthly heatmap, link to consumption, last 90 days
5. Insights - seasonality, weekday, holidays, temperature response
6. History - data explorer
7. About - method, backtest, tariff used, limitations

## Files
- `app.py` - dashboard (`app_v1.py` is the earlier forecast-only version)
- `tariff.py` - TNB tariff, AFA schedule, bill calculation, recommendations
- `weather_live.py` - Open-Meteo 10-day conditions
- `features.py`, `build_artifacts.py` - feature engineering and model training
- `test_tariff.py` - tariff sanity tests
- `data/`, `artifacts/` - merged daily dataset, saved model and backtest

## Updating the tariff
Edit the constants at the top of `tariff.py` (rates, `AFA_BY_MONTH`, relief period). Verify against
TNB's calculator at www.mytnb.com.my/tariff.
