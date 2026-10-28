import datetime as dt
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from malaysia_tnb_bill import EEI_BANDS, calc_bill, eei_rate_sen

st.set_page_config(page_title="Malaysia Household Electricity Forecast", page_icon="⚡", layout="wide")

# ---- Model performance on the hold-out test set (Jan 2025 - Jun 2026). Edit if you retrain. ----
TEST_MAE_KWH = 14.7
TEST_MAPE_PCT = 3.8

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


@st.cache_resource
def load_model():
    return joblib.load("malaysia_final_model.pkl")


@st.cache_data
def load_climate():
    """Monthly average environmental conditions from the training data (optional file)."""
    p = Path("malaysia_model_ready_dataset.csv")
    if not p.exists():
        return None
    d = pd.read_csv(p)
    cols = ["avg_temperature_c", "avg_humidity_pct", "avg_rainfall_mm_per_day", "avg_wind_speed_ms"]
    return d.groupby("month")[cols].mean()


def make_row(year, month, temp, hum, rain, wind):
    return pd.DataFrame([{
        "year": year,
        "month_sin": np.sin(2 * np.pi * month / 12),
        "month_cos": np.cos(2 * np.pi * month / 12),
        "avg_temperature_c": temp,
        "avg_humidity_pct": hum,
        "avg_rainfall_mm_per_day": rain,
        "avg_wind_speed_ms": wind,
    }])


saved = load_model()
model, FEATURES, RANGES = saved["model"], saved["features"], saved["ranges"]
climate = load_climate()


def predict(year, month, temp, hum, rain, wind) -> float:
    return float(model.predict(make_row(year, month, temp, hum, rain, wind)[FEATURES])[0])


# ------------------------------------------------------------------ sidebar
st.sidebar.header("Tariff settings")
st.sidebar.caption("TNB Tariff A (Peninsular Malaysia, domestic), structure from 1 Jul 2025.")
protect = st.sidebar.selectbox(
    "Retail/AFA/SST exemption threshold (kWh)", [800, 600], index=0,
    help="Raised from 600 to 800 kWh for Sep-31 Dec 2026. Use 600 for other periods.")
afa = st.sidebar.number_input(
    "AFA (sen/kWh)", value=3.67, step=0.01, format="%.2f",
    help="Set monthly by the Energy Commission. 3.67 is the Sep 2026 rate. Update from mytnb.com.my/tariff.")
st.sidebar.caption("AFA only applies above the exemption threshold, so it has no effect on small bills.")

# ------------------------------------------------------------------ header
st.title("⚡ Malaysian Household Electricity Forecast")
st.write("Estimate average monthly electricity use and bill for one household from environmental conditions.")

today = dt.date.today()
c1, c2 = st.columns(2)
year = c1.number_input("Year", min_value=2018, max_value=2035, value=today.year, step=1)
month = c2.selectbox("Month", range(1, 13), index=today.month - 1, format_func=lambda m: MONTHS[m - 1])

defaults = {}
for key, fallback in [("avg_temperature_c", 27.0), ("avg_humidity_pct", 82.0),
                      ("avg_rainfall_mm_per_day", 6.0), ("avg_wind_speed_ms", 1.5)]:
    defaults[key] = float(climate.loc[month, key]) if climate is not None else fallback

st.subheader("Environmental Conditions")
s1, s2, s3, s4 = st.columns(4)


def slider(col, label, key, unit, step):
    lo, hi = RANGES[key]
    return col.slider(f"{label} ({unit})", float(np.floor(lo - 1)), float(np.ceil(hi + 1)),
                      round(defaults[key], 2), step)


temp = slider(s1, "Temperature", "avg_temperature_c", "°C", 0.1)
hum = slider(s2, "Humidity", "avg_humidity_pct", "%", 0.5)
rain = slider(s3, "Rainfall", "avg_rainfall_mm_per_day", "mm/day", 0.1)
wind = slider(s4, "Wind speed", "avg_wind_speed_ms", "m/s", 0.1)

for label, key, val in [("Temperature", "avg_temperature_c", temp), ("Humidity", "avg_humidity_pct", hum),
                        ("Rainfall", "avg_rainfall_mm_per_day", rain), ("Wind speed", "avg_wind_speed_ms", wind)]:
    lo, hi = RANGES[key]
    if val < lo or val > hi:
        st.warning(f"{label} is outside the range the model was trained on ({lo:.1f} to {hi:.1f}). "
                   "The forecast is less reliable here.")
if year > 2027:
    st.warning("Forecast years well beyond the training data rely on extending the historical trend.")

# ------------------------------------------------------------------ prediction
kwh = predict(year, month, temp, hum, rain, wind)
bill = calc_bill(kwh, afa, protect)

st.divider()
m1, m2, m3 = st.columns(3)
m1.metric("Forecast consumption", f"{kwh:,.0f} kWh/household")
m2.metric("Estimated monthly bill", f"RM {bill['Total']:,.2f}")
m3.metric("Effective rate", f"{bill['Total'] / max(bill['kwh'], 1) * 100:,.1f} sen/kWh")
st.caption(f"Typical model error on unseen months: about ±{TEST_MAE_KWH:.0f} kWh (MAPE {TEST_MAPE_PCT}%). "
           "The bill is an estimate for an average household, not a quote.")

tab1, tab2, tab3 = st.tabs(["Bill breakdown", "Insights", "About"])

with tab1:
    rows = [(k, v) for k, v in bill.items() if k not in ("kwh", "Total")]
    df_b = pd.DataFrame(rows, columns=["Component", "RM"])
    df_b.loc[len(df_b)] = ["Total", bill["Total"]]
    st.dataframe(df_b.style.format({"RM": "{:,.2f}"}), hide_index=True, width="stretch")
    st.caption(f"Energy Efficiency Incentive band: {eei_rate_sen(bill['kwh']):.2f} sen/kWh "
               f"applied to all {bill['kwh']} kWh.")

with tab2:
    st.markdown("#### 1. How much does temperature matter?")
    delta = 0.5
    hi_kwh = predict(year, month, min(temp + delta, RANGES['avg_temperature_c'][1] + 1), hum, rain, wind)
    lo_kwh = predict(year, month, max(temp - delta, RANGES['avg_temperature_c'][0] - 1), hum, rain, wind)
    st.write(f"Holding everything else fixed, a **{delta}°C** rise is associated with about "
             f"**{(hi_kwh - lo_kwh) / 2:+.1f} kWh** more per household per month "
             f"(≈ RM {calc_bill(kwh + (hi_kwh - lo_kwh) / 2, afa, protect)['Total'] - bill['Total']:+.2f}). "
             "Temperature was the only environmental factor with a consistent effect in the data; "
             "humidity, rainfall and wind effects were weak or unstable, so they are not interpreted individually.")

    st.markdown("#### 2. Seasonal pattern (model-implied, for the selected year)")
    seasonal = []
    for m in range(1, 13):
        if climate is not None:
            c = climate.loc[m]
            seasonal.append(predict(year, m, c["avg_temperature_c"], c["avg_humidity_pct"],
                                    c["avg_rainfall_mm_per_day"], c["avg_wind_speed_ms"]))
        else:
            seasonal.append(predict(year, m, temp, hum, rain, wind))
    chart = pd.DataFrame({"kWh per household": seasonal}, index=MONTHS)
    st.line_chart(chart)
    peak_m = MONTHS[int(np.argmax(seasonal))]
    low_m = MONTHS[int(np.argmin(seasonal))]
    st.caption(f"Under typical conditions, consumption peaks around {peak_m} and is lowest around {low_m}.")

    st.markdown("#### 3. Suggestions")
    tips = []
    if bill["kwh"] > 300 and bill["kwh"] <= 600:
        tips.append("You are under 600 kWh, so retail charge, AFA and service tax are waived. "
                    "Staying under this level keeps those exemptions.")
    if bill["kwh"] > protect:
        tips.append(f"Above {protect} kWh, retail charge, AFA and service tax apply. "
                    "Cutting usage back under this level removes them.")
    nxt = [u for u, _ in EEI_BANDS if u >= bill["kwh"]]
    if nxt and nxt[0] - bill["kwh"] <= 15:
        tips.append(f"You are within {nxt[0] - bill['kwh']} kWh of the next Energy Efficiency Incentive band; "
                    "crossing it lowers the discount on every kWh.")
    if bill["kwh"] > 1000:
        tips.append("Above 1,000 kWh the Energy Efficiency Incentive is zero. Solar or efficiency upgrades "
                    "pay back fastest in this range.")
    tips.append("Air-conditioning is usually the largest load. Setting it to 25-26°C with a fan and "
                "servicing filters regularly lowers consumption without much loss of comfort.")
    for t in tips:
        st.write("• " + t)

with tab3:
    st.markdown(f"""
**Model:** Ridge regression (standardised inputs) trained on monthly national data, Mar 2018 - Jun 2026,
using year, month (cyclical), temperature, humidity, rainfall and wind speed to predict average kWh per household.

**Evaluation:** time-ordered hold-out (train Mar 2018 - Dec 2024, test Jan 2025 - Jun 2026).
MAE ≈ {TEST_MAE_KWH} kWh, MAPE ≈ {TEST_MAPE_PCT}%. The deployed model is refit on all months.

**Limitations:** monthly national averages smooth out weather effects; the model systematically
under-predicts unusually high months; no household-level or state-level detail.

**Tariff:** TNB domestic tariff (generation + capacity + network, Energy Efficiency Incentive, AFA, KWTBB, service tax)
as published for Peninsular Malaysia. Verify rates against your bill and mytnb.com.my/tariff.
""")