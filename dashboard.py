"""
Household Electricity Consumption Forecast - Malaysia (FYP dashboard) - v2
Run:  streamlit run app.py
"""
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import tariff as T
from features import MY_HOL, TARGET, WEATHER, add_features, find_data, holiday_name, weather_for
from weather_live import CITIES, fetch_daily_conditions

ROOT = Path(__file__).parent
MIN_D, MAX_D = date(2026, 7, 1), date(2027, 12, 31)
MONTH_OPTIONS = list(pd.period_range("2026-07", "2027-12", freq="M"))
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
LABELS = {"temperature_C": "Temperature (°C)", "humidity_pct": "Humidity (%)",
          "rainfall_mm_per_day": "Rainfall (mm/day)", "wind_speed_m_s": "Wind speed (m/s)"}
COLORS = {"green": "#2ca02c", "blue": "#1f77b4", "orange": "#ff7f0e", "red": "#d62728"}

st.set_page_config(page_title="Malaysia Household Electricity Forecast", page_icon="⚡", layout="wide")


# ================================================================== loaders & model helpers
@st.cache_resource
def load_bundle():
    return joblib.load(ROOT / "artifacts" / "model_bundle.joblib")


@st.cache_data
def load_history():
    return pd.read_csv(find_data(ROOT), parse_dates=["date"])


@st.cache_data
def load_backtest():
    return pd.read_csv(ROOT / "artifacts" / "backtest.csv", parse_dates=["date"])


bundle, hist, bt = load_bundle(), load_history(), load_backtest()
clim = bundle["clim"]


def predict(X: pd.DataFrame):
    mid = bundle["linear"].predict(X[["days_since_start"]]) + bundle["xgb"].predict(X[bundle["feats"]])
    return mid, mid + bundle["lo"], mid + bundle["hi"]


def scenario_frame(dates, d_temp=0.0, d_hum=0.0, rain_mult=1.0, wind_mult=1.0):
    w = weather_for(dates, clim)
    w["temperature_C"] += d_temp
    w["humidity_pct"] = (w["humidity_pct"] + d_hum).clip(0, 100)
    w["rainfall_mm_per_day"] *= rain_mult
    w["wind_speed_m_s"] *= wind_mult
    return add_features(w)


@st.cache_data
def typical_all():
    """Typical-conditions forecast for every selectable day (per average household)."""
    d = pd.Series(pd.date_range(MIN_D, MAX_D, freq="D"))
    X = scenario_frame(d)
    X["pred"], _, _ = predict(X)
    return X[["date", "pred"]]


@st.cache_data
def typical_monthly():
    t = typical_all()
    return t.groupby(t["date"].dt.to_period("M"))["pred"].sum()


@st.cache_data
def temperature_sensitivity():
    d = pd.Series(pd.date_range(MIN_D, date(2027, 6, 30), freq="D"))
    rows = [{"Temperature change (°C)": dt, "Avg kWh/household/day": predict(scenario_frame(d, d_temp=float(dt)))[0].mean()}
            for dt in np.arange(-2, 2.01, 0.5)]
    return pd.DataFrame(rows)


@st.cache_data(ttl=1800, show_spinner=False)
def live_conditions(city: str) -> pd.DataFrame:
    lat, lon = CITIES[city]
    return fetch_daily_conditions(lat, lon, 10)


# ================================================================== sidebar (global settings)
avg_month_kwh = hist.tail(365)[TARGET].mean() * 365.25 / 12

with st.sidebar:
    st.header("⚙️ Settings")
    factor = st.slider("My household vs average household", 0.3, 3.0, 1.0, 0.05,
                       help="1.0 = the average household in the data. 1.5 = uses 50% more electricity.")
    st.caption(f"Average household ≈ {avg_month_kwh:,.0f} kWh/month. Yours ≈ {avg_month_kwh * factor:,.0f} kWh/month.")

    with st.expander("Environmental scenario (Forecast & Bill)"):
        d_temp = st.slider("Temperature change (°C)", -3.0, 3.0, 0.0, 0.5)
        d_hum = st.slider("Humidity change (% points)", -10, 10, 0, 1)
        rain_mult = st.slider("Rainfall (× typical)", 0.5, 2.0, 1.0, 0.1)
        wind_mult = st.slider("Wind speed (× typical)", 0.5, 2.0, 1.0, 0.1)

    with st.expander("Tariff assumptions"):
        thr_label = st.selectbox("Limit for AFA / retail charge / service-tax exemption",
                                 ["Auto (800 kWh until Dec 2026, then 600)", "Always 600 kWh", "Always 800 kWh"])
        thr_mode = {"Auto (800 kWh until Dec 2026, then 600)": "auto", "Always 600 kWh": "600",
                    "Always 800 kWh": "800"}[thr_label]
        tier_all = st.selectbox("Generation rate above 1,500 kWh",
                                ["37.03 sen applies to all kWh", "37.03 sen only on kWh above 1,500"]).startswith("37.03 sen applies")
        afa_on = st.checkbox("Override AFA for every month")
        afa_val = st.number_input("AFA override (sen/kWh)", -10.0, 10.0, 3.0, 0.1, disabled=not afa_on)
        afa_later = st.number_input("Assumed AFA when none is published (sen/kWh)", -10.0, 10.0,
                                    float(T.AFA_DEFAULT_LATER), 0.1)
    afa_over = float(afa_val) if afa_on else None

    st.markdown("**10-day outlook**")
    use_live = st.checkbox("Use live weather forecast", value=True,
                           help="Fetches the next 10 days of conditions from Open-Meteo. Falls back to typical conditions if offline.")
    city = st.selectbox("Location", list(CITIES), disabled=not use_live)
    st.caption(f"Tariff rates last checked {T.TARIFF_CHECKED}.")


def month_cfg(period):
    thr = T.exemption_threshold(period, thr_mode)
    afa, src = T.afa_for(period, afa_over, afa_later)
    return thr, afa, src


def bill_for(kwh, period):
    thr, afa, _ = month_cfg(period)
    return T.compute_bill(kwh, afa, thr, tier_all)


def eff_rate_rm(period):
    """Indicative RM per kWh in a month (bill / kWh) for the user's household."""
    tm = typical_monthly()
    p = min(max(period, tm.index[0]), tm.index[-1])
    kwh = tm.loc[p] * factor
    return bill_for(kwh, p)["total"] / kwh if kwh else 0.0


def sky(rain, temp):
    if rain >= 15:
        return "🌧️"
    if rain >= 5:
        return "🌦️"
    return "☀️" if temp >= 28 else "⛅"


# ================================================================== header & tabs
st.title("⚡ Household Electricity Forecast & TNB Bill Estimator")
st.caption(f"Malaysia · daily electricity use per household (kWh) · model trained on data up to {bundle['trained_until']}")

(tab_10, tab_fc, tab_bill, tab_env, tab_ins, tab_hist, tab_model) = st.tabs(
    ["🌤️ 10-Day Outlook", "📈 Forecast", "💵 Monthly Bill", "🌦️ Environmental Conditions",
     "💡 Insights", "🗂️ History", "🧠 About"])

# ================================================================== 10-DAY OUTLOOK (shared frame)
today = date.today()
dates10 = pd.Series(pd.date_range(today, periods=10, freq="D"))
typ10 = weather_for(dates10, clim)
w10, source10, live_ok = typ10.copy(), "Typical conditions for this time of year", False
if use_live:
    try:
        live = live_conditions(city).set_index("date").reindex(dates10.values)
        for c in WEATHER:
            w10[c] = np.where(np.isnan(live[c].values), typ10[c].values, live[c].values)
        source10, live_ok = f"Live forecast for {city} (Open-Meteo)", True
    except Exception:
        source10 = "Typical conditions for this time of year (live forecast unavailable)"
X10 = add_features(w10)
X10["pred"], X10["lower"], X10["upper"] = predict(X10)
Xt10 = add_features(typ10)
typ_pred10, _, _ = predict(Xt10)

with tab_10:
    st.subheader("Next 10 days")
    st.caption(f"Conditions: {source10}")
    if use_live and not live_ok:
        st.warning("Could not reach the live weather service, so typical conditions are used instead.")
    if today > MAX_D:
        st.warning("Today is beyond the period the model was validated for; treat these numbers with extra caution.")

    rates = [eff_rate_rm(pd.Period(d, "M")) for d in X10["date"]]
    cost10 = X10["pred"].values * factor * np.array(rates)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total, 10 days", f"{X10['pred'].sum() * factor:,.0f} kWh",
              f"{(X10['pred'].sum() - typ_pred10.sum()) * factor:+.1f} vs typical conditions" if live_ok else None)
    k2.metric("Indicative cost, 10 days", f"RM {cost10.sum():,.2f}")
    top = X10["pred"].idxmax()
    k3.metric("Highest day", X10.loc[top, "date"].strftime("%a %d %b"), f"{X10.loc[top, 'pred'] * factor:.1f} kWh", delta_color="off")
    k4.metric("Average per day", f"{X10['pred'].mean() * factor:.1f} kWh")

    for row in range(2):
        cols = st.columns(5)
        for j, col in enumerate(cols):
            i = row * 5 + j
            r = X10.iloc[i]
            with col.container(border=True):
                st.markdown(f"**{'Today' if i == 0 else r['date'].strftime('%a')}**  \n{r['date'].strftime('%d %b')}")
                st.markdown(f"<div style='font-size:2.1rem;line-height:1.15'>{sky(r['rainfall_mm_per_day'], r['temperature_C'])}</div>",
                            unsafe_allow_html=True)
                st.markdown(f"### {r['pred'] * factor:.1f} kWh")
                st.caption(f"{r['lower'] * factor:.1f}–{r['upper'] * factor:.1f} · ≈ RM {cost10[i]:.2f}")
                st.caption(f"🌡️ {r['temperature_C']:.1f}°C · 💧 {r['humidity_pct']:.0f}%  \n🌧️ {r['rainfall_mm_per_day']:.1f} mm · 💨 {r['wind_speed_m_s']:.1f} m/s")
                hn = holiday_name(r["date"])
                if hn:
                    st.caption(f"🎉 {hn}")

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    err_up = (X10["upper"] - X10["pred"]).values * factor
    err_dn = (X10["pred"] - X10["lower"]).values * factor
    fig.add_trace(go.Bar(x=X10["date"], y=X10["pred"] * factor, name="Predicted kWh",
                         error_y=dict(type="data", array=err_up, arrayminus=err_dn, thickness=1.2),
                         marker_color="#1f77b4"), secondary_y=False)
    fig.add_trace(go.Scatter(x=X10["date"], y=X10["temperature_C"], name="Temperature (°C)", mode="lines+markers",
                             line=dict(color="#ff7f0e")), secondary_y=True)
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", y=1.12))
    fig.update_yaxes(title_text="kWh per day", secondary_y=False)
    fig.update_yaxes(title_text="°C", secondary_y=True, showgrid=False)
    st.plotly_chart(fig, width="stretch")
    st.caption("Bars show the predicted consumption with the ~90% range. Cost is indicative: your projected monthly bill "
               "divided by projected monthly kWh, applied to each day. TNB bills monthly, not daily.")

# ================================================================== FORECAST
with tab_fc:
    st.subheader("Forecast and what-if scenarios")
    c1, c2 = st.columns(2)
    start = c1.date_input("From", value=MIN_D, min_value=MIN_D, max_value=MAX_D, key="fc_start")
    end = c2.date_input("To", value=date(2027, 6, 30), min_value=MIN_D, max_value=MAX_D, key="fc_end")
    st.caption("Environmental scenario and household size are set in the sidebar.")

    if end < start:
        st.error("'To' date must be on or after 'From' date.")
    else:
        dates = pd.Series(pd.date_range(start, end, freq="D"))
        X = scenario_frame(dates, d_temp, d_hum, rain_mult, wind_mult)
        X["pred"], X["lower"], X["upper"] = predict(X)
        for c in ["pred", "lower", "upper"]:
            X[c] = X[c] * factor
        base_pred = predict(scenario_frame(dates))[0] * factor

        avg, delta = X["pred"].mean(), X["pred"].mean() - base_pred.mean()
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Average per day", f"{avg:.2f} kWh",
                  f"{delta:+.2f} vs typical" if (d_temp or d_hum or rain_mult != 1 or wind_mult != 1) else None)
        k2.metric("Total for the period", f"{X['pred'].sum():,.0f} kWh")
        k3.metric("Highest month", MONTHS[X.groupby("month")["pred"].mean().idxmax() - 1])
        k4.metric("Uncertainty (~90%)", f"{bundle['lo'] * factor:+.2f} / {bundle['hi'] * factor:+.2f} kWh/day")

        past = hist[(hist["date"] >= pd.Timestamp(start) - pd.Timedelta(days=365)) & (hist["date"] < pd.Timestamp(start))]
        fig = go.Figure()
        if len(past):
            fig.add_trace(go.Scatter(x=past["date"], y=past[TARGET] * factor, name="Actual (history)",
                                     line=dict(color="#888", width=1)))
        fig.add_trace(go.Scatter(x=X["date"], y=X["upper"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=X["date"], y=X["lower"], fill="tonexty", line=dict(width=0),
                                 fillcolor="rgba(31,119,180,0.18)", name="~90% band", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=X["date"], y=X["pred"], name="Forecast", line=dict(color="#1f77b4", width=1.6)))
        fig.update_layout(height=400, margin=dict(l=10, r=10, t=20, b=10), yaxis_title="kWh per day",
                          hovermode="x unified", legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, width="stretch")

        monthly = (X.assign(Month=X["date"].dt.to_period("M").astype(str))
                     .groupby("Month").agg(Days=("date", "size"), **{"Avg kWh/day": ("pred", "mean")}))
        monthly["Total kWh"] = monthly["Days"] * monthly["Avg kWh/day"]
        st.dataframe(monthly.round(2), width="stretch")

        out = X[["date", "pred", "lower", "upper"] + WEATHER].rename(
            columns={"pred": "forecast_kwh", "lower": "lower_90", "upper": "upper_90"})
        num = out.columns.drop("date")
        out[num] = out[num].round(3)
        st.download_button("⬇️ Download forecast (CSV)", out.to_csv(index=False).encode(), "forecast.csv", "text/csv")
        st.caption("Uses typical environmental conditions for each day of the year, shifted by your scenario.")

    with st.expander("📅 Check one specific day with your own environmental conditions"):
        day = st.date_input("Date", value=date(2026, 12, 15), min_value=MIN_D, max_value=MAX_D, key="day")
        typ = clim.loc[pd.Timestamp(day).dayofyear]
        e1, e2, e3, e4 = st.columns(4)
        temp = e1.number_input(LABELS["temperature_C"], 15.0, 40.0, round(float(typ["temperature_C"]), 1), 0.1, key=f"t_{day}")
        hum = e2.number_input(LABELS["humidity_pct"], 0.0, 100.0, round(float(typ["humidity_pct"]), 1), 0.5, key=f"h_{day}")
        rain = e3.number_input(LABELS["rainfall_mm_per_day"], 0.0, 300.0, round(float(typ["rainfall_mm_per_day"]), 1), 0.5, key=f"r_{day}")
        wind = e4.number_input(LABELS["wind_speed_m_s"], 0.0, 30.0, round(float(typ["wind_speed_m_s"]), 1), 0.1, key=f"w_{day}")
        one = add_features(pd.DataFrame({"date": [pd.Timestamp(day)], "temperature_C": [temp], "humidity_pct": [hum],
                                         "rainfall_mm_per_day": [rain], "wind_speed_m_s": [wind]}))
        p, lo, hi = (v[0] * factor for v in predict(one))
        tp = predict(add_features(weather_for(pd.Series([pd.Timestamp(day)]), clim)))[0][0] * factor
        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted consumption", f"{p:.2f} kWh", f"{p - tp:+.2f} vs typical conditions")
        m2.metric("Likely range (~90%)", f"{lo:.2f} – {hi:.2f} kWh")
        hn = holiday_name(day)
        m3.metric("Day type", DAYS[pd.Timestamp(day).dayofweek] + (" · holiday" if hn else ""), hn if hn else None, delta_color="off")
        if pd.Timestamp(day).day == 1:
            st.info("The 1st of a month shows an artificial jump in this dataset, so this prediction is less reliable.")

# ================================================================== MONTHLY BILL
with tab_bill:
    st.subheader("Estimated monthly TNB electricity bill")
    default_p = pd.Period(min(max(today, MIN_D), MAX_D), "M")
    period = st.selectbox("Billing month", MONTH_OPTIONS, index=MONTH_OPTIONS.index(default_p),
                          format_func=lambda p: p.strftime("%B %Y"))

    days = pd.Series(pd.date_range(period.start_time.normalize(), period.end_time.normalize(), freq="D"))
    Xm = scenario_frame(days, d_temp, d_hum, rain_mult, wind_mult)
    mid, lo_d, hi_d = predict(Xm)
    kwh, kwh_lo, kwh_hi = mid.sum() * factor, lo_d.sum() * factor, hi_d.sum() * factor
    thr, afa, afa_src = month_cfg(period)
    bill = T.compute_bill(kwh, afa, thr, tier_all)
    b_lo = T.compute_bill(kwh_lo, afa, thr, tier_all)["total"]
    b_hi = T.compute_bill(kwh_hi, afa, thr, tier_all)["total"]

    # 12-month projection starting from the chosen month
    proj_periods = [p for p in MONTH_OPTIONS if period <= p < period + 12]
    proj = []
    for p in proj_periods:
        dd = pd.Series(pd.date_range(p.start_time.normalize(), p.end_time.normalize(), freq="D"))
        k = predict(scenario_frame(dd, d_temp, d_hum, rain_mult, wind_mult))[0].sum() * factor
        b = bill_for(k, p)
        proj.append({"period": p, "Month": p.strftime("%b %Y"), "kWh": k, "Bill (RM)": b["total"],
                     "level": T.level(k, month_cfg(p)[0])})
    proj = pd.DataFrame(proj)
    usual_bill = proj["Bill (RM)"].mean()

    lab, colr = T.level(kwh, thr)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Projected usage", f"{kwh:,.0f} kWh")
    k2.metric("Estimated bill", f"RM {bill['total']:,.2f}")
    k3.metric("Average cost", f"{bill['sen_per_kwh']:.1f} sen/kWh")
    k4.metric("Usage level", lab)
    st.markdown(f"Status: :{colr}[**{lab}**] · conservative range RM {b_lo:,.2f} – RM {b_hi:,.2f} "
                f"(assumes daily errors do not cancel out)")
    relief = " (temporary relief until 31 Dec 2026)" if thr == 800 else ""
    st.caption(f"This month: AFA {afa:+.2f} sen/kWh ({afa_src}) · AFA, retail charge and service tax are waived up to "
               f"{thr} kWh{relief}.")

    left, right = st.columns([3, 2])
    with left:
        parts = bill["parts"]
        energy = parts["Generation"] + parts["Capacity"] + parts["Network"]
        items = [("Energy (generation + capacity + network)", energy), ("Retail charge", parts["Retail charge"]),
                 ("AFA", parts["AFA"]), ("Energy Efficiency Incentive", parts["Energy Efficiency Incentive"]),
                 ("RE levy (KWTBB)", parts["RE levy (KWTBB)"]), ("Service tax (8%)", parts["Service tax (8%)"])]
        wf = go.Figure(go.Waterfall(
            orientation="v", measure=["relative"] * len(items) + ["total"],
            x=[n for n, _ in items] + ["Total"], y=[v for _, v in items] + [bill["total"]],
            text=[f"{v:+.2f}" for _, v in items] + [f"{bill['total']:.2f}"], textposition="outside",
            decreasing=dict(marker=dict(color="#2ca02c")), increasing=dict(marker=dict(color="#d62728")),
            totals=dict(marker=dict(color="#1f77b4"))))
        wf.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10), yaxis_title="RM", showlegend=False)
        st.plotly_chart(wf, width="stretch")
    with right:
        tbl = pd.DataFrame({"Component": list(parts.keys()) + ["Total"], "RM": list(parts.values()) + [bill["total"]]})
        st.dataframe(tbl, hide_index=True, width="stretch")
        if thr == 800 and kwh > 600 and kwh <= 800:
            st.success(f"The temporary relief saves you about RM{T.compute_bill(kwh, afa, 600, tier_all)['total'] - bill['total']:,.2f} "
                       "this month compared with the normal 600 kWh limit.")

    st.markdown("#### Next 12 months")
    colors = [COLORS[c] for _, c in proj["level"]]
    f12 = make_subplots(specs=[[{"secondary_y": True}]])
    f12.add_trace(go.Bar(x=proj["Month"], y=proj["Bill (RM)"], marker_color=colors, name="Estimated bill (RM)",
                         text=proj["Bill (RM)"].round(0).astype(int), textposition="outside"), secondary_y=False)
    f12.add_trace(go.Scatter(x=proj["Month"], y=proj["kWh"], name="kWh", mode="lines+markers",
                             line=dict(color="#555")), secondary_y=True)
    f12.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", y=1.15))
    f12.update_yaxes(title_text="RM", secondary_y=False)
    f12.update_yaxes(title_text="kWh", secondary_y=True, showgrid=False)
    st.plotly_chart(f12, width="stretch")
    peak_m = proj.loc[proj["Bill (RM)"].idxmax()]
    st.caption(f"Highest projected bill: {peak_m['Month']} (RM {peak_m['Bill (RM)']:,.2f}). "
               f"Bar colours: green = low, blue = moderate, orange = above exemption limit, red = high. "
               f"AFA is only published about a month ahead, later months use your assumption in the sidebar.")

    st.markdown("#### Recommendations")
    for sev, title, text in T.recommendations(kwh, bill, afa, thr, tier_all, usual_bill):
        getattr(st, sev)(f"**{title}**  \n{text}")

    cl = T.cliffs(kwh, afa, thr, tier_all)
    if cl:
        st.markdown("**Thresholds within reach**")
        st.dataframe(pd.DataFrame(cl).rename(columns={"target": "If usage drops to", "cut_kwh": "Cut (kWh)", "cut_pct": "Cut (%)",
                                                      "bill_after": "New bill (RM)", "saving": "Saving (RM)"}),
                     hide_index=True, width="stretch")

    st.markdown("**What if I use less?**")
    cut = st.slider("Reduce usage by (%)", 0, 40, 10, 1, key="cut")
    nb = T.compute_bill(kwh * (1 - cut / 100), afa, thr, tier_all)["total"]
    w1, w2, w3 = st.columns(3)
    w1.metric("New usage", f"{kwh * (1 - cut / 100):,.0f} kWh", f"-{kwh * cut / 100:,.0f} kWh", delta_color="off")
    w2.metric("New bill", f"RM {nb:,.2f}", f"{nb - bill['total']:+,.2f}", delta_color="inverse")
    w3.metric("Saving per year (if sustained)", f"RM {(bill['total'] - nb) * 12:,.0f}")
    with st.expander("Savings at a glance"):
        st.dataframe(pd.DataFrame(T.savings_table(kwh, afa, thr, tier_all)), hide_index=True, width="stretch")

    st.markdown("**Tips to lower your usage**")
    for tip in T.GENERAL_TIPS["high" if lab in ("Above exemption limit", "High", "Very high") else "normal"]:
        st.markdown(f"- {tip}")
    st.caption("The bill is an estimate for an ordinary domestic account in Peninsular Malaysia on the standard tariff "
               f"(rates checked {T.TARIFF_CHECKED}). Verify with TNB's official calculator at mytnb.com.my/tariff. "
               "Tips are general guidance, not model outputs.")

# ================================================================== ENVIRONMENTAL CONDITIONS
with tab_env:
    st.subheader("Environmental conditions")

    st.markdown("#### Next 10 days")
    st.caption(f"Source: {source10}")
    f = make_subplots(rows=2, cols=2, subplot_titles=[LABELS[w] for w in WEATHER], vertical_spacing=0.18)
    f.add_trace(go.Scatter(x=X10["date"], y=X10["temperature_C"], mode="lines+markers", line=dict(color="#d62728")), 1, 1)
    f.add_trace(go.Scatter(x=X10["date"], y=X10["humidity_pct"], mode="lines+markers", line=dict(color="#1f77b4")), 1, 2)
    f.add_trace(go.Bar(x=X10["date"], y=X10["rainfall_mm_per_day"], marker_color="#17becf"), 2, 1)
    f.add_trace(go.Scatter(x=X10["date"], y=X10["wind_speed_m_s"], mode="lines+markers", line=dict(color="#7f7f7f")), 2, 2)
    f.update_layout(height=430, showlegend=False, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(f, width="stretch")

    st.markdown("#### Typical climate through the year")
    v1, v2 = st.columns(2)
    var = v1.selectbox("Variable", WEATHER, format_func=lambda k: LABELS[k], key="env_var")
    yrs = sorted(hist["date"].dt.year.unique())
    yr = v2.selectbox("Compare with year", yrs, index=yrs.index(2025), key="env_year")
    hh = hist.assign(doy=hist["date"].dt.dayofyear)
    g = hh.groupby("doy")[var]
    band = pd.DataFrame({"mean": g.mean(), "p10": g.quantile(0.1), "p90": g.quantile(0.9)}).reindex(range(1, 367)).interpolate()
    band = band.rolling(15, center=True, min_periods=1).mean()
    one_year = hh[hh["date"].dt.year == yr].set_index("doy")[var].rolling(7, center=True, min_periods=1).mean()
    xd = pd.to_datetime("2025-01-01") + pd.to_timedelta(band.index - 1, unit="D")
    fc_ = go.Figure()
    fc_.add_trace(go.Scatter(x=xd, y=band["p90"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fc_.add_trace(go.Scatter(x=xd, y=band["p10"], fill="tonexty", line=dict(width=0), fillcolor="rgba(31,119,180,0.18)",
                             name="Normal range (10th-90th percentile)", hoverinfo="skip"))
    fc_.add_trace(go.Scatter(x=xd, y=band["mean"], name="Average 2018-2026", line=dict(color="#1f77b4")))
    fc_.add_trace(go.Scatter(x=pd.to_datetime("2025-01-01") + pd.to_timedelta(one_year.index - 1, unit="D"), y=one_year.values,
                             name=str(yr), line=dict(color="#ff7f0e", width=1.2)))
    fc_.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10), yaxis_title=LABELS[var], xaxis_tickformat="%b",
                      legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fc_, width="stretch")

    st.markdown("#### Monthly averages by year")
    piv = hist.assign(year=hist["date"].dt.year, m=hist["date"].dt.month).pivot_table(index="year", columns="m", values=var)
    hm = px.imshow(piv.values, x=[MONTHS[m - 1] for m in piv.columns], y=[str(y) for y in piv.index], aspect="auto",
                   color_continuous_scale="RdYlBu_r" if var == "temperature_C" else "Blues", labels=dict(color=LABELS[var]))
    hm.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(hm, width="stretch")

    st.markdown("#### How conditions relate to consumption")
    rel = hist.copy()
    rel["relative"] = rel[TARGET] / rel.groupby(rel["date"].dt.year)[TARGET].transform("mean")
    rel = rel[rel["date"].dt.day != 1]
    binned = (rel.assign(bin=pd.qcut(rel[var], 10, duplicates="drop"))
                 .groupby("bin", observed=True).agg(x=(var, "mean"), y=("relative", "mean")))
    sc = px.scatter(rel, x=var, y="relative", opacity=0.25, labels={var: LABELS[var], "relative": "Consumption vs yearly average"})
    sc.add_trace(go.Scatter(x=binned["x"], y=binned["y"], mode="lines+markers", name="Average per decile",
                            line=dict(color="#d62728", width=2)))
    sc.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", y=1.12))
    st.plotly_chart(sc, width="stretch")
    corr = rel[WEATHER + ["relative"]].corr()["relative"].drop("relative").rename("Correlation").round(3)
    st.dataframe(corr.rename(index=LABELS).to_frame(), width="stretch")
    st.caption("Correlation with consumption relative to its yearly average (1st-of-month days excluded). "
               "Values close to 0 mean a weak relationship in this dataset.")

    st.markdown("#### Last 90 days")
    rec = hist.tail(90)
    f4 = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04, subplot_titles=[LABELS[w] for w in WEATHER])
    for i, w in enumerate(WEATHER, start=1):
        trace = go.Bar(x=rec["date"], y=rec[w]) if w == "rainfall_mm_per_day" else go.Scatter(x=rec["date"], y=rec[w], mode="lines")
        f4.add_trace(trace, row=i, col=1)
    f4.update_layout(height=620, showlegend=False, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(f4, width="stretch")

# ================================================================== INSIGHTS
with tab_ins:
    st.subheader("What drives household consumption in this data?")
    h = hist.copy()
    h["year"] = h["date"].dt.year
    full = h[h["year"] <= 2025]
    month_idx = (full[TARGET] / full.groupby("year")[TARGET].transform("mean")).groupby(full["date"].dt.month).mean()
    h["rel7"] = h[TARGET] / h[TARGET].rolling(7, center=True).mean()
    nd = h[h["date"].dt.day != 1]
    wk_idx = nd.groupby(nd["date"].dt.dayofweek)["rel7"].mean()
    is_hol = nd["date"].dt.date.isin(MY_HOL)
    hol_idx = pd.Series({"Holiday": nd.loc[is_hol, "rel7"].mean(), "Normal day": nd.loc[~is_hol, "rel7"].mean()})

    i1, i2 = st.columns(2)
    with i1:
        f = px.bar(x=MONTHS, y=month_idx.values, labels={"x": "", "y": "Index (1.00 = yearly average)"}, title="Seasonal pattern (2018-2025)")
        f.add_hline(y=1.0, line_dash="dot", line_color="grey")
        f.update_yaxes(range=[min(0.9, month_idx.min() - 0.02), max(1.1, month_idx.max() + 0.02)])
        st.plotly_chart(f, width="stretch")
    with i2:
        f = px.bar(x=DAYS, y=wk_idx.values, labels={"x": "", "y": "Index (1.00 = weekly average)"}, title="Day-of-week pattern")
        f.add_hline(y=1.0, line_dash="dot", line_color="grey")
        f.update_yaxes(range=[min(0.97, wk_idx.min() - 0.01), max(1.03, wk_idx.max() + 0.01)])
        st.plotly_chart(f, width="stretch")
    i3, i4 = st.columns(2)
    with i3:
        sens = temperature_sensitivity()
        st.plotly_chart(px.line(sens, x="Temperature change (°C)", y="Avg kWh/household/day", markers=True,
                                title="Model response to temperature (all else typical)"), width="stretch")
    with i4:
        f = px.bar(x=hol_idx.index, y=hol_idx.values, labels={"x": "", "y": "Index (1.00 = surrounding week)"},
                   title="Public holidays vs normal days")
        f.add_hline(y=1.0, line_dash="dot", line_color="grey")
        f.update_yaxes(range=[0.9, 1.1])
        st.plotly_chart(f, width="stretch")

    lo_t = sens.iloc[sens["Temperature change (°C)"].sub(-1).abs().idxmin()]["Avg kWh/household/day"]
    hi_t = sens.iloc[sens["Temperature change (°C)"].sub(1).abs().idxmin()]["Avg kWh/household/day"]
    per_deg = (hi_t - lo_t) / 2
    ty = typical_all()
    top_months = ty.groupby(ty["date"].dt.month)["pred"].mean().sort_values(ascending=False).head(3).index
    st.markdown(
        f"- **Seasonality is the strongest pattern**: highest around **{MONTHS[int(month_idx.idxmax()) - 1]}**, "
        f"lowest around **{MONTHS[int(month_idx.idxmin()) - 1]}**.\n"
        f"- **Temperature:** +1 °C changes average daily consumption by about **{per_deg:+.2f} kWh** "
        f"({per_deg / sens['Avg kWh/household/day'].mean() * 100:+.1f}%), modest compared with seasonality and trend.\n"
        f"- **Forecast peak months** (typical conditions): **{', '.join(MONTHS[m - 1] for m in sorted(top_months))}**.\n"
        f"- **Long-term trend:** about **{bundle['trend_per_year']:+.2f} kWh/household/day per year**."
    )
    st.info("Personalised bill recommendations are in the **Monthly Bill** tab.")

# ================================================================== HISTORY
with tab_hist:
    st.subheader("Historical data explorer")
    lo_d, hi_d = hist["date"].min().date(), hist["date"].max().date()
    r1, r2, r3 = st.columns(3)
    a = r1.date_input("From", value=date(2025, 1, 1), min_value=lo_d, max_value=hi_d, key="h_from")
    b = r2.date_input("To", value=hi_d, min_value=lo_d, max_value=hi_d, key="h_to")
    overlay = r3.selectbox("Overlay environmental variable", ["None"] + [LABELS[w] for w in WEATHER])
    sub = hist[(hist["date"] >= pd.Timestamp(a)) & (hist["date"] <= pd.Timestamp(b))]
    if sub.empty:
        st.warning("No data in this range.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sub["date"], y=sub[TARGET], name="kWh per household", line=dict(color="#1f77b4", width=1.3)))
        if overlay != "None":
            col = [k for k, v in LABELS.items() if v == overlay][0]
            fig.add_trace(go.Scatter(x=sub["date"], y=sub[col], name=overlay, yaxis="y2", line=dict(color="#ff7f0e", width=1), opacity=0.7))
            fig.update_layout(yaxis2=dict(title=overlay, overlaying="y", side="right", showgrid=False))
        fig.update_layout(height=400, margin=dict(l=10, r=10, t=20, b=10), hovermode="x unified",
                          yaxis_title="kWh per household per day", legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, width="stretch")
        q1, q2, q3 = st.columns(3)
        q1.metric("Average per day", f"{sub[TARGET].mean():.2f} kWh")
        q2.metric("Highest day", f"{sub[TARGET].max():.2f} kWh", sub.loc[sub[TARGET].idxmax(), "date"].strftime("%d %b %Y"), delta_color="off")
        q3.metric("Lowest day", f"{sub[TARGET].min():.2f} kWh", sub.loc[sub[TARGET].idxmin(), "date"].strftime("%d %b %Y"), delta_color="off")
        st.download_button("⬇️ Download this range (CSV)", sub.to_csv(index=False).encode(), "history.csv", "text/csv")

# ================================================================== ABOUT
with tab_model:
    st.subheader("How the forecast is made")
    st.markdown(
        "- **Target:** daily electricity consumption per household (kWh) = national consumption ÷ number of households.\n"
        "- **Inputs:** environmental conditions (temperature, humidity, rainfall, wind speed) and calendar information "
        "(month, day of week, public holidays, day of month).\n"
        "- **Model:** a linear trend plus an XGBoost model for seasonal and calendar patterns.\n"
        "- **Validation:** trained on 2018-2024, tested on Jan 2025 – Jun 2026 without using the actual conditions of the test period.\n"
        "- **10-day outlook:** uses a live forecast of the four environmental variables when available, otherwise typical conditions. "
        "The model was trained on a Malaysia-wide average, so a single city's forecast is an approximation."
    )
    st.markdown("**Backtest results (test period)**")
    st.dataframe(bundle["backtest"], width="stretch", hide_index=True)
    st.caption("MAE/RMSE in kWh per household per day. Bias > 0 means the model under-predicts on average.")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bt["date"], y=bt["actual"], name="Actual", line=dict(color="black", width=1.1)))
    fig.add_trace(go.Scatter(x=bt["date"], y=bt["predicted"], name="Predicted", line=dict(color="#ff7f0e", width=1)))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10), title="Backtest: actual vs predicted",
                      yaxis_title="kWh per household per day", hovermode="x unified", legend=dict(orientation="h", y=1.15))
    st.plotly_chart(fig, width="stretch")
    imp = bundle["importance"].sort_values()
    f = px.bar(x=imp.values, y=imp.index, orientation="h", labels={"x": "Importance", "y": ""}, title="Feature importance (XGBoost part)")
    f.update_layout(height=420)
    st.plotly_chart(f, width="stretch")

    st.subheader("Tariff used for the bill")
    st.markdown(
        f"TNB domestic tariff since 1 July 2025 (rates checked {T.TARIFF_CHECKED}): generation 27.03 sen/kWh (37.03 above 1,500 kWh), "
        "capacity 4.55, network 12.85, retail charge RM10/month, monthly AFA, an Energy Efficiency Incentive of up to 25 sen/kWh for "
        "usage up to 1,000 kWh, 1.6% RE levy above 300 kWh and 8% service tax on usage above 600 kWh. Retail charge, AFA and service tax "
        "are waived up to 600 kWh, raised to 800 kWh from September to 31 December 2026."
    )
    afa_tbl = pd.DataFrame([{"Month": k, "AFA (sen/kWh)": v[0], "Status": v[1]} for k, v in T.AFA_BY_MONTH.items()
                            if k >= "2026-05"])
    st.dataframe(afa_tbl, hide_index=True, width="stretch")

    st.subheader("Limitations")
    st.warning(
        "**This dashboard uses synthetic consumption data.** The daily series was generated from the real monthly OpenDOSM "
        "electricity data (Prophet, SARIMAX and PCHIP interpolation), so accuracy figures describe the model on this synthetic "
        "data, not on real household meters.\n\n"
        "- Bills are estimates for a standard domestic account in Peninsular Malaysia. Rates, AFA and relief measures change; "
        "AFA after the latest published month is an assumption you can change in the sidebar.\n"
        "- How the 37.03 sen generation rate applies above 1,500 kWh is reported differently by different sources, so it is a setting.\n"
        "- Forecasts assume the long-term trend continues in a straight line and use typical conditions beyond the 10-day window.\n"
        "- The 1st day of each month has an artificial jump, which lowers accuracy on those days.\n"
        "- Public holidays are national only (state holidays are not included) and some Islamic holiday dates are approximate."
    )