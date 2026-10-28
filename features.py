"""Shared feature engineering (used by build_artifacts.py and app.py)."""
from pathlib import Path

import numpy as np
import pandas as pd
import holidays

START = pd.Timestamp("2018-01-01")
TARGET = "kwh_per_household"
WEATHER = ["temperature_C", "humidity_pct", "rainfall_mm_per_day", "wind_speed_m_s"]

# Malaysian national public holidays (state-level holidays not included)
MY_HOL = holidays.country_holidays("MY", years=range(2018, 2029))

# Version A features (environmental conditions + calendar). The upward trend is handled
# separately by a linear model, so days_since_start is NOT in this list.
FEATS = [
    "temperature_C", "humidity_pct", "rainfall_mm_per_day", "wind_speed_m_s",
    "temp_x_humidity", "rain_3d_mean",
    "month", "dayofweek", "is_weekend", "is_holiday",
    "day_of_month", "is_month_start",
    "month_sin", "month_cos", "doy_sin", "doy_cos",
]
FEATS_DIRECT = FEATS + ["days_since_start"]


def add_features(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy().sort_values("date").reset_index(drop=True)
    d["month"] = d["date"].dt.month
    d["dayofweek"] = d["date"].dt.dayofweek
    d["dayofyear"] = d["date"].dt.dayofyear
    d["is_weekend"] = (d["dayofweek"] >= 5).astype(int)
    d["is_holiday"] = d["date"].dt.date.isin(MY_HOL).astype(int)
    d["day_of_month"] = d["date"].dt.day
    d["is_month_start"] = (d["day_of_month"] == 1).astype(int)
    d["month_sin"] = np.sin(2 * np.pi * d["month"] / 12)
    d["month_cos"] = np.cos(2 * np.pi * d["month"] / 12)
    d["doy_sin"] = np.sin(2 * np.pi * d["dayofyear"] / 365.25)
    d["doy_cos"] = np.cos(2 * np.pi * d["dayofyear"] / 365.25)
    d["days_since_start"] = (d["date"] - START).dt.days
    d["temp_x_humidity"] = d["temperature_C"] * d["humidity_pct"] / 100
    d["rain_3d_mean"] = d["rainfall_mm_per_day"].rolling(3, min_periods=1).mean()
    return d


def climatology(hist: pd.DataFrame, window: int = 15) -> pd.DataFrame:
    """Typical environmental conditions for each day of the year (smoothed, circular)."""
    clim = hist.groupby(hist["date"].dt.dayofyear)[WEATHER].mean()
    clim = clim.reindex(range(1, 367)).interpolate().bfill().ffill()
    pad = window // 2
    wrapped = pd.concat([clim.iloc[-pad:], clim, clim.iloc[:pad]])
    return wrapped.rolling(window, center=True).mean().iloc[pad:-pad]


def weather_for(dates: pd.Series, clim: pd.DataFrame) -> pd.DataFrame:
    dates = pd.Series(pd.to_datetime(dates)).reset_index(drop=True)
    w = clim.loc[dates.dt.dayofyear.values].reset_index(drop=True)
    return pd.concat([dates.rename("date"), w], axis=1)


def holiday_name(d) -> str | None:
    return MY_HOL.get(pd.Timestamp(d).date())


DATA_NAME = "malaysia_daily_merged_2018_2026.csv"


def find_data(root: Path) -> Path:
    """Look for the merged dataset next to the scripts, in a data/ subfolder, or in the current folder."""
    root = Path(root)
    candidates = [root / "data" / DATA_NAME, root / DATA_NAME, root.parent / DATA_NAME,
                  Path.cwd() / "data" / DATA_NAME, Path.cwd() / DATA_NAME]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Cannot find {DATA_NAME}. Put it in the same folder as the scripts (or in a 'data' "
                            "subfolder). Looked in:\n  " + "\n  ".join(str(p) for p in candidates))