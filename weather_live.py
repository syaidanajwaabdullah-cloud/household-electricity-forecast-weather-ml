"""Optional live 10-day environmental forecast from Open-Meteo (free, no API key).

Returns the same four variables the model was trained on, as daily values:
temperature (°C, mean), humidity (%, mean), rainfall (mm/day, total), wind speed (m/s, mean).
If the request fails (offline, blocked network), the app falls back to typical conditions.
"""
from __future__ import annotations

import pandas as pd
import requests

URL = "https://api.open-meteo.com/v1/forecast"

CITIES = {
    "Kuala Lumpur": (3.1390, 101.6869),
    "Ipoh": (4.5975, 101.0901),
    "George Town (Penang)": (5.4141, 100.3288),
    "Alor Setar": (6.1248, 100.3678),
    "Kota Bharu": (6.1254, 102.2381),
    "Kuala Terengganu": (5.3302, 103.1408),
    "Kuantan": (3.8077, 103.3260),
    "Seremban": (2.7258, 101.9424),
    "Melaka": (2.1896, 102.2501),
    "Johor Bahru": (1.4927, 103.7414),
    "Kuching": (1.5535, 110.3593),
    "Kota Kinabalu": (5.9804, 116.0735),
}


def fetch_daily_conditions(lat: float, lon: float, days: int = 10, timeout: int = 6) -> pd.DataFrame:
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
        "wind_speed_unit": "ms", "timezone": "Asia/Kuala_Lumpur", "forecast_days": days,
    }
    r = requests.get(URL, params=params, timeout=timeout)
    r.raise_for_status()
    h = pd.DataFrame(r.json()["hourly"])
    h["date"] = pd.to_datetime(h["time"]).dt.normalize()
    daily = h.groupby("date").agg(
        temperature_C=("temperature_2m", "mean"),
        humidity_pct=("relative_humidity_2m", "mean"),
        rainfall_mm_per_day=("precipitation", "sum"),
        wind_speed_m_s=("wind_speed_10m", "mean"),
    ).reset_index()
    return daily.head(days)