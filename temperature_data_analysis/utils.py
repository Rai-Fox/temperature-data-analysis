import requests
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

CURRENT_WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"
month_to_season = {
    12: "winter",
    1: "winter",
    2: "winter",
    3: "spring",
    4: "spring",
    5: "spring",
    6: "summer",
    7: "summer",
    8: "summer",
    9: "autumn",
    10: "autumn",
    11: "autumn",
}


def calculate_stats_for_city(city_df: pd.DataFrame, window_size=30):
    city_df = city_df.copy()
    city_df = city_df.sort_values(by="timestamp")
    city_df["rolling_mean"] = city_df["temperature"].rolling(window=window_size).mean()
    city_df["rolling_std"] = city_df["temperature"].rolling(window=window_size).std()
    city_df["is_anomaly"] = (
        np.abs(city_df["temperature"] - city_df["rolling_mean"]) > 2 * city_df["rolling_std"]
    ).astype(int)

    season_profile = city_df.groupby("season")["temperature"].agg(["mean", "std", "min", "max"]).reset_index()
    global_stats = city_df["temperature"].agg(["mean", "std", "min", "max"]).reset_index()

    model = LinearRegression()
    model = model.fit(np.arange(len(city_df)).reshape(-1, 1), city_df["temperature"])
    trend_info = {"trend_coef": model.coef_[0], "is_positive_trend": (model.coef_[0] > 0).astype(int)}

    return city_df, season_profile, global_stats, trend_info


def calculate_cities_stats(df: pd.DataFrame):
    cities_stats = {}
    for city in df["city"].unique():
        city_df = df[df["city"] == city]
        city_df, season_profile, global_stats, trend_info = calculate_stats_for_city(city_df)
        cities_stats[city] = {
            "city_df": city_df,
            "season_profile": season_profile,
            "global_stats": global_stats,
            "trend_info": trend_info,
        }
    return cities_stats


def get_current_temperature(city, api_token):
    try:
        response = requests.get(CURRENT_WEATHER_URL, params={"q": city, "appid": api_token, "units": "metric"})
    except Exception as e:
        return None, None

    if not response.ok:
        return None, None

    temp = response.json()["main"]["temp"]
    season = month_to_season[datetime.fromtimestamp(response.json()["dt"]).month]
    return temp, season


def get_current_temperature_stats(city_stats, temp, season):
    season_profile = city_stats["season_profile"]
    global_stats = city_stats["global_stats"]
    trend_info = city_stats["trend_info"]

    season_mean = season_profile[season_profile["season"] == season]["mean"].values[0]
    season_std = season_profile[season_profile["season"] == season]["std"].values[0]
    season_min = season_profile[season_profile["season"] == season]["min"].values[0]
    season_max = season_profile[season_profile["season"] == season]["max"].values[0]
    global_mean = global_stats[global_stats["index"] == "mean"]["temperature"].values[0]
    global_std = global_stats[global_stats["index"] == "std"]["temperature"].values[0]
    global_min = global_stats[global_stats["index"] == "min"]["temperature"].values[0]
    global_max = global_stats[global_stats["index"] == "max"]["temperature"].values[0]
    trend_coef = trend_info["trend_coef"]
    is_positive_trend = trend_info["is_positive_trend"]
    is_season_anomaly = (temp < season_mean - 2 * season_std) or (temp > season_mean + 2 * season_std)
    is_global_anomaly = (temp < global_mean - 2 * global_std) or (temp > global_mean + 2 * global_std)

    return {
        "temp": temp,
        "season": season,
        "season_mean": season_mean,
        "season_std": season_std,
        "season_min": season_min,
        "season_max": season_max,
        "global_mean": global_mean,
        "global_std": global_std,
        "global_min": global_min,
        "global_max": global_max,
        "trend_coef": trend_coef,
        "is_positive_trend": is_positive_trend,
        "is_season_anomaly": is_season_anomaly,
        "is_global_anomaly": is_global_anomaly,
    }
