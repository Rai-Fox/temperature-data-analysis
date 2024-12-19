import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from utils import month_to_season


def plot_temperature_hist(city_stats, city):
    fig = plt.figure(figsize=(12, 6))

    city_df = city_stats["city_df"]
    city_df["timestamp"] = pd.to_datetime(city_df["timestamp"])
    city_df = city_df.sort_values("timestamp")

    plt.scatter(city_df["timestamp"], city_df["temperature"], label="Температура", s=3)
    plt.plot(city_df["timestamp"], city_df["rolling_mean"], label="Скользящее среднее (30 дней)", color="orange")
    plt.scatter(
        city_df[city_df["is_anomaly"] == 1]["timestamp"],
        city_df[city_df["is_anomaly"] == 1]["temperature"],
        s=3,
        color="red",
        label="Аномалии",
    )

    plt.title(f"Температуры в городе {city}")
    plt.xlabel("Дата")
    plt.ylabel("Температура, °C")
    plt.legend()
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.gcf().autofmt_xdate()

    return fig


def plot_season_profile(season_profile, city):
    fig = plt.figure(figsize=(12, 6))

    # Отсортируем сезоны по порядку
    season_profile["season"] = pd.Categorical(season_profile["season"], ["winter", "spring", "summer", "autumn"])
    season_profile = season_profile.sort_values("season")

    plt.plot(season_profile["season"], season_profile["mean"], label="Средняя температура")
    plt.fill_between(
        season_profile["season"],
        season_profile["mean"] - 2 * season_profile["std"],
        season_profile["mean"] + 2 * season_profile["std"],
        alpha=0.3,
        label="Стандартное отклонение",
    )
    plt.scatter(season_profile["season"], season_profile["min"], label="Минимальная температура", color="red")
    plt.scatter(season_profile["season"], season_profile["max"], label="Максимальная температура", color="green")

    plt.title(f"Профиль сезонности в городе {city}")
    plt.xlabel("Сезон")
    plt.ylabel("Температура, °C")
    plt.legend()

    return fig


def plot_season_anomalies(city_stats, city):
    fig = plt.figure(figsize=(12, 6))

    city_df = city_stats["city_df"].copy()
    city_df["timestamp"] = pd.to_datetime(city_df["timestamp"])
    city_df = city_df.sort_values("timestamp")
    city_df["year"] = city_df["timestamp"].dt.year

    city_df["season_year"] = city_df["year"]

    winter_condition = city_df["season"] == "winter"
    december_condition = city_df["timestamp"].dt.month == 12
    city_df.loc[winter_condition & december_condition, "season_year"] = city_df["year"] + 1

    plt.scatter(city_df["timestamp"], city_df["temperature"], label="Температура", s=3)

    season_profile = city_stats["season_profile"]

    for i, row in season_profile.iterrows():
        s = row["season"]
        s_mean = row["mean"]
        s_std = row["std"]
        s_min = row["min"]
        s_max = row["max"]

        season_years = sorted(city_df.loc[city_df["season"] == s, "season_year"].unique())
        for j, sy in enumerate(season_years):
            season_data = city_df[(city_df["season"] == s) & (city_df["season_year"] == sy)].sort_values("timestamp")
            if season_data.empty:
                continue

            plt.scatter(season_data["timestamp"], season_data["temperature"], s=2, c="blue", alpha=0.5)

            plt.fill_between(
                season_data["timestamp"],
                s_mean - 2 * row["std"],
                s_mean + 2 * row["std"],
                alpha=0.3,
                color="gray",
                label=None,
            )

            anomalies = np.abs(season_data["temperature"] - s_mean) > 2 * s_std

            plt.scatter(
                season_data[anomalies]["timestamp"],
                season_data[anomalies]["temperature"],
                s=3,
                color="red",
                label="Аномалии" if i == 0 and j == 0 else None,
            )

    plt.title(f"Температуры в городе {city}")
    plt.xlabel("Дата")
    plt.ylabel("Температура, °C")
    plt.legend()
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.gcf().autofmt_xdate()

    return fig
