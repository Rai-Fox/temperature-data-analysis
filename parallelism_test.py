from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


def calc_time(func):
    def wrapper(*args, **kwargs):
        import time
        time_sum = 0
        n = 10
        for _ in range(n):
            start = time.time()
            result = func(*args, **kwargs)
            time_sum += time.time() - start
        print(f"Average execution time: {time_sum / n:.2f} seconds")
        return result
    return wrapper


def async_calc_time(func):
    async def wrapper(*args, **kwargs):
        import time
        time_sum = 0
        n = 10
        for _ in range(n):
            start = time.time()
            result = await func(*args, **kwargs)
            time_sum += time.time() - start
        print(f"Average execution time: {time_sum / n:.2f} seconds")
        return result
    return wrapper


def calculate_stats_for_city(city_df: pd.DataFrame, window_size=30): 
    city_df = city_df.copy()
    city_df = city_df.sort_values(by="timestamp")
    city_df["rolling_mean"] = city_df["temperature"].rolling(window=window_size).mean()
    city_df["rolling_std"] = city_df["temperature"].rolling(window=window_size).std()
    city_df["is_anomaly"] = (np.abs(city_df["temperature"] - city_df["rolling_mean"]) > 2 * city_df["rolling_std"]).astype(int)

    season_profile = city_df.groupby("season")["temperature"].agg(["mean", "std", "min", "max"]).reset_index()
    global_stats = city_df["temperature"].agg(["mean", "std", "min", "max"]).reset_index()
    
    model = LinearRegression()
    model = model.fit(np.arange(len(city_df)).reshape(-1, 1), city_df["temperature"])
    trend_info = {
        "trend_coef": model.coef_[0],
        "is_positive_trend": (model.coef_[0] > 0).astype(int)
    }

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
            "trend_info": trend_info
        }
    return cities_stats


def parallel_calculate_cities_stats(df: pd.DataFrame, n_cores=10):
    if n_cores >= 20:
        raise ValueError("Too many cores")
    with ProcessPoolExecutor(n_cores) as executor:
        tasks = []
        for city in df["city"].unique():
            city_df = df[df["city"] == city]
            tasks.append(city_df)
        
        results = executor.map(calculate_stats_for_city, tasks)

        cities_stats = {}
        for city_df, season_profile, global_stats, trend_info in results:
            city = city_df["city"].iloc[0]
            cities_stats[city] = {
                "city_df": city_df,
                "season_profile": season_profile,
                "global_stats": global_stats,
                "trend_info": trend_info
            }        
    
    return cities_stats


def parallel_w_created_executor_calculate_cities_stats(executor: ProcessPoolExecutor, df: pd.DataFrame):
    tasks = []
    for city in df["city"].unique():
        city_df = df[df["city"] == city]
        tasks.append(city_df)
    
    results = executor.map(calculate_stats_for_city, tasks)

    cities_stats = {}
    for city_df, season_profile, global_stats, trend_info in results:
        city = city_df["city"].iloc[0]
        cities_stats[city] = {
            "city_df": city_df,
            "season_profile": season_profile,
            "global_stats": global_stats,
            "trend_info": trend_info
        }        
    
    return cities_stats


@calc_time
def test_calculate_cities_stats(df):
    cities_stats = calculate_cities_stats(df)

@calc_time
def test_parallel_calculate_cities_stats(df, n_cores):
    cities_stats = parallel_calculate_cities_stats(df, n_cores)

@calc_time
def test_parallel_w_created_executor_calculate_cities_stats(executor, df):
    cities_stats = parallel_w_created_executor_calculate_cities_stats(executor, df)


def main():
    data = pd.read_csv('temperature_data.csv')
    print('Processing data without parallelism')
    cities_stats = test_calculate_cities_stats(data)
    print()

    for n_cores in [1, 2, 5, 10]:
        print(f"Processing data with {n_cores} cores")
        cities_stats = test_parallel_calculate_cities_stats(data, n_cores)
        print()

    for n_cores in [1, 2, 5, 10]:
        print(f"Processing data with {n_cores} cores using created executor")
        with ProcessPoolExecutor(n_cores) as executor:
            cities_stats = test_parallel_w_created_executor_calculate_cities_stats(executor, data)
        print()


if __name__ == "__main__":
    main()