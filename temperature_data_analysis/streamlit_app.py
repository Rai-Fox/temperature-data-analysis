import streamlit as st
import pandas as pd


from utils import (
    calculate_cities_stats,
    get_current_temperature,
    get_current_temperature_stats,
)
from plots import plot_temperature_hist, plot_season_profile, plot_season_anomalies


def process_app():
    st.set_page_config(
        layout="wide",
        initial_sidebar_state="auto",
        page_title="Temperature data analysis",
    )
    city, api_token, df = show_side_bar_inputs()
    show_main_page(city, api_token, df)


def show_upload_data_form():
    uploaded_file = st.sidebar.file_uploader("Выберите файл с данными", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        return df
    return None


def show_side_bar_inputs():
    st.sidebar.header("Настройки")

    st.sidebar.subheader("Загрузка данных")
    df = show_upload_data_form()

    if df is None:
        return None, None, None

    st.sidebar.subheader("Выбор города")
    city = st.sidebar.text_input("Введите название города", "Moscow")

    st.sidebar.subheader("API ключ")
    api_token = st.sidebar.text_input("Введите API ключ OpenWeatherMap")

    return city, api_token, df


def show_current_temperature_analysis(city_stats, city, api_token):
    if api_token is None or api_token == "":
        return
    st.subheader("Анализ текущей температуры")

    temp, season = get_current_temperature(city, api_token)
    if temp is not None:
        st.write(f"Текущая температура в городе **{city}**: **{temp}°C**, сезон: **{season}**")
    else:
        st.write("Ошибка при загрузке данных. Проверьте название города и API ключ")
        return

    current_temperature_stats = get_current_temperature_stats(city_stats, temp, season)

    with st.expander("Статистика по температуре:"):
        st.write(f"Средняя температура в сезоне: **{current_temperature_stats['season_mean']:.2f}°C**")
        st.write(f"Стандартное отклонение в сезоне: **{current_temperature_stats['season_std']:.2f}°C**")
        st.write(f"Минимальная температура в сезоне: **{current_temperature_stats['season_min']:.2f}°C**")
        st.write(f"Максимальная температура в сезоне: **{current_temperature_stats['season_max']:.2f}°C**")
        st.write(f"Средняя температура за все время: **{current_temperature_stats['global_mean']:.2f}°C**")
        st.write(f"Стандартное отклонение за все время: **{current_temperature_stats['global_std']:.2f}°C**")
        st.write(f"Минимальная температура за все время: **{current_temperature_stats['global_min']:.2f}°C**")
        st.write(f"Максимальная температура за все время: **{current_temperature_stats['global_max']:.2f}°C**")
        st.write(f"Коэффициент тренда: **{current_temperature_stats['trend_coef']:.5f}**")

        st.write(
            f"Судя по коэффициенту тренда, температура в городе {city} имеет {'положительный' if current_temperature_stats['is_positive_trend'] else 'отрицательный'} тренд"
        )

        if current_temperature_stats["is_season_anomaly"]:
            st.write("Текущая температура **является аномальной** для сезона")
        if current_temperature_stats["is_global_anomaly"]:
            st.write("Текущая температура **является аномальной** по сравнению со всеми данными")


def show_hist_data_analysis(city_stats, city):
    st.subheader("Анализ исторических данных")

    with st.expander("История температур"):
        st.write(f"График температур в городе {city}")
        fig = plot_temperature_hist(city_stats, city)
        st.pyplot(fig)

    with st.expander("Статистика по сезонам"):
        st.write(f"Профиль сезонности в городе {city}")
        fig = plot_season_profile(city_stats["season_profile"], city)
        st.pyplot(fig)

    with st.expander("Анализ аномалий в разрезе сезонов"):
        st.write("Аномалии в разрезе сезонов")
        fig = plot_season_anomalies(city_stats, city)
        st.pyplot(fig)


def show_main_page(city, api_token, df):
    st.header("Анализ температурных данных и мониторинг текущей температуры через OpenWeatherMap API")

    if df is not None:
        with st.expander("Загруженные данные"):
            st.write(df)

    if city is None or city == "":
        return

    city_stats = calculate_cities_stats(df)[city]

    show_current_temperature_analysis(city_stats, city, api_token)

    show_hist_data_analysis(city_stats, city)


if __name__ == "__main__":
    process_app()
