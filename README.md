# temperature-data-analysis

# EDA

EDA исторических температурных данных проведена в ноутбуке EDA.ipynb. В ноутбуке рассмотрены следующие вопросы:

1. Написана функция для получения температурных статистик для города
2. Проведены эксперименты по параллельному расчету статистик (копия кода для запуска экспериментов вынесена в parallelism_test.py)
3. Написана функция для получения текущей температуры в городе с помощью API OpenWeatherMap в синхронном и асинхронном режимах
4. Проведены эксперименты по синхронному и асинхронному получению температуры в городах

# Streamlit

Код для веб-приложения на Streamlit находится в файле streamlit_app.py модуля temperature_data_analysis. Вспомогательные функции для работы с данными и их визуализации находятся в файле utils.py и plots.py.

Ссылка на веб-приложение: https://ai-applied-python-hw1-temperature-data-analysis.streamlit.app/
