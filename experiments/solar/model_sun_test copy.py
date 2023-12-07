import pandas as pd
import lightgbm as lgb
import pickle
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from matplotlib.pyplot import subplots
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score as cvs
from sklearn.metrics import mean_squared_error
from refactor_data_weather import refactor_date


<<<<<<< Updated upstream:experiments/model_weather.py
def clear_date(valid):
    # Valid=True - датасет с вырезанными днями (так сказать ручная валидация)
    # Valid=False исходный датасет
    if valid:
        filename_weather = "data\\station_1\\Station_1_weather_clear_06_06.csv"
        filename_energy = "data\\station_1\\fact_year_custom.csv"
        valid = "valid"
    else:
        filename_weather = "data\\station_1\\Station_1_weather_clear_06_06.csv"
        filename_energy = "data\\station_1\\fact_year_original.csv"
        valid = ""

    # импорт данных
    data_x = pd.read_csv(filename_weather, sep=";")
    data_y = pd.read_csv(filename_energy, sep=",", index_col=0)
=======
def clear_date():

    # импорт данных
    filename_weather = 'data\\station_1\\Station_1_weather_clear_06_06.csv'
    filename_energy = 'data\\station_1\\fact_year_custom.csv'
    data_x = pd.read_csv(filename_weather, sep=';')
    data_y = pd.read_csv(filename_energy, sep=',', index_col=0)
>>>>>>> Stashed changes:experiments/model_sun_test copy.py

    # удаление дупликатов по параметру дататайм, с оставлением последнего (второго) экземпляра
    data_x = data_x.drop_duplicates(subset=["dt"], keep="last")

    # преобразование даты и времени в тип данных дататайм64
    data_x.loc[:, "dt"] = pd.to_datetime(data_x["dt"])
    data_y.loc[:, "dt"] = pd.to_datetime(data_y["dt"])

    # слияние матрицы признаков и матрицы целевых переменных
<<<<<<< Updated upstream:experiments/model_weather.py
    data = pd.merge(data_y, data_x, on="dt")
    print(data.head())
=======
    data = pd.merge(data_y, data_x, on='dt')

>>>>>>> Stashed changes:experiments/model_sun_test copy.py
    # срез изключающий из выборки последние 10 дней для анализа
    # data = pd.merge(data_y, data_x, on='dt').iloc[:-240]

    # создание доп матрицы признаков из дататайм
    dayofyear = data["dt"].dt.dayofyear
    hours = data["dt"].dt.hour
    month = data["dt"].dt.month
    days_hours = pd.concat(
        [dayofyear, hours, month],
        axis=1,
        join="inner",
        keys=["dayofyear", "hours", "month"],
    )
    print(days_hours.head())
    # data.to_csv("test_date.csv")
<<<<<<< Updated upstream:experiments/model_weather.py

    # удаление явно ненужных sdfстолбцов из матрицы признаков
    data = data.drop(
        ["id", "gtpp", "load_time", "predict", "Visibility", "Surface_pressure"], axis=1
    )
=======
    
    # удаление явно ненужных столбцов из матрицы признаков
    data = data.drop(['id', 'gtpp', 'load_time', 'predict', 'Visibility', 'Surface_pressure'], axis=1)
>>>>>>> Stashed changes:experiments/model_sun_test copy.py
    data = pd.concat([data, days_hours], axis=1)

    # зимой станцию заметает снегом.
    data = data.drop(data[data["fact"].values <= 10].index)

    data = data.drop(["month"], axis=1)

    # очистка данных от лишних символов (не нужно)
    data.replace("[^a-zA-Z0-9]", " ", regex=True)
    data = data.rename(columns=lambda x: re.sub("[^A-Za-z0-9_]+:[C]:", "", x))
    filename_data = refactor_date(data)

    import seaborn as sns
    corr = data.corr()
    sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)   

    return filename_data


def main(valid=False, new_data=True):
    filename = "data_full"

    # Если True, то чистим дату
    if new_data:
        filename = clear_date(valid)

    data = pd.read_csv(f"data\Station_1\{filename}.csv", sep=",", index_col=["dt"])

    corr = data.corr()
    sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values)

    plt.show()
    # выделение матриц признаков и целевых переменных (целевая переменная 'fact')
    y_train = data[data.columns[0]]
    X_train = data[data.columns[1:]]

    # формирование набора для обучения и тестирования
<<<<<<< Updated upstream:experiments/model_weather.py
    X_train, X_test, y_train, y_test = train_test_split(
        X_train, y_train, test_size=0.01, random_state=42
    )
=======
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.01, random_state=42) 
>>>>>>> Stashed changes:experiments/model_sun_test copy.py

    # формирование набора с учетом таймсерий (TODO: проработать)
    tscv = TimeSeriesSplit()

    # описание параметров модели TODO: попробовать другие метрики, другие гипепараметры
    # TODO: попробовать optuna
    my_lgb = lgb.LGBMRegressor(metric="rmse")
    param_search = {
        "num_leaves": [10, 20, 31, 40, 50, 100],
        "max_depth": [-1, 3, 5],
        "min_data_in_leaf": [10, 20, 30],
        "lambda_l1": [0, 0.1, 0.2],
        "lambda_l2": [0, 0.1, 0.2],
    }

    # формирование набора с учетом таймсерий (TODO: проработать)
    tscv = TimeSeriesSplit()

    # конфигурирование модели
    gsearch = GridSearchCV(estimator=my_lgb, cv=tscv, param_grid=param_search)

    # обучение модели
    # gsearch.fit(X_train, y_train)
    best_lgb = gsearch.best_estimator_

    # оценка модели с лучшими параметрами mse метрикой TODO: проработать
    cv_scores = cvs(
        best_lgb, X_train, y_train, cv=tscv, n_jobs=-1, scoring="neg_mean_squared_error"
    )
    print("cv_scores", cv_scores)

    model = gsearch
    # прогноз выработки
    y_pred = model.predict(X_test)
    # сохранение модели с лучшими параметрами
    n_args = len(X_train.columns)
    pickle.dump(
        model, open(f"models\\station_1\\model_v5_{n_args}_args_{valid}.sav", "wb")
    )

    # скоринг TODO: что за скоринг?
    result = model.score(X_test, y_test)
    print("score", result)

    # среднеквадратичная ошибка предсказания
    print("mse", mean_squared_error(y_test, y_pred, squared=False))

    # отклонение прогноза в долях единицы для графика
    difference = (y_pred - y_test) / y_test

    # накопление отклонения (через создание списка)
    dif_sum = []
    for x in range(0, len(y_pred)):
        if x <= 60:
            dif_sum.append(np.sum(difference.iloc[0:x]))
        else:
            dif_sum.append(np.sum(difference.iloc[x - 60 : x]))

    dif_sum = pd.Series(dif_sum)

    # для графика
    day_tr_start, day_tr_finish = 0, len(data)

    # отрисовка плотов
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax2 = ax1.twinx()
    ax1.bar(np.arange(0, len(y_test)), y_test, label="Real Power")
    ax1.bar(np.arange(0, len(y_test)), y_pred, alpha=0.7, label="Predicted Power")
    ax2.plot(
        np.arange(0, len(y_test)), difference, label="dif", alpha=0.99, color="red"
    )
    ax1.legend(loc="upper left", fontsize=10)
    ax1.set_title("Difference")
    ax1.set_ylabel("power_value")
    fig.autofmt_xdate(rotation=45)

    plt.show()

<<<<<<< Updated upstream:experiments/model_weather.py

# new_data=True если датасет нужно очистить с помощью clear_data. В clear_data прописать путь
# new_data=False если датасет уже очищен
main(valid=True, new_data=False)
# main()
=======
main()
>>>>>>> Stashed changes:experiments/model_sun_test copy.py
