import pandas as pd
import re
from sklearn.metrics import *
from datetime import timedelta
import numpy as np

"""Приводим датасет погоды и выработки к необходимому формату и ретерним его"""

# добавляет в датафрейм строки
def insert_row(row_number, df, row_value):

    start_upper = 0
    end_upper = row_number
    start_lower = row_number
    end_lower = df.shape[0]

    upper_half = [*range(start_upper, end_upper, 1)]
    lower_half = [*range(start_lower, end_lower, 1)]

    lower_half = [x.__add__(1) for x in lower_half]

    index_ = upper_half + lower_half
    df.index = index_

    df.loc[row_number] = row_value

    df = df.sort_index()

    return df


# заполняем строки с пропущенным часом и np.nan в признаках
def refactor_date(filename_weather):

    data_x = pd.read_csv(filename_weather, sep=";")

    data_x.loc[:, "dt"] = pd.to_datetime(data_x["dt"])
    count = 0
    l = len(data_x)
    try:
        for n in range(1, l + 500):
            if data_x["dt"][n - 1].hour == 10 and data_x["dt"][n].hour == 12:
                row_number = n
                row_value = [
                    np.nan,
                    np.nan,
                    (data_x["dt"][n - 1]) + timedelta(hours=1),
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ]
                data_x = insert_row(row_number, data_x, row_value)
                count += 1
    except KeyError:
        print(f"IndexError because l={l} and new_l = {len(data_x)}")

    # Заполнение NaN
    # data_x = pd.to_datetime(data_x['dt'])

    list = [
        "id",
        "gtpp",
        "dt",
        "load_time",
        "predict",
        "10_metre_V_wind_component",
        "Snow_density",
        "Snowfall",
        "Visibility",
        "Surface_pressure",
        "Convective_precipitation",
        "Visual_cloud_cover",
        "Total_cloud_cover",
        "Precipitation_type",
        "Instantaneous_10_metre_wind_gust",
        "Medium_cloud_cover",
        "Total_precipitation_rate",
        "Convective_available_potential_energy",
        "10_metre_U_wind_component",
        "Skin_temperature",
        "2_metre_temperature",
        "Surface_solar_radiation_downwards",
        "Wind_speed",
        "Low_cloud_cover",
        "Snow_depth",
        "High_cloud_cover",
        "Evaporation",
        "Wind_Direction",
        "2_metre_dewpoint_temperature",
        "Total_precipitation",
        "2_metre_relative_humidity",
        "Clear_sky_direct_solar_radiation_at_surface",
        "Snow_height",
    ]

    # заполняем пропуски
    for x in list:
        # data_x[x] = data_x[x].fillna(data_x[x].median())
        data_x[x] = data_x[x].fillna(method="ffill")

    # создаем и заполняем список со строками, подлежащими удалению
    del_rows = []
    for n in range(2, len(data_x)):
        if (
            data_x["dt"][n].hour != data_x["dt"][n - 1].hour + 1
            and data_x["dt"][n - 1].hour != 23
            and data_x["dt"][n] != 0
        ):
            p = n - 1
            while data_x["dt"][p].hour != 23:
                del_rows.append(data_x["dt"][p])
                if p != 0:
                    p -= 1
                else:
                    break
        if data_x["dt"][n - 1].hour == 23 and data_x["dt"][n] != 0:
            p = n
            while data_x["dt"][p].hour != 0:
                del_rows.append(data_x["dt"][p])
                p += 1
        if (
            data_x["dt"][n].hour != data_x["dt"][n - 1].hour + 1
            and data_x["dt"][n - 1].hour != 23
        ):
            p = n - 1
            while data_x["dt"][p].hour != 23:
                del_rows.append(data_x["dt"][p])
                if p != 0:
                    p -= 1
                else:
                    break

    # TODO: 12 часовые обрезки есть.

    # исключаем строки из датафрейма
    data_x = data_x.loc[~data_x["dt"].isin(del_rows)]

    #new_filename_data = f"{filename_weather.replace('.csv', '')}_ref.csv"
    #data_x.to_csv(new_filename_data, index=False)

    return data_x


# фич инжинирим датафрейм, убирая всё лишнее и добавляя нужное
def improving_features(filename_weather, filename_energy, date_slice):

    # импорт и предобработка признаков
    data_x = refactor_date(filename_weather)
    # импорт целевых признаков
    data_y = pd.read_csv(filename_energy, sep=",", index_col=0)

    # преобразование даты и времени в тип данных дататайм64
    data_y.loc[:, "dt"] = pd.to_datetime(data_y["dt"])

    # удаление дупликатов по параметру дататайм, с оставлением последнего (второго) экземпляра
    data_x = data_x.drop_duplicates(subset=["dt"], keep="last")

    # создание доп матрицы признаков из дататайм
    # TODO: проверить, правильно ли день и час использовать без кодировки?
    # 24 ближе к 1 чем к 20, а 365 ближе к 1 чем к 360
    dayofyear = data_x["dt"].dt.dayofyear
    hours = data_x["dt"].dt.hour
    month = data_x["dt"].dt.month
    days_hours = pd.concat(
        [dayofyear, hours, month],
        axis=1,
        join="inner",
        keys=["dayofyear", "hours", "month"],
    )
    data_x = pd.concat([data_x, days_hours], axis=1)

        # слияние матрицы признаков и матрицы целевых переменных
    data = pd.merge(data_x, data_y, on="dt")

    # удаление явно ненужных столбцов из матрицы признаков
    data = data.drop(
        [
            "id",
            "gtpp",
            "load_time",
        ],
        axis=1,
    )

    # зимой станцию заметает снегом, поэтому убираем строки где выработка ниже 10
    data = data.drop(data[data["target"].values <= 10].index)
    # для формирования "летнего датасета"
    # data = data.drop(data[data["dayofyear"].values >= 270].index)
    # data = data.drop(data[data["dayofyear"].values <= 90].index)

    # показалось что месяц это избыточный признак TODO: ПРОВЕРИТЬ
    data = data.drop(["month"], axis=1)

    # очистка данных от лишних символов (TODO: не нужно?)
    data.replace("[^a-zA-Z0-9]", " ", regex=True)
    data = data.rename(columns=lambda x: re.sub("[^A-Za-z0-9_]+:[C]:", "", x))

    data.to_csv(filename_weather.replace("weather.csv", "fulldata.csv"), index=False)
    data_train = data[data["dt"] < date_slice]
    data_train.to_csv(
        filename_weather.replace("weather.csv", "fulldata_train.csv"), index=False
    )
    data_valid = data[data["dt"] > date_slice]
    data_valid.to_csv(
        filename_weather.replace("train/weather.csv", "valid/fulldata_valid.csv"),
        index=False,
    )


station_name = "station_1"
station_type = "solar"

improving_features(
    filename_weather=f"data/{station_type}/{station_name}/train/weather.csv",
    filename_energy=f"data/{station_type}/{station_name}/train/target.csv",
    date_slice="2022-07-18",
)
