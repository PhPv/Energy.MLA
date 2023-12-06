# ToDo: сделать признаки день час, длительность светового дня
import pandas as pd

# путь до папки с исходным датасетом
dir_name = "data/solar/station_vinc/train/"

# грузим файлы
weather_df = pd.read_csv(f"{dir_name}weather_solcast60.csv")
target_df = pd.read_csv(f"{dir_name}fact.csv")  # , index_col=0)

# приводим дату к нужному формату
weather_df.loc[:, "PeriodEnd"] = pd.to_datetime(weather_df["PeriodEnd"])
weather_df.loc[:, "PeriodStart"] = pd.to_datetime(weather_df["PeriodStart"])


# формируем доп. признаки из даты
days_hours = pd.concat(
    [weather_df["PeriodEnd"].dt.dayofyear, weather_df["PeriodEnd"].dt.hour],
    axis=1,
    join="inner",
    keys=["dayofyear", "hours"],
)
weather_df = pd.concat([weather_df, days_hours], axis=1)


# убираем дату из датафрейма погоды
weather_df = weather_df.drop(["PeriodStart", "Period"], axis=1)
# переименовываем колонки датасета с выработкой под наш формат
target_df = target_df.rename(columns={"Date\Time": "dt", "Power": "target"})
target_df.loc[:, "dt"] = pd.to_datetime(target_df["dt"], utc=True)
weather_df = weather_df.rename(columns={"PeriodEnd": "dt"})
weather_df.loc[:, "dt"] = pd.to_datetime(weather_df["dt"], utc=True)


# собираем окончательный датасет
data_all = pd.merge(weather_df, target_df, on=["dt"])

# появились наны в выработке, возможно пропущеные значения в датасете. Заменим на 0, который дропнем на обучении
data_all.fillna(0, inplace=True)

data_all.to_csv(f"{dir_name}fulldata.csv", index=0)

# для формирования валидационной выборки узнаем год и месяц последней записи в общем датасете
last_year, last_month = (
    data_all["dt"].dt.year.iloc[-1],
    data_all["dt"].dt.month.iloc[-1],
)

data_all_train = data_all.drop(
    data_all[
        (data_all["dt"].dt.year.values == last_year)
        & (data_all["dt"].dt.month.values > last_month - 3)
    ].index
)
data_all_test = data_all.drop(
    data_all[
        (data_all["dt"].dt.year.values < last_year)
        | (data_all["dt"].dt.month.values <= last_month - 6)
    ].index
)

# сейвим датасеты
data_all_train.to_csv(f"{dir_name}fulldata_train_find.csv", index=0)
data_all_test.to_csv(
    f"{dir_name}fulldata_valid_find.csv".replace("train", "valid"), index=0
)
