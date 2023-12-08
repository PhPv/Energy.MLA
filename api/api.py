from datetime import datetime, date, timedelta
from functools import lru_cache

import numpy as np
import requests
from fastapi import FastAPI, UploadFile
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error

from common import logger
from fastapi_models import MetricsRequest
from common.utils import load_obj, fmt_percent
from models import train_cb, train_lgbm, train_test_split
import pickle
from uuid import uuid4, UUID
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse
from config.settings import POWER_PLANTS, DEBUG

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache
def load_pickle_obj(path):
    return load_obj(path)


def smape(a, f):
    return 1/len(a) * np.sum(2 * np.abs(f-a) / (np.abs(a) + np.abs(f))*100)


@app.get('/models')
async def get_models(object_id: str):
    object_dct = POWER_PLANTS[object_id]
    available_models = [md for md in object_dct["models"].keys()]
    return available_models


@app.get('/date_range')
async def get_models(object_id: str, model_name: str):
    model_dct = POWER_PLANTS[object_id]["models"][model_name]
    df = pd.read_csv(model_dct["data"], index_col=0)
    df = df[model_dct["data_test_bound"]:]
    min_val = df.index.min()
    max_val = df.index.max()

    return (min_val, max_val)


def get_forecast_data(lat, lon):
    api_key = "****"
    endpoint = 'https://my.meteoblue.com/packages/basic-1h_basic-day'
    resp = requests.get(url=f"{endpoint}?apikey={api_key}&lat={lat}&lon={lon}&asl=19&format=json")
    df = pd.DataFrame(resp)
    return df.to_records()


@app.post('/preds')
async def get_preds(data: MetricsRequest):
    object_id = data.object_id
    date_range = data.date_range if data.date_range != () else None
    model_name = data.model_name

    object_dct = POWER_PLANTS[object_id]
    model_dct = object_dct["models"][model_name]
    model = load_pickle_obj(model_dct["model"])
    data = pd.read_csv(model_dct["data"], index_col=0)
    data = data[model_dct["data_test_bound"]:]

    # forecast data block
    parsed_date_start = datetime.strptime(date_range[0], "%Y-%m-%d")
    parsed_date_end = datetime.strptime(date_range[1], "%Y-%m-%d")
    date_delta_days = (parsed_date_end - parsed_date_start).days
    if parsed_date_start.date() >= datetime.now().date():
        data = data[:date_delta_days*24]
        # data = get_forecast_data(lat, lon)

    elif date_range is not None:
        data = data.loc[date_range[0]: date_range[1]]

    times = data.index.tolist()
    data = data.values
    x_test, y_test = data[:, :-1], data[:, -1]

    preds = model.predict(x_test).tolist()
    y_test = y_test.tolist()

    df_dct = {
        "times": times,
        "preds": preds,
        "target": y_test,
    }
    # df = pd.DataFrame(df_dct)

    data = pd.DataFrame(df_dct)
    date_column = data.columns[0]
    data[date_column] = data[date_column].astype(np.datetime64)
    data = data[(data[date_column].dt.hour < 22) & (data[date_column].dt.hour > 5)]

    return data.to_dict(orient="records")


@app.post('/metrics')
async def get_metrics(data: MetricsRequest):
    object_id = data.object_id
    date_range = data.date_range if data.date_range != () else None
    model_name = data.model_name

    object_dct = POWER_PLANTS[object_id]
    model_dct = object_dct["models"][model_name]
    model = load_pickle_obj(model_dct["model"])
    data = pd.read_csv(model_dct["data"], index_col=0)
    data = data[model_dct["data_test_bound"]:]

    if date_range is not None:
        data = data.loc[date_range[0]: date_range[1]]

    timestamp = data.index
    data_np = data.values
    x_test, y_test = data_np[:, :-1], data_np[:, -1]

    preds = model.predict(x_test)
    data['preds'] = preds

    metrics = pd.DataFrame.from_dict(
        {
            "Model": [model_name],
            "MAE": [round(mean_absolute_error(y_test, preds))],
            "RMSE": [round(mean_squared_error(y_test, preds, squared=False))],
            "MAPE": [f'{round(smape(y_test, preds), 2)} %'],
        }
    ).to_dict(orient="records")

    # Business metrics
    data = data.reset_index()
    date_column = data.columns[0]
    data[date_column] = data[date_column].astype(np.datetime64)
    data = data[(data[date_column].dt.hour < 22) & (data[date_column].dt.hour > 5)]

    # logger.info(f'data {data.dtypes}')
    accuracy_agg_day = data.groupby([data[data.columns[0]].dt.day]).agg({
        data.columns[-2]: "sum",  # target column
        "preds": "sum"
    })#.values

    accuracy_agg_day.rename(mapper={'AP': 'target'}, axis=1, inplace=True)
    accuracy_agg_day['day_accuracy'] = (accuracy_agg_day['preds'] - accuracy_agg_day['target']).apply(abs)
    accuracy_agg_day['day_accuracy'] = accuracy_agg_day['day_accuracy'] / accuracy_agg_day['target']
    accuracy_agg_day['day_accuracy'] = accuracy_agg_day['day_accuracy'].apply(lambda x: 0.5 if x >= 1 else x)
    accuracy_agg_day['day_accuracy'] = (100 - accuracy_agg_day['day_accuracy'] * 100).apply(lambda x: round(abs(x), 2))
    accuracy_agg_day = accuracy_agg_day.values
    logger.info(accuracy_agg_day)
    # round(100 - (abs(preds - fact) / fact) * 100, 2)

    fact = np.round(np.mean(accuracy_agg_day[:, 0]))
    preds = np.round(np.mean(accuracy_agg_day[:, 1]))
    accuracy = np.round(np.mean(accuracy_agg_day[:, 2]))

    # logger.info(f'accuracy_agg_day {accuracy_agg_day[:5]}')

    business_metrics = pd.DataFrame.from_dict(
        {
            "Выработка факт": [f'{fact} кВт*ч'],
            "Выработка предсказано": [f'{preds} кВт*ч'],
            "Точность": [f'{accuracy} %']
        }
    ).to_dict(orient="records")

    return metrics, business_metrics




# -------------- OLD ----------------------


@app.post('/api/train')
async def train_models(y_file: UploadFile) -> UUID:
    """
    Обучение моделей на сервере

    Args:
        y_file: Данные по выработке

    Returns:
        ID обученных моделей

    TODO:
        Обернуть препроцессинг в отдельный универсальный пайплайн. 
        Убрать захардкоженную выборку с признаками после добавления погодного API
    """

    # Препроцессинг данных
    data_x = pd.read_csv('../data_weather/station_1/Station_1_weather_clear_06_06.csv', sep=";")
    data_y = pd.read_csv(y_file.file, index_col=0)

    data_x = data_x.drop_duplicates(subset=['dt'], keep='last')

    data_x.loc[:, 'dt'] = pd.to_datetime(data_x['dt'])
    data_y.loc[:, 'dt'] = pd.to_datetime(data_y['dt'])

    data = pd.merge(data_y, data_x, on='dt')

    dayofyear = data['dt'].dt.dayofyear
    hours = data['dt'].dt.hour
    days_hours = pd.concat([dayofyear, hours], axis=1, join='inner', keys=['dayofyear', 'hours'])

    data = data.drop(['id', 'gtpp', 'load_time', 'predict', 'dt', 'Visibility', 'Surface_pressure'], axis=1)
    data = pd.concat([data, days_hours], axis=1)
    data = data.drop(data[data['fact'].values <= 10].index)

    # Обучение моделей
    lgbm = train_lgbm(data)
    cb = train_cb(data)

    
    # Сохранение моделей на сервере
    model_id = uuid4()
    lgbm_filename = '..\\models\\model_lgbm_{0}.sav'.format(model_id)
    cb_filename = '..\\models\\model_cb_{0}.sav'.format(model_id)

    with (open(lgbm_filename, 'wb') as file):
        pickle.dump(lgbm, file)

    with (open(cb_filename, 'wb') as file):
        pickle.dump(cb, file)

    return model_id


@app.get('/api/get_model')
async def get_model_by_id(model_id: str, model_type: str) -> StreamingResponse:
    """
    Получение модели по ID

    Args:
        model_id:   ID модели
        model_type: Тип модели

    Returns:
        Модель в виде обернутого байтового потока
    """
    filename = '../models/model_{0}_{1}.sav'.format(model_type, model_id)

    with (open(filename, 'rb') as model_file):
        model = pickle.load(model_file)

    return StreamingResponse(to_bytes_stream(model))

@app.get('/api/forecast')
async def get_forecast(model_type: str, model_id: str, time_interval: tuple[str, str], loc_name: str) -> StreamingResponse:
    """
    Получение предсказаний модели

    Args:
        model_type:     Тип модели
        model_id:       ID модели
        time_interval:  Диапазон дат для получения предсказаний
        loc_name:       Географическое название

    Returns:
        Датафрейм с предсказаниями
    """
    def get_coordinates(loc_name: str) -> tuple[float, float]:
        """
        Получение координат по географическому названию места

        Args:
            loc_name: Географическое название

        Returns:
            Кортеж с координатами

        TODO: 
            Доделать
        """
        pass

    if model_type == 'lgbm':
        return StreamingResponse(to_bytes_stream(get_lgbm_forecast(model_id, time_interval, get_coordinates(loc_name))))
    elif model_type == 'cb':
        return StreamingResponse(to_bytes_stream(get_cb_forecast(model_id, time_interval, get_coordinates(loc_name))))
    else:
        raise ValueError


@app.get('/api/forecast_tmp')
async def get_tmp_forecast(model_type: str, model_id: str, time_interval: tuple[str, str]) -> StreamingResponse:
    """
    Получение предсказаний выработки (временная заглушка)

    Args:
        model_type:     Тип модели
        model_id:       ID модели
        time_interval:  Диапазон дат для получения предсказаний

    Returns:
        Датафрейм с предсказаниями выработки

    TODO: 
        Удалить, когда будет готово погодное API
    """
    if model_type == 'lgbm':
        return StreamingResponse(to_bytes_stream(get_lgbm_forecast_tmp(model_id, time_interval)))
    elif model_type == 'cb':
        return StreamingResponse(to_bytes_stream(get_cb_forecast_tmp(model_id, time_interval)))
    else:
        raise ValueError


@app.get('/api/metrics')
async def get_metrics(model_type: str, model_id: str, time_interval: tuple[str, str]) -> StreamingResponse:
    """
    Получение метрик модели (временная заглушка)

    Args:
        model_type:     Тип модели
        model_id:       ID модели
        time_interval:  Диапазон дат для получения предсказаний

    Returns:
        Датафрейм с метриками модели

    TODO: 
        Удалить предобработку данных, когда будет готово погодное API
    """
    # Предобработка данных
    data_x = pd.read_csv('../data_weather/station_1/Station_1_weather_clear_06_06.csv', sep=";")
    data_y = pd.read_csv('../data_weather/station_1/fact_Station_1_2022-08-05.csv', sep=',', index_col=0)

    data_x = data_x.drop_duplicates(subset=['dt'], keep = 'last')

    data_x.loc[:, 'dt'] = pd.to_datetime(data_x['dt'])
    data_y.loc[:, 'dt'] = pd.to_datetime(data_y['dt'])

    data = pd.merge(data_y, data_x, on='dt')

    dayofyear = data['dt'].dt.dayofyear
    hours = data['dt'].dt.hour
    days_hours = pd.concat([dayofyear, hours], axis=1, join='inner', keys=['dayofyear', 'hours'])

    data = data.drop(['id', 'gtpp', 'load_time', 'predict', 'dt', 'Visibility', 'Surface_pressure'], axis=1)
    data = pd.concat([data, days_hours], axis=1)
    data = data.drop(data[data['fact'].values <= 10].index)

    y_train = data[data.columns[1]]
    X_train = data[data.columns[2:]]

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.01, random_state=42)

    if model_type == 'lgbm':
        return StreamingResponse(to_bytes_stream(get_lgbm_metrics(model_id, time_interval, X_test, y_test)))
    elif model_type == 'cb':
        return StreamingResponse(to_bytes_stream(get_cb_metrics(model_id, time_interval, X_test, y_test)))
    else:
        raise ValueError


def get_weather(start_date: str, end_date: str, coordinates: tuple[float,float]) -> pd.DataFrame:
    '''
    Получение данных о погоде для заданных координат в заданном диапазоне дат

    Args:
        start_date:     Начало диапазона дат
        end_date:       Конец диапазона дат
        coordinates:    Кортеж с координатами (широта, долгота)

    Returns:
        Датафрейм с погодой

    TODO:
        Прописать получение данных о погоде (прогноза и исторических)
        для заданных координат и в заданном диапазоне дат
    '''
    pass

def get_historical_weather(start_date: str, end_date: str, coordinates: tuple[float,float]) -> pd.DataFrame:
    '''
    Получение исторических данных о погоде для заданных координат в заданном диапазоне дат

    Args:
        start_date:     Начало диапазона дат
        end_date:       Конец диапазона дат
        coordinates:    Кортеж с координатами (широта, долгота)

    Returns:
        Датафрейм с историческими данными о погоде

    TODO:
        Прописать получение исторических данных о погоде
        для заданных координат и в заданном диапазоне дат
    '''
    pass

def get_weather_forecast(start_date: str, end_date: str, coordinates: tuple[float,float]) -> pd.DataFrame:
    '''
    Получение прогнозных данных о погоде для заданных координат в заданном диапазоне дат

    Args:
        start_date:     Начало диапазона дат
        end_date:       Конец диапазона дат
        coordinates:    Кортеж с координатами (широта, долгота)

    Returns:
        Датафрейм с прогнозными данными о погоде

    TODO:
        Прописать получение прогнозных данных о погоде
        для заданных координат и в заданном диапазоне дат
    '''
    pass

