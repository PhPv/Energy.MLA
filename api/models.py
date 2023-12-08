from sklearn.metrics import *
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import pandas as pd
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import optuna
import pickle


def train_lgbm(data: pd.DataFrame) -> LGBMRegressor:
    '''
    Обучение LightGBM модели

    Args:
        data: Обучающая выборка

    Returns:
        Обученная LightGBM модель
    '''
    y_train = data[data.columns[1]]
    X_train = data[data.columns[2:]]
    X_train, _, y_train, _ = train_test_split(X_train, y_train, test_size=0.01, random_state=42)

    lgbm = LGBMRegressor(metric = 'rmse')

    # Сетка для поиска оптимальных гиперпараметров
    param_search = {'num_leaves': [10, 20, 31, 40, 50, 100], 
                    'max_depth' : [-1, 3, 5], 
                    'min_data_in_leaf' : [10, 20, 30], 
                    'lambda_l1': [0, 0.1, 0.2], 
                    'lambda_l2': [0, 0.1, 0.2]}

    tscv = TimeSeriesSplit()

    gsearch = GridSearchCV(estimator=lgbm, cv=tscv, param_grid=param_search)

    gsearch.fit(X_train, y_train)

    return gsearch.best_estimator_

def train_cb(data: pd.DataFrame) -> CatBoostRegressor:
    '''
    Обучение Catboost модели

    Args:
        data: Обучающая выборка

    Returns:
        Обученная Catboost модель
    '''
    y_train = data[data.columns[1]]
    X_train = data[data.columns[2:]]

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.01, random_state=42)

    # Оптимизация гиперпараметров при помощи Optuna
    def objective(trial):
        param = {
            'n_estimators': trial.suggest_categorical('n_estimators', [750, 1000, 1250, 1500, 1750, 2000]),
            'learning_rate': trial.suggest_categorical('learning_rate', [0.005, 0.01, 0.015, 0.02, 0.025, 0.03]),
            'subsample': trial.suggest_categorical('subsample', [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
            'min_child_samples': trial.suggest_int('min_child_samples', 1, 300),
            'max_depth': trial.suggest_categorical('max_depth', [4, 6, 8]),
        }
        model = CatBoostRegressor(**param)  
        
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=100, verbose=False)
        
        score = model.score(X_test, y_test)
        
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    params = study.best_trial.params
    cb = CatBoostRegressor(**params)

    cb.fit(X_train, y_train)

    return cb

def get_lgbm_forecast(model_id: str, time_interval: tuple[str, str], coordinates: tuple[float,float]) -> pd.DataFrame:
    '''
    Получение предсказаний выработки на основе LightGBM модели

    Args:
        model_id:       ID обученной модели
        time_interval:  Диапазон дат для получения предсказаний
        coordinates:    Кортеж с координатами (широта, долгота)

    Returns:
        Датафрейм с предсказаниями LightGBM модели
    '''
    filename = '..\\models\\model_lgbm_{0}.sav'.format(model_id)
    with (open(filename, 'rb') as model_file):
        lgbm: LGBMRegressor = pickle.load(model_file)

    weather_df = get_weather(*time_interval, coordinates)

    preds = lgbm.predict(weather_df)

    return pd.DataFrame([weather_df['dt'], preds])

def get_cb_forecast(model_id: str, time_interval: tuple[str, str], coordinates: tuple[float,float]) -> pd.DataFrame:
    '''
    Получение предсказаний выработки на основе Catboost модели

    Args:
        model_id:       ID обученной модели
        time_interval:  Диапазон дат для получения предсказаний
        coordinates:    Кортеж с координатами (широта, долгота)

    Returns:
        Датафрейм с предсказаниями Catboost модели 
    '''
    filename = '..\\models\\model_cb_{0}.sav'.format(model_id)
    with (open(filename, 'rb') as model_file):
        cb: CatBoostRegressor = pickle.load(model_file)

    weather_df = get_weather(*time_interval, coordinates)

    preds = cb.predict(weather_df)

    return pd.DataFrame([weather_df['dt'], preds])

def get_lgbm_forecast_tmp(model_id: str, time_interval: tuple[str, str]) -> pd.DataFrame:
    '''
    Получение предсказаний выработки на основе LightGBM модели (временная заглушка)

    Args:
        model_id:       ID обученной модели
        time_interval:  Диапазон дат для получения предсказаний
        coordinates:    Кортеж с координатами (широта, долгота)

    Returns:
        Датафрейм с предсказаниями LightGBM модели

    TODO:
        Удалить, когда будет готово погодное API
    '''
    filename = '..\\models\\model_lgbm_{0}.sav'.format(model_id)
    with (open(filename, 'rb') as model_file):
        lgbm: LGBMRegressor = pickle.load(model_file)

    weather_df = pd.read_csv('../data_weather/station_1/Station_1_weather_clear_06_06.csv', sep=";")

    weather_df.loc[:, 'dt'] = pd.to_datetime(weather_df['dt'])
    start_date = pd.Timestamp(time_interval[0])
    end_date = pd.Timestamp(time_interval[1])

    weather_df = weather_df[(weather_df['dt'] >= start_date) & (weather_df['dt'] <= end_date)]

    preds = lgbm.predict(weather_df)

    return pd.DataFrame([weather_df['dt'], preds])

def get_cb_forecast_tmp(model_id: str, time_interval: tuple[str, str]) -> pd.DataFrame:
    '''
    Получение предсказаний выработки на основе Catbooost модели (временная заглушка)

    Args:
        model_id:       ID обученной модели
        time_interval:  Диапазон дат для получения предсказаний
        coordinates:    Кортеж с координатами (широта, долгота)

    Returns:
        Датафрейм с предсказаниями Catbooost модели

    TODO:
        Удалить, когда будет готово погодное API
    '''
    filename = '..\\models\\model_cb_{0}.sav'.format(model_id)
    with (open(filename, 'rb') as model_file):
        cb: CatBoostRegressor = pickle.load(model_file)

    weather_df = pd.read_csv('../data_weather/station_1/Station_1_weather_clear_06_06.csv', sep=";")

    weather_df.loc[:, 'dt'] = pd.to_datetime(weather_df['dt'])
    start_date = pd.Timestamp(time_interval[0])
    end_date = pd.Timestamp(time_interval[1])

    weather_df = weather_df[(weather_df['dt'] >= start_date) & (weather_df['dt'] <= end_date)]

    preds = cb.predict(weather_df)

    return pd.DataFrame([weather_df['dt'], preds])

def get_lgbm_metrics(model_id: str, time_interval: tuple[str, str], X_test: pd.DataFrame, y_test: pd.DataFrame) -> pd.DataFrame:
    '''
    Получение метрик LightGBM модели

    Args:
        model_id:       ID обученной модели
        time_interval:  Диапазон дат для получения предсказаний
        X_test:         Валидационный датафрейм с признаками
        y_test:         Валидационный датафрейм с целевой переменной

    Returns:
        Датафрейм с метриками LightGBM модели

    TODO:
        Убрать тестовые датафреймы, когда будет готово погодное API, и подтягивать данные напрямую
    '''
    filename = '..\\models\\model_lgbm_{0}.sav'.format(model_id)
    with (open(filename, 'rb') as model_file):
        lgbm: LGBMRegressor = pickle.load(model_file)

    weather_df = pd.read_csv('../data_weather/station_1/Station_1_weather_clear_06_06.csv', sep=";")

    weather_df.loc[:, 'dt'] = pd.to_datetime(weather_df['dt'])
    start_date = pd.Timestamp(time_interval[0])
    end_date = pd.Timestamp(time_interval[1])

    weather_df = weather_df[(weather_df['dt'] >= start_date) & (weather_df['dt'] <= end_date)]

    preds = lgbm.predict(weather_df)

    metrics = {
                "Model": ["LightGBM"],
                "Score": [lgbm.score(X_test, preds)],
                "MSE": [mean_squared_error(y_test, preds, squared=False)],
                "MAPE": [mean_absolute_percentage_error(y_test, preds)],
            }
    return pd.DataFrame(metrics)

def get_cb_metrics(model_id: str, time_interval: tuple[str, str], X_test: pd.DataFrame, y_test: pd.DataFrame) -> pd.DataFrame:
    '''
    Получение метрик Catboost модели

    Args:
        model_id:       ID обученной модели
        time_interval:  Диапазон дат для получения предсказаний
        X_test:         Валидационный датафрейм с признаками
        y_test:         Валидационный датафрейм с целевой переменной

    Returns:
        Датафрейм с метриками Catboost модели

    TODO:
        Убрать тестовые датафреймы, когда будет готово погодное API, и подтягивать данные напрямую
    '''
    filename = '..\\models\\model_cb_{0}.sav'.format(model_id)
    with (open(filename, 'rb') as model_file):
        cb: CatBoostRegressor = pickle.load(model_file)

    weather_df = pd.read_csv('../data_weather/station_1/Station_1_weather_clear_06_06.csv', sep=";")

    weather_df.loc[:, 'dt'] = pd.to_datetime(weather_df['dt'])
    start_date = pd.Timestamp(time_interval[0])
    end_date = pd.Timestamp(time_interval[1])

    weather_df = weather_df[(weather_df['dt'] >= start_date) & (weather_df['dt'] <= end_date)]

    preds = cb.predict(weather_df)

    metrics = {
                "Model": ["CatBoost"],
                "Score": [cb.score(X_test, preds)],
                "MSE": [mean_squared_error(y_test, preds, squared=False)],
                "MAPE": [mean_absolute_percentage_error(y_test, preds)],
            }

    return pd.DataFrame(metrics)


