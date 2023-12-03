import pandas as pd
import lightgbm as lgb
import pickle
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score as cvs
from sklearn.metrics import mean_squared_error
from clearml import Task, Logger

from custom_metrics import dmapeMetric, dwmapeMetric
from config import station_name, station_type, data_source, exp_name, tags


"""Основная модель для солнечной и ветряной станции на GridSearchCV, LightGBM, tcsv
1.  Датасет разбивается последовательно, поэтому для датасета с 01.01 по 31.12 тест выборка это зима (плохо)
    Для классических метрик можно зашафлить
2.  Реализованы 2 Кастомные метрики, считающие суточные значения на основании часовых прогнозов. 
    dmape и dwmape (с учетом коэффициента Выработка текущего дня / на выработку лучшего дня)
    Их значение не очень валидно т.к. тестовый набор берёт первые дни датасета (лето от зимы сильно отличается)
    Сама модель учиться нормально т.к. используется кроссвалидация 
    (tscv вроде как разбивает на несколько групп, поэтому группировка по суткам для кастомной метрики возможна)
3.  3 Датасета для Винк:
        NEMS - Meteoblue, 42 признака - топчик
        ERA - meteoblue, 64 признака - похуже
        Solcast - Solcast, 19 признаков
    2 Датасета для Wind Yalova
        kaggle - без погоды
        SOLCAST
    2 Датасета для станции Б
        G
        SOLCAST
    2 Датасета для станции i
        G
        SOLCAST 
4.  Благодаря расчету ratio можно отсеить данные и разделить всё на "высокую" и "низкую" выработку
    И учить модели отдельно для них
5.  Пути оптимизации:
        Две-4 модели на все сезоны
        Выделять по умному переход с сезона на сезон 
            (есть плавная составляющая а етсь резкая, напимер снег в марте)
        коэф раб оборудования позволит отстроится от заснеженности
"""

def main(data_source=None, exp_name=None):

    task = Task.init(
        project_name="Energy.ai",
        task_name=f"{exp_name}{data_source}_{dt.datetime.now()}",
        task_type="training",
        tags = tags
    )

    # читаем датасет
    X_train, X_test, y_train, y_test = read_clear_data()

    # учим модель
    model = train_model(X_train, X_test, y_train, y_test)

    # считаем прогноз и метрики
    calc_metrics(model.best_estimator_, X_test, y_test)
    clear_ml_metrics(model.best_estimator_, X_test, y_test)

    task.close()

    return



def read_clear_data():
    '''Загрузка, предобработка датасета'''
    data = pd.read_csv(
        f"data/{station_type}/{station_name}/train/fulldata_train_{data_source}.csv",
        sep=";",
        index_col=0,
        # index_col=["dt"],
    )
    data = data.drop(data[data["target"].values <= 0.00001].index)
    try:
        #data = data.drop(["ratio"], axis=1)
        #data = data.drop(["GtiTracking"], axis=1)
        data = data.drop(data[data["ratio"].values <= data['ratio'].quantile(0.20)].index)
        data = data.drop(data[data["ratio"].values >= data['ratio'].quantile(0.98)].index)
        data = data.drop(["ratio"], axis=1)
    except:
        pass
    #data = data.drop(data[data["ratio"].values >= 5].index)
    # появились наны в выработке, возможно пропущеные значения в датасете. Заменим на 0, который дропнем на обучении
    data.fillna(0, inplace=True)
    # выделение матриц признаков и целевых переменных (целевая переменная 'target')
    y_train = data[data.columns[-1]]
    X_train = data[data.columns[:-1]]

    # формирование набора для обучения и тестирования Сравнить shuffle=True и False
    X_train, X_test, y_train, y_test = train_test_split(
        X_train, y_train, test_size=0.05, random_state=42, shuffle=False
    )

    return X_train, X_test, y_train, y_test

def train_model(X_train, X_test, y_train, y_test):
    '''Обучение модели'''
    # описание параметров модели TODO: попробовать другие метрики, другие гипепараметры
    # TODO: прикрутить optuna

    my_lgb = lgb.LGBMRegressor()
    param_search = {
        "num_leaves":   [10, 25, 50],
        "max_depth": [-1, 3, 5], 
        "min_data_in_leaf": [10, 20, 30],
        "lambda_l1": [0, 0.1, 0.2],
        "lambda_l2": [0, 0.1, 0.2],
    }

    scoring = dwmapeMetric()

    # формирование набора с учетом таймсерий (TODO: проработать)
    tscv = TimeSeriesSplit()

    # конфигурирование модели
    gsearch = GridSearchCV(estimator=my_lgb, cv=tscv, param_grid=param_search, scoring=scoring) 

    # обучение модели
    gsearch.fit(X_train, y_train)
    model = gsearch


    # сохранение модели с лучшими параметрами
    pickle.dump(
        model.best_estimator_,
        open(f"models/{station_type}/{station_name}/{exp_name}model_{data_source}.sav","wb",),)

    # оценка модели с лучшими параметрами mse метрикой TODO: проработать
    cv_scores = cvs(model.best_estimator_, X_train, y_train, cv=tscv, scoring=scoring) 
    print("cv_scores", cv_scores)
    
    # прогноз выработки
    # TODO: что за скор? явно не тот, что указан в cv_scrores
    result = model.best_estimator_.score(X_test, y_test)
    print("score", result)

    return model


    

def calc_metrics(model, X_test, y_test):
    '''Расчет метрик и вывод в консоль'''
    y_pred = model.predict(X_test)

    df1 = pd.DataFrame(pd.Series(list(y_pred)), columns=["forecast"])
    df2 = y_test.reset_index(drop=True)
    df3 = X_test[["dayofyear", "hours"]].reset_index(drop=True)
    df_res = pd.concat([df1, df2, df3], axis=1)

    # группируем чтобы получить значения за сутки
    df_res_group = df_res.groupby(by=["dayofyear"]).sum()

    # считаем dmape с учетом знака для каждых суток
    value_error = (df_res_group["forecast"] / df_res_group["target"]) - 1

    # для средневзвешенного ищем максимальное значение выработки 
    max_electr = df_res_group["target"].max()

    # считаем dwmape как dmape * долю дня от максимального значения для каждых суток
    w_value_error = ((df_res_group["forecast"] / df_res_group["target"]) - 1) * df_res_group["target"] / max_electr

    # собираем в один массив для экселя
    df_value_error = pd.concat(
        [value_error, w_value_error, abs(value_error), abs(w_value_error)],
        axis=1,
        join="inner",
        keys=["value_error", "w_value_error", "abs_value_error", "abs_w_value_error"],
    )
    df_res = pd.concat([df_res_group, df_value_error], axis=1)

    # считаем среднее значение dmape и dwmape 
    dmape = round(df_res['abs_value_error'].mean()*100,2)
    dwmape = round(df_res['abs_w_value_error'].mean()*100,2)
    
    df_res.to_excel(
        f"data/{station_type}/{station_name}/predict/{station_name}_train_predict_metrics_{data_source}.xlsx"
    )

    # метрики на тестовом датасете
    print(f"средняя суточная ошибка на тестовом наборе составила {dmape}%")
    print(f"средневзвешенная суточная ошибка на тестовом наборе составила {dwmape}%")
    print()
    dmape, dwmape = dmapeMetric(), dwmapeMetric()
    print("dmape", dmape(model, X_test, pd.DataFrame(y_test, columns=["target"])))
    print("dwmape", dwmape(model, X_test, pd.DataFrame(y_test, columns=["target"])))
    print("rmse", mean_squared_error(y_test, y_pred, squared=False))
    print("mae", mean_absolute_error(y_test, y_pred))
    print("mape", mean_absolute_percentage_error(y_test, y_pred))
    return

def clear_ml_metrics(model, X_test, y_test):
    '''Расчет метрик и вывод в ClearML'''
    log = Logger.current_logger()

    y_pred = model.predict(X_test)
    dmape, dwmape = dmapeMetric(), dwmapeMetric()
    log.report_single_value("dmape", value=dmape(model, X_test, pd.DataFrame(y_test, columns=["target"])))
    log.report_single_value("dwmape", value=dwmape(model, X_test, pd.DataFrame(y_test, columns=["target"])))
    log.report_single_value("rmse", value=mean_squared_error(y_test, y_pred, squared=False))
    log.report_single_value("mae", value=mean_absolute_error(y_test, y_pred))
    log.report_single_value("mape", value=mean_absolute_percentage_error(y_test, y_pred))
    log.report_single_value("wape", value=(y_test - y_pred).abs().sum() / y_test.sum())
    log.report_single_value("wmse", value=np.average((y_test - y_pred) / y_test, axis=0, weights=y_test))
    log.report_single_value("rmse", value=mean_squared_error(y_test, y_pred, squared=False))

def print_plots(data, y_pred, y_test):
    '''Графическое представление результатов'''
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
    OY_max = max(max(y_test), max(y_pred)) * 1.2

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.axis([0, len(y_test), 0, OY_max])
    ax2 = ax1.twinx()
    ax1.bar(np.arange(0, len(y_test)), y_test, label="Real Power")
    ax1.bar(np.arange(0, len(y_test)), y_pred, alpha=0.7, label="Predicted Power")
    ax2.plot(
        np.arange(0, len(y_test)), difference, label="dif", alpha=0.99, color="red"
    )
    ax1.legend(loc="upper left", fontsize=10)
    ax1.set_title("Difference")
    ax1.set_ylabel("power_value")
    ax1.text(
        18,
        0.01 * OY_max,
        f"Точность:{round((1 - abs(sum(y_test) - sum(y_pred))/sum(y_test))*100, 1)}%",
        ha="left",
    )
    ax1.text(18, 0.06 * OY_max, f"Прогноз:{round(np.sum(y_pred))} кВтч", ha="left")
    ax1.text(18, 0.11 * OY_max, f"Реально:{round(np.sum(y_test))} кВтч", ha="left")
    ax1.text(18, 0.16 * OY_max, f"За сутки:", ha="left")
    fig.autofmt_xdate(rotation=45)
    return

# rnd
main(data_source, exp_name)


    
