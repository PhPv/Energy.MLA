import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import lightgbm as lgb
import datetime as dt
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from re import X
import sweetviz as sv
from clearml import Task, Logger
import shap
import pandas

from custom_metrics import MoneyMetric
from config import station_name, station_type, data_source, exp_name, tags

# ввести название станции, тип и имя модели и источник данных через '_'
# добавить МАЕ метрику 
# добавить расчет в О.Е.М.

def main_valide(station_type, station_name, data_source, exp_name=None):
    n = 1
    task = Task.init(
        project_name="Energy.ai",
        task_name=f"{exp_name}{data_source}_{dt.datetime.now()}",
        task_type="inference",
        tags = tags
    )
    log = Logger.current_logger()
    dirname = f"data/{station_type}/{station_name}/valid"
    filename_model = f"models/{station_type}/{station_name}/{exp_name}model_{data_source}_lgbm.sav"

    loaded_model = pickle.load(open(filename_model, "rb"))

    # импорт данных
    data = pd.read_csv(
        f"data/{station_type}/{station_name}/valid/fulldata_valid_{data_source}.csv",
        sep=",",
        index_col=0,
        # index_col=["dt"],
    )
    data = data.drop(data[data["target"].values <= 0.000001].index)
    try:
        pass
        #data = data.drop(["ratio"], axis=1)
        #data = data.drop(["GtiTracking"], axis=1)
        #data = data.drop(data[data["ratio"].values <= data['ratio'].quantile(0.20)].index)
        #data = data.drop(data[data["ratio"].values >= data['ratio'].quantile(0.98)].index)
        data = data.drop(["ratio"], axis=1)
    except:
        pass
    data.fillna(0, inplace=True)
    y_test = data[data.columns[-1]]
    X_test = data[data.columns[:-1]]

    #X_test = X_test.drop(columns=['Dni', 'GtiTracking', 'RelativeHumidity', 'hours', 'SnowWater', 'AlbedoDaily'], axis=1)

    # важность признаков
    lgb.plot_importance(loaded_model, figsize=(15, 8))
    plt.show()

    data_train = pd.read_csv(
        f"data/{station_type}/{station_name}/train/fulldata_train_{data_source}.csv",
        sep=",",
        index_col=0,
        # index_col=["dt"],
    )
    data_train = data_train.drop(["target"], axis=1)
    try:
        pass
        #data = data.drop(data[data["ratio"].values <= data['ratio'].quantile(0.01)].index)
        #data = data.drop(data[data["ratio"].values >= data['ratio'].quantile(0.99)].index)
        data_train = data_train.drop(["ratio"], axis=1)
    except:
        pass
    
    shap_test = shap.TreeExplainer(loaded_model).shap_values(data_train)
    #shap.summary_plot(shap_test, data_train, max_display=25, auto_size_plot=True)
    # прогноз выработки
    y_pred = loaded_model.predict(X_test)
    # скоринг
    result = loaded_model.score(X_test, y_test)

    print("score", result)
    # метрики
    print("rmse", mean_squared_error(y_test, y_pred, squared=False))
    print("mae", mean_absolute_error(y_test, y_pred))
    print("mape", mean_absolute_percentage_error(y_test, y_pred))
    print("wape", (y_test - y_pred).abs().sum() / y_test.sum())
    # отклонение прогноза в долях единицы
    difference = (y_pred - y_test) / y_test    

    # накопление отклонения (через создание списка)
    dif_sum = []
    for x in range(0, len(y_pred)):
        if x <= 60:
            dif_sum.append(np.sum(difference.iloc[0:x]))
        else:
            dif_sum.append(np.sum(difference.iloc[x - 60 : x]))

    dif_sum = pd.Series(dif_sum)

    # compare the results with sklearn package
    weighted_mean_sq_error_sklearn = np.average(
        (y_test - y_pred) / y_test, axis=0, weights=y_test
    )

    print("wmse", weighted_mean_sq_error_sklearn)

    df1 = pd.DataFrame(pd.Series(list(y_pred)), columns=["forecast"])
    # df2 = data["hours"].reset_index(drop=True)

    # для анализа экселек
    df2 = data[["target", "dayofyear", "hours"]].reset_index(drop=True)

    df_res = pd.concat([df1, df2], axis=1)
    # df_res.to_excel(f"data/{station_type}/{station_name}/predict/valid_predict.csv")

    # для анализа экселек
    df_res.to_excel(f"data/{station_type}/{station_name}/predict/{station_name}_valid_predict_{data_source}.xlsx")

    # аналитическая сводка
    df_res_group = df_res.groupby(by=["dayofyear"]).sum()
    df_res_group_cnt = df_res.groupby(by=["dayofyear"]).count()
    dfgu, dfgo = df_res_group, df_res_group
    dfgu = dfgu.drop(df_res_group[df_res_group["target"].values < df_res_group["forecast"].values].index)
    dfgo = dfgo.drop(df_res_group[df_res_group["target"].values > df_res_group["forecast"].values].index)
    undo_err = (dfgu['target'] - dfgu['forecast']).sum()/df_res_group['target'].sum()*100
    over_err = (dfgo['forecast'] - dfgo['target']).sum()/df_res_group['target'].sum()*100

    max_electr = df_res_group["target"].max()
    value_error = (df_res_group["forecast"] / df_res_group["target"]) - 1
    w_value_error = ((df_res_group["forecast"] / df_res_group["target"]) - 1) * df_res_group["target"] / max_electr
    df_value_error = pd.concat(
        [value_error, w_value_error, abs(value_error), abs(w_value_error)],
        axis=1,
        join="inner",
        keys=["value_error", "w_value_error", "abs_value_error", "abs_w_value_error"],
    )
    
    df_res = pd.concat([df_res_group, df_value_error], axis=1)
    # количество прогнозируемых часов в дне
    df_res['hours'] = df_res_group_cnt['hours']
    dmape = round(df_res['abs_value_error'].mean()*100,2)
    dwmape = round(df_res['abs_w_value_error'].mean()*100,2)
    #money = MoneyMetric()
    #print("Чистый доход в % от потенциально возможного составил ", money(loaded_model, X_test, pd.DataFrame(y_test, columns=["target"])))
    print(f"средняя ошибка на валидационном наборе составила {dmape}%")
    print(f"средневзвешенная ошибка на валидационном наборе составила {dwmape}%")
    df_res.to_excel(f"data/{station_type}/{station_name}/predict/{station_name}_{exp_name}_valid_predict_metrics_{data_source}.xlsx")

    # максимальное значение шкалы по оси OY
    OY_max = max(max(y_test), max(y_pred)) * 1.2

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.axis([0, len(y_test), 0, OY_max])
    ax2 = ax1.twinx()
    ax1.bar(np.arange(0, len(y_test)), y_test, label="Реальная выработка")
    ax1.bar(
        np.arange(0, len(y_test)),
        y_pred,
        alpha=0.7,
        label="Спрогнозированная выработка",
    )
    ax1.legend(loc="upper left", fontsize=10)
    ax1.set_ylabel("Выработка")
    ax1.text(
        150,
        -0.06 * OY_max,
        f"Ошибка (сутки):{dmape}%",
        ha="left",
        bbox={"facecolor": "white", "alpha": 1, "pad": 2},
    )
    ax1.text(
        150,
        -0.11 * OY_max,
        f"Прогноз:{round(np.sum(y_pred))} кВтч",
        ha="left",
        bbox={"facecolor": "white", "alpha": 1, "pad": 2},
    )
    ax1.text(
        150,
        -0.16 * OY_max,
        f"Реально:{round(np.sum(y_test))} кВтч",
        ha="left",
        bbox={"facecolor": "white", "alpha": 1, "pad": 2},
    )
    ax1.text(
        150,
        -0.21 * OY_max,
        f"За рассматриваемый период",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0, "pad": 2},
    )
    log.report_single_value("ERROR", value=over_err+undo_err)
    log.report_single_value("UNDO_ERR", value=undo_err)
    log.report_single_value("OVER_ERR", value=over_err)
    log.report_single_value("dmape", value=dmape)
    log.report_single_value("dwmape", value=dwmape)
    log.report_single_value("rmse", value=mean_squared_error(y_test, y_pred, squared=False))
    log.report_single_value("mae", value=mean_absolute_error(y_test, y_pred))
    log.report_single_value("mape", value=mean_absolute_percentage_error(y_test, y_pred))
    log.report_single_value("wape", value=(y_test - y_pred).abs().sum() / y_test.sum())
    log.report_single_value("wmse", value=np.average((y_test - y_pred) / y_test, axis=0, weights=y_test))
    fig.autofmt_xdate(rotation=45)

    #plt.show()
    task.close()

main_valide(station_type, station_name, data_source, exp_name)