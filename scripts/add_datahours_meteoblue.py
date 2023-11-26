import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import lightgbm as lgb
import datetime
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from re import X
import sweetviz as sv
from math import pi, cos

"""
Global Horizontal Irradiance*^ (GHI) (W/m2) - СУММА ПРЯМОЙ И РАССЕЯНОЙ
Direct Normal Irradiance*^ (DNI) (W/m2)
Diffuse Horizontal Irradiance*^ (DHI) (W/m2)
Global Tilted Irradiance*^ (GTI) (W/m2)

Dhi,Dni,Ebh,Ghi, GtiFixedTilt,GtiTracking,
"""
station_name = "station_sam"
station_type = "solar"
data_source = 'SOLCAST'
power = 25

dir_name = f"data/{station_type}/{station_name}/train/"
dir_name2 = f"data/{station_type}/{station_name}/valid/"

if data_source == 'NEMS':
    data_train_name = 'Meteoblue_NEMS_dataexport_20230215T051617_prep'
    data_valid_name = 'Meteoblue_NEMS'
    rad_feature_name = 'Vinkivtsi Shortwave Radiation'
elif data_source == 'ERA':
    data_train_name = 'Meteoblue_ERA_dataexport_20230215T052109_prep'
    data_valid_name = 'Meteoblue_ERA'
    rad_feature_name = 'Shortwave Radiation'
elif data_source == 'SOLCAST':
    data_train_name = 'fulldata_train_SOLCAST'
    data_valid_name = 'fulldata_valid_SOLCAST'
    rad_feature_name = 'Ghi'

def main(station_type, station_name):

    data = pd.read_csv(
        f"data/{station_type}/{station_name}/train/{data_train_name}.csv",
        sep=",",
    )
    try:
        data = data.drop(["ratio"], axis=1)
    except:
        pass
    data.loc[:, "dt"] = pd.to_datetime(data["dt"])

    # вычисляем "коэффициент эффективности"
    ratio = pd.DataFrame(data["target"]/power/1000/data[rad_feature_name], columns=['ratio'])
    ratio.replace([np.inf, -np.inf], np.nan, inplace=True)

    target_df = data["target"]
    # значение выработки дано для часового пояса +3
    # target_df,target_head = data["target"], data["target"].iloc[:3]
    # target_df.drop(target_df.index[:3], inplace=True)
    # target_df = target_df.append(target_head, ignore_index = True)

    data = data.drop(["target"], axis=1)

    data2 = pd.read_csv(
        f"data/{station_type}/{station_name}/valid/{data_valid_name}.csv",
        sep=",",
    )

    try:
        data2 = data2.drop(["ratio"], axis=1)
    except:
        pass

    target_df2 = data2["target"]

    # вычисляем "коэффициент эффективности"
    ratio2 = pd.DataFrame(data2["target"]/power/1000/data2[rad_feature_name], columns=['ratio'])
    ratio2.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    target_df2 = data2["target"]
    # значение выработки дано для часового пояса +3
    # target_df2,target_head2 = data2["target"], data2["target"].iloc[:3]
    # target_df2.drop(target_df2.index[:3], inplace=True)
    # target_df2 = target_df2.append(target_head2, ignore_index = True)

    data2 = data2.drop(["target"], axis=1)
    data2.loc[:, "dt"] = pd.to_datetime(data2["dt"])

    # формируем доп. признаки из даты
        
    # days_hours = pd.concat(
    #     [data["dt"].dt.dayofyear, data["dt"].dt.hour],
    #     axis=1,
    #     join="inner",
    #     keys=["dayofyear", "hours"],
    # )
    # days_hours2 = pd.concat(
    #     [data2["dt"].dt.dayofyear, data2["dt"].dt.hour],
    #     axis=1,
    #     join="inner",
    #     keys=["dayofyear", "hours"],
    # )
    
    """
    # через косинус
    days_hours = pd.concat(
        [
            np.cos(data["dt"].dt.dayofyear * 2 * pi / 365),
            np.cos(data["dt"].dt.hour * 2 * pi / 24),
        ],
        axis=1,
        join="inner",
        keys=["dayofyear", "hours"],
    )
    days_hours2 = pd.concat(
        [
            np.cos(data2["dt"].dt.dayofyear * 2 * pi / 365),
            np.cos(data2["dt"].dt.hour * 2 * pi / 24),
        ],
        axis=1,
        join="inner",
        keys=["dayofyear", "hours"],
    )
    """
    fulldata = pd.concat([data, 
                        #days_hours, 
                        ratio, target_df], axis=1)
    fulldata2 = pd.concat([data2, 
                        #days_hours2, 
                        ratio2, target_df2], axis=1)

    fulldata.to_csv(f"{dir_name}fulldata_train_{data_source}.csv", index=0)
    fulldata2.to_csv(f"{dir_name2}fulldata_valid_{data_source}.csv", index=0)


main(station_type, station_name)
