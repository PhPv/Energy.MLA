import os

from common.utils import load_obj

DEBUG = True
API_PORT = 50001
# API_HOST = '0.0.0.0'
API_HOST = 'api'
WEB_PORT = 19999

CITIES = {
    '1': {
        'geo': {
            'lat': 123,
            'lon': 123,
        },
    },
}

POWER_PLANTS = {
    'object_1': {
        'name': '"Солнечная" СЭС',
        'city_id': '1',
        'models': {
            "LGBM": {
                'model': f'{os.environ["PYTHONPATH"]}/models/solar/station_1/model_v3.sav',
                'data': f'{os.environ["PYTHONPATH"]}/data/solar/station_1/data_v3.csv',
                'data_test_bound': 4062,
                # 'global_metrics': {
                #     'MAPE': ['11.66 %'],
                #     'MAE': [688],
                #     'RMSE': [1396],
                # },
            },
        },
    },
    'object_2': {
        'name': '"Ветряная" ВЭС',
        'city_id': '2',
        'models': {
            "LGBM": {
                'model': f'{os.environ["PYTHONPATH"]}/models/wind/model.sav',
                'data': f'{os.environ["PYTHONPATH"]}/data/wind/valid.csv',
                'data_test_bound': 0,
                # 'global_metrics': {
                #     'MAPE': ['11.66 %'],
                #     'MAE': [688],
                #     'RMSE': [1396],
                # },
            }
        },
    },
    'object_3': {
        'name': 'СЭС №1',
        'city_id': '3',
        'models': {
            "LGBM": {
                'model': f'{os.environ["PYTHONPATH"]}/models/solar/station_vinc/model.sav',
                'data': f'{os.environ["PYTHONPATH"]}/data/solar/station_vinc/valid/fulldata_valid.csv',
                'data_test_bound': 0,
                # 'global_metrics': {
                #     'MAPE': ['11.66 %'],
                #     'MAE': [688],
                #     'RMSE': [1396],
                # },
            }
        },
    },
}


