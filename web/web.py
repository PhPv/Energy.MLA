import re
import pandas as pd
import requests
import streamlit as st
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from streamlit.components.v1 import html
from datetime import timedelta, date, datetime
from common.utils import from_bytes_stream, to_excel, post_request, get_request
import plotly.graph_objects as go
from config.settings import POWER_PLANTS, DEBUG, API_PORT, API_HOST
from common import logger

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden; }
        footer {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)
models = [{'model': None, 'model_type': mt[0], 'model_name': mt[1], 'metrics': pd.DataFrame(), 'preds': None}
          for mt in [
              ('lgbm', 'Предсказанная мощность'),
              # ('cb', 'Catboost'),
              # ('xgb', 'XGBoost')
          ]]
API_URL = f'http://{API_HOST}:{API_PORT}'


@st.cache_data
def get_preds(chosen_power_plant, model_name, date_range):
    graph_data = post_request(API_URL,
                              endpoint='preds',
                              data={
                                  "object_id": chosen_power_plant,
                                  "model_name": model_name,
                                  "date_range": (str(date_range[0]), str(date_range[1])),
                              })
    return graph_data


def make_plots_plotly(graph_data, model_name, history_data):
    if not history_data:
        target = graph_data["target"]
    else:
        target = None

    times = graph_data["times"]
    preds = graph_data["preds"]

    tickformatstops = [
        # dict(dtickrange=[None, 1000], value="%H:%M:%S.%L ms"),
        # dict(dtickrange=[1000, 60000], value="%H:%M:%S s"),
        # dict(dtickrange=[60000, 3600000], value="%H:%M m"),
        dict(dtickrange=[None, 86400000], value="%H:%M"),
        dict(dtickrange=[86400000, 604800000], value="%b %d"),  # 1 day - week
        dict(dtickrange=[604800000, "M1"], value="%b %e, %Y"),  # week - month
        dict(dtickrange=["M1", "M12"], value="%b '%y"),  # month - year
        dict(dtickrange=["M12", None], value="%Y Y")
    ]

    fig_data = [
        go.Bar(x=times, y=preds,
               name=model_name,
               ),
    ]
    fig_title = 'Предсказания значений выработки электроэнергии'

    if target is not None:
        fig_data.append(go.Bar(x=times, y=target,
                               name='Фактическая мощность'
                               ))
    fig_title = 'Предсказания и фактические значения выработки электроэнергии'

    fig = {
        'data': fig_data,
        'layout': go.Layout(barmode='group',
                            title=fig_title,
                            xaxis={
                                "title": 'Часы',
                                #"tickmode": 'linear',
                                "tickformatstops": tickformatstops,
                            },
                            yaxis={"title": 'кВт*ч'},
                            )
    }

    return fig


def main():
    st.title("Energy.ai")
    st.write("Прогнозирование выработки электроэнергии на основе входных данных о погоде")

    @st.cache_data
    def load_model(model_id: str, model_type: str):
        """
        Получить с бэка модель по айдишнику
        :param model_id: uuid модели
        :param model_type: тип модели
        :return: объект модели
        """
        url = 'api'
        # url = '0.0.0.0'
        req = requests.get(f"http://{url}:{API_PORT}/api/get_model",
                           params={
                               'model_id': model_id,
                               "model_type": model_type,
                           }
                           )

        if req.status_code != 200:
            return None
        else:
            return from_bytes_stream(req.content)

    @st.cache_data
    def convert_df(df):
        return df.to_csv().encode('utf-8')

    @st.cache_data
    def get_metrics(model_id, model_type):
        pass

    def transform_data(data_x, data_y):
        data_x = data_x.drop_duplicates(subset=['dt'], keep='last')
        data_x.loc[:, 'dt'] = pd.to_datetime(data_x['dt'])
        data_y.loc[:, 'dt'] = pd.to_datetime(data_y['dt'])
        data = pd.merge(data_y, data_x, on='dt')
        data = data.drop(['fact_y', 'dt'], axis=1)
        data.replace('[^a-zA-Z0-9]', ' ', regex=True)
        data = data.rename(columns=lambda xx: re.sub('[^A-Za-z0-9_]+:[C]:', '', xx))

        y_test = data['fact_x']
        X_test = data.drop('fact_x', axis=1)

        return y_test, X_test, data

    chosen_power_plant = st.selectbox('Выберите объект', options=POWER_PLANTS.keys(),
                                      format_func=lambda x: POWER_PLANTS[x]['name'])
    #inference_data = st.file_uploader("Загрузка данных для предсказания выработки электроэнергии", type={"csv"})

    # try:
    #     inference_data = '../data/station_1/04.08.2022.csv'
    #
    #     inference_df = pd.read_csv(inference_data, sep=';', parse_dates=['dt'])
    #     start_date = inference_df['dt'].min().date()
    #     end_date = inference_df['dt'].max().date()
    # except Exception as e:
    #     st.error(e)

    # resp = requests.get()
    object_models = get_request(API_URL,
                                endpoint='models',
                                data={
                                    "object_id": chosen_power_plant
                                })
    model_name = st.selectbox("Выберите модель для объекта", options=object_models)

    # Выбоо дат
    date_range_dates = get_request(API_URL,
                                   endpoint='date_range',
                                   data={
                                       "object_id": chosen_power_plant,
                                       "model_name": model_name,
                                   })
    # start_date = date(2018, 1, 1)
    # end_date = date(2018, 12, 31)
    date_fmt = "%Y-%m-%d"
    date_range_dates = (datetime.strptime(date_range_dates[0].split()[0], date_fmt),
                        datetime.strptime(date_range_dates[1].split()[0], date_fmt),)

    forecast_date_start = datetime.today().date()
    forecast_date_end = forecast_date_start + timedelta(days=14)
    show_forecast_data = st.radio("Тип данных", options=["History", "Forecast"])
    show_history = show_forecast_data == "History"
    if show_history:
        date_range = st.date_input("Выбор периода", date_range_dates,
                                   min_value=date_range_dates[0],
                                   max_value=date_range_dates[1],
                                   )
    else:
        date_range = st.date_input("Выбор периода", (forecast_date_start, forecast_date_end),
                                   min_value=forecast_date_start,
                                   max_value=forecast_date_end,
                                   )

    if len(date_range) != 2:
        date_range = (date_range[0], date_range[0] + timedelta(days=1))
    else:
        date_range = (date_range[0], date_range[1] + timedelta(days=1))

    # st.plotly_chart(make_plots_plotly(models, data))
    # csv = convert_df(X_test)
    # st.download_button(
    #     "Скачать результат",
    #     to_excel(df_metrics),
    #     file_name="file.xlsx",
    #     mime="text/xlsx",
    #     key='download-xlsx'
    # )

    hide_dataframe_row_index = """
                                        <style>
                                        .row_heading.level0 {display:none}
                                        .blank {display:none}
                                        </style>
                                        """
    st.markdown(hide_dataframe_row_index, unsafe_allow_html=True)

    if show_history:
        metrics, business_metrics = post_request(API_URL,
                                                 endpoint='metrics',
                                                 data={
                                                      "object_id": chosen_power_plant,
                                                      "model_name": model_name,
                                                      "date_range": (str(date_range[0]), str(date_range[1])),
                                                 })
    else:
        metrics, business_metrics = '', ''

    # logger.info(type(metrics))

    # global_metrics = POWER_PLANTS[chosen_power_plant]["models"][model_name]["global_metrics"]
    # df_metrics = pd.DataFrame(global_metrics)
    # st.write("Общие метрики модели")
    # st.table(df_metrics)
    if show_history:
        df_metrics_period = pd.DataFrame().from_dict(metrics)
        st.write("ML Метрики модели на периоде")
        st.table(df_metrics_period)

        df_metrics_period = pd.DataFrame().from_dict(business_metrics)
        st.write("Бизнес метрики модели средние за сутки")
        st.table(df_metrics_period)

    # График
    graph_data = get_preds(chosen_power_plant, model_name, date_range)
    graph_data = pd.DataFrame().from_dict(graph_data)

    st.plotly_chart(make_plots_plotly(graph_data, model_name, history_data=show_history))


if __name__ == "__main__":
    main()














