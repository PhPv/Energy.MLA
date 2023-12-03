import pandas as pd

class dmapeMetric: 
    def __init__(self):
        pass
    
    def __name__(self):
        return "dmape"
    
    def __str__(self):
        return "dmape"

    def __call__(self, estimator, X, y):
        y_pred = estimator.predict(X)
        df1 = pd.DataFrame(pd.Series(list(y_pred)), columns=["forecast"])
        df2 = y.reset_index(drop=True)
        df3 = X[["dayofyear", "hours"]].reset_index(drop=True)
        df_res = pd.concat([df1, df2, df3], axis=1)

        df_res_group = df_res.groupby(by=["dayofyear"]).sum()
        value_error = (df_res_group["forecast"] / df_res_group["target"]) - 1
        df_value_error = pd.concat(
            [value_error, abs(value_error)],
            axis=1,
            join="inner",
            keys=["dmape_error", "abs_dmape_error"],
        )
        df_res = pd.concat([df_res_group, df_value_error], axis=1)
        dmape = round(df_res['abs_dmape_error'].mean()*100,2)
        return dmape

class dwmapeMetric: 
    def __init__(self):
        pass
    
    def __name__(self):
        return "dwmape"
    
    def __str__(self):
        return "dwmape"

    def __call__(self, estimator, X, y):
        y_pred = estimator.predict(X)
        df1 = pd.DataFrame(pd.Series(list(y_pred)), columns=["forecast"])
        df2 = y.reset_index(drop=True)
        df3 = X[["dayofyear", "hours"]].reset_index(drop=True)
        df_res = pd.concat([df1, df2, df3], axis=1)

        df_res_group = df_res.groupby(by=["dayofyear"]).sum()
        max_electr = df_res_group["target"].max()
        w_value_error = ((df_res_group["forecast"] / df_res_group["target"]) - 1) * df_res_group["target"] / max_electr
        df_value_error = pd.concat(
            [w_value_error, abs(w_value_error)],
            axis=1,
            join="inner",
            keys=["dwmape_error", "abs_dwmape_error"],
        )
        df_res = pd.concat([df_res_group, df_value_error], axis=1)
        dwmape = round(df_res['abs_dwmape_error'].mean()*100,2)
        return dwmape

class wmapeMetric: 
    def __init__(self):
        pass
    
    def __name__(self):
        return "dwmape"
    
    def __str__(self):
        return "dwmape"

    def __call__(self, estimator, X, y):
        y_pred = estimator.predict(X)
        df1 = pd.DataFrame(pd.Series(list(y_pred)), columns=["forecast"])
        df2 = y.reset_index(drop=True)
        df3 = X[["dayofyear", "hours"]].reset_index(drop=True)
        df_res = pd.concat([df1, df2, df3], axis=1)

        max_electr = df_res["target"].max()
        w_value_error = ((df_res["forecast"] / df_res["target"]) - 1) * df_res["target"] / max_electr
        df_value_error = pd.concat(
            [w_value_error, abs(w_value_error)],
            axis=1,
            join="inner",
            keys=["wmape_error", "abs_wmape_error"],
        )
        df_res = pd.concat([df_res, df_value_error], axis=1)
        wmape = round(df_res['abs_wmape_error'].mean()*100,2)
        return wmape

class MoneyMetric: 
    def __init__(self):
        pass
    
    def __name__(self):
        return "money"
    
    def __str__(self):
        return "money"

    def __call__(self, estimator, X, y):
        y_pred = estimator.predict(X)
        df1 = pd.DataFrame(pd.Series(list(y_pred)), columns=["forecast"])
        df2 = y.reset_index(drop=True)
        df3 = X[["dayofyear", "hours"]].reset_index(drop=True)
        df_res = pd.concat([df1, df2, df3], axis=1)

        df_res_group = df_res.groupby(by=["dayofyear"]).sum()
        economic_df = pd.DataFrame([[],[],[]], columns = ["viruchka", "clear", "potencial"])
        economic_df = pd.concat([df3['dayofyear'], economic_df], axis=1)
        dif_all = df_res_group["forecast"] - df_res_group["target"]
        for i in range(len(dif_all)):
            economic_df[i]["viruchka"] = df_res_group[i]["forecast"] * 1.3
            economic_df[i]["potencial"] = df_res_group[i]["target"] * 1.3
            if dif_all[i] > 0:
                economic_df[i]["clear"] = economic_df[i]["viruchka"] - dif_all[i] * (1.3 + 1.3 * 0.3)
            else:
                economic_df[i]["clear"] = economic_df[i]["viruchka"]
        money = economic_df["clear"]/economic_df["target"] * 100

        return money

"""
# параметры по результатам работы optuna
params = {
    "reg_alpha": 0.03511823860228952,
    "reg_lambda": 0.7222690250563875,
    "colsample_bytree": 0.4,
    "subsample": 1.0,
    "learning_rate": 0.014,
    "max_depth": 100,
    "num_leaves": 208,
    "min_child_samples": 100,
    "cat_smooth": 88,
}
{'reg_alpha': 0.2890986687295702, 'reg_lambda': 0.0030163997576014715, 'colsample_bytree': 0.5, 'subsample': 0.8, 'learning_rate': 0.017, 'max_depth': 100, 'num_leaves': 664, 'min_child_samples': 101, 'min_data_per_groups': 65}
model = lgb.LGBMRegressor(**params)
model.fit(X_train, y_train)
"""