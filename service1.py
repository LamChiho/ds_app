import streamlit as st
import pandas as pd
import catboost
import xgboost
import lightgbm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, mean_squared_error
from sklearn.model_selection import cross_validate,KFold
import optuna


k = 5
kf = KFold(n_splits=k)

st.set_page_config(
    page_title="Basic helper"
)

uploaded_file = st.file_uploader("Choose a  csv file")

mission_type = "Xgb"
def objective(trial,chosen=mission_type):
    params = {
        # "learning_rate":trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
        # "n_estimators":trial.suggest_int("n_estimators", 1000, 8000),
        "max_depth": trial.suggest_categorical('max_depth', [6, 7, 8, 9, 10, 11, 13, 15, 17, 21, 25, 29]),
    }

    sc = 0
    for train_index, test_index in kf.split(data):
        if chosen == "Xgb":#, "lgbm", "Catboost"
            model = xgboost.XGBClassifier(**params)
        elif chosen == "lgbm":
            model = lightgbm.LGBMClassifier(**params)
        else:
            model = catboost.CatBoostClassifier(**params)

        # model = xgboost.XGBClassifier(**param)
        X_train1, X_test1 = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train1, y_test1 = y[train_index], y[test_index]
        #X_res, y_res = ros.fit_resample(X_train, y_train)
        model.fit(X_train1, y_train1, eval_set=[(X_test1, y_test1)], early_stopping_rounds=100, verbose=False)
        preds = model.predict(X_test1)
        sc += f1_score(y_test1, preds, average='macro')

    return sc

def objective_regress(trial,chosen=mission_type):
    params = {
        # "learning_rate":trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
        # "n_estimators":trial.suggest_int("n_estimators", 1000, 8000),
        "max_depth": trial.suggest_categorical('max_depth', [6, 7, 8, 9, 10, 11, 13, 15, 17, 21]),
    }

    sc = 0
    for train_index, test_index in kf.split(data):
        if chosen == "Xgb":#, "lgbm", "Catboost"
            model = xgboost.XGBClassifier(**params)
        elif chosen == "lgbm":
            model = lightgbm.LGBMClassifier(**params)
        else:
            model = catboost.CatBoostClassifier(**params)

        # model = xgboost.XGBClassifier(**param)
        X_train1, X_test1 = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train1, y_test1 = y[train_index], y[test_index]
        #X_res, y_res = ros.fit_resample(X_train, y_train)
        model.fit(X_train1, y_train1, eval_set=[(X_test1, y_test1)], early_stopping_rounds=100, verbose=False)
        preds = model.predict(X_test1)
        sc += mse(y_test1, preds)

    return sc


if uploaded_file is not None:
    # To read file as bytes:
    data = pd.read_csv(uploaded_file)
    st.write(data.head())
    st.subheader('Descriptive Statistics')
    st.write(data.describe())
    fill_option = st.selectbox("Fillna?", ("Yes", "No"))
    if fill_option == "YES":
        for c in data.columns:
            l = data[c].isnull()
            if data[c].type == object:
                mal = 0
                d = {}
                for i in range(l):
                    if l[i] is False:
                        mal = max(mal, len(str(data[c][i])))
                        s = str(data[c])
                        if s not in d:
                            d[s] = len(d)+1
                data[c] = data[c].fillna('0'*(mal+1))
                d['0'*(mal+1)] = len(d)+1
                data[c] = data[c].map(d)
            else:
                data[c] = data[c].fillna(min(data[c]-1))
    st.write(data.columns)
    data = data[[c for c in data.columns if (len(data[c]) == sum([1-int(t) for t in data[c].isnull()]))]]
    targ = st.selectbox("Target Column id,please set smaller than size of data column", (i for i in range(100)))
    exceptions = st.multiselect("Columns not to use", [i for i in range(100)])
    st.write(exceptions)
    mission_type = st.selectbox("Mission type", ["Regression", "Classification"])
    model_type = st.selectbox("Model to use", ["Xgb", "lgbm", "Catboost"])
    test_ratio = 0.2
    test_ratio = st.slider("Test data ratio", min_value=0.01, max_value=0.99, step=0.01)
    trigger = st.selectbox("Ready?", ["No", "Yes"])
    if trigger == "Yes":
        forbcols = set()
        if exceptions is not None:
            forbcols = {data.columns[i] for i in exceptions}
        X_cols = [c for c in data.columns if (c != data.columns[targ] and c not in forbcols)]
        X, y = data[X_cols], data[data.columns[targ]]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=42)
        if mission_type == "Regression":
            if model_type == "Xgb":
                model = xgboost.XGBRegressor()
                model.fit(X_train, y_train)
                mse = mean_squared_error((y_test, model.predict(X_test)))
                st.write("mse:", mse)
                study = optuna.create_study(direction='minimize')
                study.optimize(objective_regress(), n_trials=3)
                st.write(study.best_params)
                model1 =xgboost.XGBRegressor(**study.best_params)
                model1.fit(X_train, y_train)
                mse1 = mean_squared_error((y_test, model1.predict(X_test)))
                st.write("mse:", mse1)
                st.write("paras = *paste params shown above*")
                st.write("model = xgboost.XGBRegressor(**paras)")
                st.write("model.fit(X_train, y_train)")
                st.write("mse = mean_squared_error((y_test, model.predict(X_test)))")
            elif model_type == "lgbm":
                model = lightgbm.LGBMRegressor()
                model.fit(X_train, y_train)
                mse = mean_squared_error((y_test, model.predict(X_test)))
                st.write("mse:", mse)
                study = optuna.create_study(direction='minimize')
                study.optimize(objective_regress(), n_trials=3)
                st.write(study.best_params)
                model1 =lightgbm.LGBMRegressor(**study.best_params)
                model1.fit(X_train, y_train)
                mse1 = mean_squared_error((y_test, model1.predict(X_test)))
                st.write("mse:", mse1)
                st.write("paras = *paste params shown above*")
                st.write("model = lightgbm.LGBMRegressor(**paras)")
                st.write("model.fit(X_train, y_train)")
                st.write("mse = mean_squared_error((y_test, model.predict(X_test)))")
            else:
                model = catboost.CatBoostRegressor()
                model.fit(X_train, y_train)
                mse = mean_squared_error((y_test, model.predict(X_test)))
                st.write("mse:", mse)
                study = optuna.create_study(direction='minimize')
                study.optimize(objective_regress(), n_trials=3)
                st.write(study.best_params)
                model1 = catboost.CatBoostRegressor(**study.best_params)
                model1.fit(X_train, y_train)
                mse1 = mean_squared_error((y_test, model1.predict(X_test)))
                st.write("mse:", mse1)
                st.write("paras = *paste params shown above*")
                st.write("model = catboost.CatBoostRegressor(**paras)")
                st.write("model.fit(X_train, y_train)")
                st.write("mse = mean_squared_error((y_test, model.predict(X_test)))")
        else:
            if model_type == "Xgb":
                model = xgboost.XGBClassifier()
                model.fit(X_train, y_train)
                f1_macro = f1_score(y_test, model.predict(X_test), average="macro")
                acc = accuracy_score(y_test, model.predict(X_test))
                study = optuna.create_study(direction='maximize')
                study.optimize(objective, n_trials=3)
                st.write(study.best_params)
                st.write("f1:", f1_macro)
                st.write("acc", acc)
                model1 = xgboost.XGBClassifier(**study.best_params)
                model1.fit(X_train, y_train)
                f1_macro1 = f1_score(y_test, model1.predict(X_test), average="macro")
                st.write(f1_macro1)
                st.write("paras = *paste params shown above*")
                st.write("model = xgboost.XGBClassifier(**paras)")
                st.write("model.fit(X_train, y_train)")
                st.write('f1_macro = f1_score(y_test, model.predict(X_test), average="macro")')
            elif model_type == "lgbm":
                model = lightgbm.LGBMClassifier()
                model.fit(X_train, y_train)
                f1_macro = f1_score(y_test, model.predict(X_test), average="macro")
                acc = accuracy_score(y_test, model.predict(X_test))
                study = optuna.create_study(direction='maximize')
                study.optimize(objective, n_trials=3)
                st.write(study.best_params)
                st.write("f1:", f1_macro)
                st.write("acc", acc)
                model1 = lightgbm.LGBMClassifier(**study.best_params)
                model1.fit(X_train, y_train)
                f1_macro1 = f1_score(y_test, model1.predict(X_test), average="macro")
                st.write(f1_macro1)
                st.write("paras = *paste params shown above*")
                st.write("model = lightgbm.LGBMClassifier(**paras)")
                st.write("model.fit(X_train, y_train)")
                st.write('f1_macro = f1_score(y_test, model.predict(X_test), average="macro")')
            else:
                model = catboost.CatBoostClassifier()
                model.fit(X_train, y_train)
                f1_macro = f1_score(y_test, model.predict(X_test), average="macro")
                acc = accuracy_score(y_test, model.predict(X_test))
                study = optuna.create_study(direction='maximize')
                study.optimize(objective, n_trials=3)
                st.write(study.best_params)
                st.write("f1:", f1_macro)
                st.write("acc", acc)
                model1 = catboost.CatBoostClassifier(**study.best_params)
                model1.fit(X_train, y_train)
                f1_macro1 = f1_score(y_test, model1.predict(X_test), average="macro")
                st.write(f1_macro1)
                st.write("paras = *paste params shown above*")
                st.write("model = catboost.CatBoostClassifier(**paras)")
                st.write("model.fit(X_train, y_train)")
                st.write('f1_macro = f1_score(y_test, model.predict(X_test), average="macro")')
