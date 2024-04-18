import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats
from scipy.stats import norm
import altair as alt
from io import StringIO
import seaborn as sns
import matplotlib.pyplot as plt
import catboost
import xgboost
import lightgbm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, mean_squared_error
import optuna


url = 'https://github.com/LamChiho/ds_app/blob/main/datasets/heart.csv'

st.set_page_config(
    page_title="Basic helper"
)
st.set_option('deprecation.showPyplotGlobalUse', False)

uploaded_file = st.file_uploader("Choose a  csv file smaller than 5mb.")




if uploaded_file is not None:
    # To read file as bytes:
    data = pd.read_csv(uploaded_file)
    st.write("data = pd.read_csv(#write down the path of the data here)")
    st.write("Here we can view some rows of data")

    st.write(data.head())
    st.write("data.head()")
    st.subheader('Descriptive Statistics')
    st.write("data.describe()")
    st.write(data.describe())
    fill_option = st.selectbox("Fillna?", ("Yes", "No"))
    st.write("We can fill the nulls in the dataframe by column.")
    if fill_option == "YES":
        for c in data.columns:
            l = data[c].isnull()
            if data[c].type == object:
                mal = 0
                d = {}
                for i in range(l):
                    if l[i] == False:
                        mal = max(mal, len(str(data[c][i])))
                        s = str(data[c])
                        if s not in d:
                            d[s] = len(d) + 1
                data[c] = data[c].fillna('0' * (mal + 1))
                d['0' * (mal + 1)] = len(d) + 1
                data[c] = data[c].map(d)
            else:
                data[c] = data[c].fillna(min(data[c] - 1))
    st.write(data.columns)
    st.write("Here we can use map function to map class items into int.")
    for c in data.columns:
        if data[c].dtype=="object":
            d = {}
            for i in range(data.shape[0]):
                if data[c][i] not in d:
                    d[data[c][i]] = len(d)
            data[c] = data[c].map(d)
    if len(data.columns) > 100:
        st.write("Sorry but this app can not fix datasets too large.")
    data = data[[c for c in data.columns if (len(data[c]) == sum([1 - int(t) for t in data[c].isnull()]))]]
    targ = st.selectbox("Target Column id,please set smaller than size of data column", (i for i in range(100)))
    phase = st.selectbox("Phase of action", ["EDA", "Model selection"])
    if phase == "EDA":
        st.write(data.corr())
        st.write(
            "We can view the relationship between columns.What would happen if we change the columns dropped before "
            "get the models trained?")
        sns.heatmap(data.corr())
        plt.title("Correlation map")
        st.pyplot()
    else:
        st.write("After viewing the relationship between columns, we can now decide columns not to use.")
        exceptions = st.multiselect("Columns not to use", [i for i in range(100)])
        st.write(exceptions)
        st.write(
            "Think twice before making the models. We can just run the models before deciding which model to use for "
            "the task.")
        mission_type = st.selectbox("Mission type", ["Regression", "Classification"])
        model_type = st.selectbox("Model to use", ["Xgb", "lgbm", "Catboost"])
        trigger = st.selectbox("Ready?", ["No", "Yes"])
        if trigger == "Yes":
            st.write(len(data.columns))
            forbcols = set()
            if exceptions != None:
                forbcols = {data.columns[i] for i in exceptions}
            X_cols = [c for c in data.columns if (c != data.columns[targ] and c not in forbcols)]
            X, y = data[X_cols], data[data.columns[targ]]
            st.write("test_ration = write down the ration of test size you want here.")
            st.write("X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=42)")
            test_ratio = st.slider("Test data ratio", min_value=0.01, max_value=0.99, step=0.01)
            trigger1 = st.selectbox("Test ratio fixed?", ["No", "Yes"])
            if trigger1 == "Yes":
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=42)
                if mission_type == "Regression":
                    if model_type == "Xgb":
                        model = xgboost.XGBRegressor()
                        st.write("model = xgboost.XGBRegressor()")
                        model.fit(X_train, y_train)
                        mse = mean_squared_error((y_test, model.predict(X_test)))
                        st.write("mse:", mse)
                    elif model_type == "lgbm":
                        model = lightgbm.LGBMRegressor()
                        model.fit(X_train, y_train)
                        st.write("model = lightgbm.LGBMRegressor()")
                        mse = mean_squared_error((y_test, model.predict(X_test)))
                        st.write("mse:", mse)
                    else:
                        model = catboost.CatBoostRegressor()
                        model.fit(X_train, y_train)
                        st.write("model = catboost.CatBoostRegressor()")
                        mse = mean_squared_error((y_test, model.predict(X_test)))
                        st.write("mse:", mse)
                    st.write("model.fit(X_train,y_train)")
                    st.write("mse = mean_squared_error((y_test,model.predict(X_test)))")
                else:
                    if model_type == "Xgb":
                        model = xgboost.XGBClassifier()
                        model.fit(X_train, y_train)
                        f1_macro = f1_score(y_test, model.predict(X_test), average="macro")
                        acc = accuracy_score(y_test, model.predict(X_test))
                        st.write("f1:", f1_macro)
                        st.write("acc", acc)
                        def objective(trial):
                        # Define the search space for hyperparameters
                            params = {
                                'objective': 'binary:logistic',
                                'eval_metric': 'error',
                                'verbosity': 0,
                                'booster': 'gbtree',
                                'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.1),
                                'max_depth': trial.suggest_int('max_depth', 3, 10),
                                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                                'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-6, 10.0),
                                'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-6, 10.0),
                            }

                            # Split data into training and validation sets
                            X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

                            # Initialize CatBoost classifier with parameters
                            model = xgboost.XGBClassifier(**params, verbose=False)

                            # Train the model
                            model.fit(X_train, y_train)

                            # Predict validation set
                            y_pred = model.predict(X_valid)

                            # Calculate accuracy
                            accuracy = accuracy_score(y_valid, y_pred)

                            # Return the accuracy as the objective value to minimize
                            return 1.000001-accuracy

                    # Set up Optuna study
                        study = optuna.create_study(direction='minimize')

                        # Optimize hyperparameters
                        study.optimize(objective, n_trials=15)

                        # Get the best hyperparameters
                        best_params = study.best_params
                        st.write(best_params)
                        st.write("We can use the optuna package to choose hyperparameters"
                                 "You can use it like"
                                 "model = xgboost.XGBClassifierr(**params=#Paste it here)"
                                 "Please remember that this is just using catboost classifier as an example"
                                 "Check the hyper parameters you chosen and think about what hyperparameters mean.")
                    elif model_type == "lgbm":
                        model = lightgbm.LGBMClassifier()
                        model.fit(X_train, y_train)
                        f1_macro = f1_score(y_test, model.predict(X_test), average="macro")
                        acc = accuracy_score(y_test, model.predict(X_test))
                        st.write("f1:", f1_macro)
                        st.write("acc", acc)
                        st.write("Sorry but the optimize for lightgbm is still developing.")
                    else:
                        model = catboost.CatBoostClassifier()
                        model.fit(X_train, y_train)
                        f1_macro = f1_score(y_test, model.predict(X_test), average="macro")
                        acc = accuracy_score(y_test, model.predict(X_test))
                        st.write("f1:", f1_macro)
                        st.write("acc", acc)
                        def objective(trial):
                        # Define the search space for hyperparameters
                            params = {
                                'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.1),
                                'depth': trial.suggest_int('depth', 2,11),
                                'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-3, 10.0),
                                'iterations': trial.suggest_int('iterations', 100, 1000),
                                'border_count': trial.suggest_int('border_count', 64, 255),
                                'random_strength': trial.suggest_loguniform('random_strength', 0.01, 10.0),
                            }

                            # Split data into training and validation sets
                            X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

                            # Initialize CatBoost classifier with parameters
                            model = catboost.CatBoostClassifier(**params, verbose=False)

                            # Train the model
                            model.fit(X_train, y_train)

                            # Predict validation set
                            y_pred = model.predict(X_valid)

                            # Calculate accuracy
                            accuracy = accuracy_score(y_valid, y_pred)

                            # Return the accuracy as the objective value to minimize
                            return 1.000001-accuracy

                    # Set up Optuna study
                        study = optuna.create_study(direction='minimize')

                        # Optimize hyperparameters
                        study.optimize(objective, n_trials=30)

                        # Get the best hyperparameters
                        best_params = study.best_params
                        st.write(best_params)
                        st.write("We can use the optuna package to choose hyperparameters"
                                 "You can use it like"
                                 "model = catboost.CatBoostClassifier(**params=#Paste it here, verbose=False)"
                                 "Please remember that this is just using catboost classifier as an example"
                                 "Check the hyper parameters you chosen and think about what hyperparameters mean.")
                    st.write("model.fit(X_train,y_train)")
                    st.write("f1_macro = f1_score(y_test,model.predict(X_test),average=" + """macro""" + ")")
                    st.write("acc = accuracy_score(y_test,model.predict(X_test))")

    # exceptions = st.multiselect("Columns not to use",[c for c in data.columns if c!=targ])
    # st.write(exceptions)
    # st.write(data[[c for c in data.columns if (c not in exceptions and c !=targ)]])

# streamlit run service1.py


