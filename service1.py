import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats
from scipy.stats import norm
import altair as alt
from io import StringIO
import catboost
import xgboost
import lightgbm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,accuracy_score,mean_squared_error

st.set_page_config(
    page_title="Basic helper"
)

uploaded_file = st.file_uploader("Choose a  csv file")



if uploaded_file is not None:
    # To read file as bytes:
    data = pd.read_csv(uploaded_file)
    st.write(data.head())
    st.subheader('Descriptive Statistics')
    st.write(data.describe())
    fill_option = st.selectbox("Fillna?",("Yes","No"))
    if fill_option=="YES":
        for c in data.columns:
            l = data[c].isnull()
            if data[c].type == object:
                mal = 0
                d = {}
                for i in range(l):
                    if l[i] == False:
                        mal = max(mal,len(str(data[c][i])))
                        s = str(data[c])
                        if s not in d:
                            d[s]=len(d)+1
                data[c] = data[c].fillna('0'*(mal+1))
                d['0'*(mal+1)]=len(d)+1
                data[c] = data[c].map(d)
            else:
                data[c] = data[c].fillna(min(data[c]-1))
    st.write(data.columns)
    data = data[[c for c in data.columns if (len(data[c]) == sum([1-int(t) for t in data[c].isnull()]))]]
    targ = st.selectbox("Target Column id,please set smaller than size of data column",(i for i in range(100)))
    exceptions = st.multiselect("Columns not to use", [i for i in range(100)])
    st.write(exceptions)
    mission_type = st.selectbox("Mission type", ["Regression", "Classification"])
    model_type = st.selectbox("Model to use", ["Xgb", "lgbm", "Catboost"])
    trigger = st.selectbox("Ready?",["No","Yes"])
    if trigger=="Yes":
        st.write(len(data.columns))
        forbcols = set()
        if exceptions!=None:
            forbcols = {data.columns[i] for i in exceptions}
        X_cols = [c for c in data.columns if (c!=data.columns[targ] and c not in forbcols)]
        X,y = data[X_cols],data[data.columns[targ]]
        test_ratio = 0.2
        test_ratio = st.slider("Test data ratio",min_value=0.01,max_value=0.99,step=0.01)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=42)
        if mission_type == "Regression":
            if model_type == "Xgb":
                model = xgboost.XGBRegressor()
                model.fit(X_train,y_train)
                mse = mean_squared_error((y_test,model.predict(X_test)))
                st.write("mse:",mse)
            elif model_type=="lgbm":
                model = lightgbm.LGBMRegressor()
                model.fit(X_train,y_train)
                mse = mean_squared_error((y_test,model.predict(X_test)))
                st.write("mse:",mse)
            else:
                model = catboost.CatBoostRegressor()
                model.fit(X_train,y_train)
                mse = mean_squared_error((y_test,model.predict(X_test)))
                st.write("mse:",mse)
        else:
            if model_type == "Xgb":
                model = xgboost.XGBClassifier()
                model.fit(X_train,y_train)
                f1_macro = f1_score(y_test,model.predict(X_test),average="macro")
                acc = accuracy_score(y_test,model.predict(X_test))
                st.write("f1:",f1_macro)
                st.write("acc",acc)
            elif model_type=="lgbm":
                model = lightgbm.LGBMClassifier()
                model.fit(X_train,y_train)
                f1_macro = f1_score(y_test,model.predict(X_test),average="macro")
                acc = accuracy_score(y_test,model.predict(X_test))
                st.write("f1:",f1_macro)
                st.write("acc",acc)
            else:
                model = catboost.CatBoostClassifier()
                model.fit(X_train,y_train)
                f1_macro = f1_score(y_test,model.predict(X_test),average="macro")
                acc = accuracy_score(y_test,model.predict(X_test))
                st.write("f1:",f1_macro)
                st.write("acc",acc)





    #exceptions = st.multiselect("Columns not to use",[c for c in data.columns if c!=targ])
    #st.write(exceptions)
    #st.write(data[[c for c in data.columns if (c not in exceptions and c !=targ)]])



