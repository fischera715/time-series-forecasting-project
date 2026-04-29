import streamlit as st
import pandas as pd
import numpy as np

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

import matplotlib.pyplot as plt

@st.cache_data
def load_data():
    df = pd.read_csv("walmart-sales-dataset-of-45stores.csv")
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    df = df.sort_values('Date')
    return df

df = load_data()

def get_store_series(df, store_id):
    temp = df[df['Store'] == store_id].copy()
    temp = temp.sort_values('Date')
    temp.set_index('Date', inplace=True)
    temp = temp.asfreq('W-FRI')
    return temp['Weekly_Sales']

st.title("Walmart Sales Forecasting Dashboard")

stores = sorted(df['Store'].unique())
selected_store = st.selectbox("Select Store", stores)

series = get_store_series(df, selected_store)

st.line_chart(series)

def sarima_forecast(series, steps=52):
    model = SARIMAX(
        series,
        order=(1, 0, 1),
        seasonal_order=(0, 0, 1, 52),
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    results = model.fit(disp=False)

    forecast_obj = results.get_forecast(steps=steps)
    pred = forecast_obj.predicted_mean
    conf_int = forecast_obj.conf_int()

    return results, pred, conf_int

st.subheader("SARIMA Forecast")

if st.button("Run SARIMA Forecast"):
    results, pred, conf_int = sarima_forecast(series)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(series, label="Actual")
    ax.plot(pred, label="Forecast")

    ax.fill_between(
        conf_int.index,
        conf_int.iloc[:, 0],
        conf_int.iloc[:, 1],
        alpha=0.3
    )

    ax.legend()
    st.pyplot(fig)

def holt_winters_forecast(series, steps=52):
    model = ExponentialSmoothing(
        series,
        trend='add',
        seasonal='add',
        seasonal_periods=52
    )

    fit = model.fit()
    forecast = fit.forecast(steps)

    return fit, forecast

st.subheader("Holt-Winters Forecast")

if st.button("Run Holt-Winters Forecast"):
    fit, forecast = holt_winters_forecast(series)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(series, label="Actual")
    ax.plot(forecast, label="Forecast")

    ax.legend()
    st.pyplot(fig)

def create_ml_features(df, store_id):
    store = df[df['Store'] == store_id].copy()
    store = store.sort_values('Date')

    store['lag_1'] = store['Weekly_Sales'].shift(1)
    store['lag_2'] = store['Weekly_Sales'].shift(2)
    store['lag_52'] = store['Weekly_Sales'].shift(52)

    store['rolling_mean_4'] = store['Weekly_Sales'].rolling(window=4).mean()
    store['rolling_mean_12'] = store['Weekly_Sales'].rolling(window=12).mean()

    store = store.dropna()

    features = ['lag_1', 'lag_2', 'lag_52', 'rolling_mean_4', 'rolling_mean_12']

    X = store[features]
    y = store['Weekly_Sales']

    return X, y

from sklearn.metrics import mean_absolute_error, mean_squared_error

def run_ml_models(df, store_id):
    X, y = create_ml_features(df, store_id)

    split = int(len(X) * 0.8)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Random Forest
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    # Neural Net
    nn = MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=1000, random_state=42)
    nn.fit(X_train, y_train)
    nn_pred = nn.predict(X_test)

    results = {
        "rf_pred": rf_pred,
        "nn_pred": nn_pred,
        "y_test": y_test.values,
        "rf_mae": mean_absolute_error(y_test, rf_pred),
        "nn_mae": mean_absolute_error(y_test, nn_pred),
        "rf_rmse": np.sqrt(mean_squared_error(y_test, rf_pred)),
        "nn_rmse": np.sqrt(mean_squared_error(y_test, nn_pred))
    }

    return results

st.subheader("Machine Learning Forecast (Random Forest + Neural Net)")

if st.button("Run ML Models"):
    results = run_ml_models(df, selected_store)

    # Plot predictions
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(results["y_test"], label="Actual")
    ax.plot(results["rf_pred"], label="Random Forest")
    ax.plot(results["nn_pred"], label="Neural Net")

    ax.legend()
    st.pyplot(fig)

    # Metrics table
    st.write("### Model Performance")
    st.write({
        "RF MAE": results["rf_mae"],
        "NN MAE": results["nn_mae"],
        "RF RMSE": results["rf_rmse"],
        "NN RMSE": results["nn_rmse"]
    })





















