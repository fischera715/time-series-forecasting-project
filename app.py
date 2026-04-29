import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("walmart-sales-dataset-of-45stores.csv")
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

st.title("Walmart Time Series Forecasting Dashboard")

# -----------------------------
# STORE SELECTION
# -----------------------------
stores = [1, 20, 33]
store_choice = st.selectbox("Select Store", stores)

store_df = df[df['Store'] == store_choice].sort_values('Date')
store_df = store_df.set_index('Date')
series = store_df['Weekly_Sales']

# -----------------------------
# TRAIN / TEST SPLIT
# -----------------------------
train_size = int(len(series) * 0.8)
train, test = series[:train_size], series[train_size:]

# -----------------------------
# HOLT-WINTERS MODEL
# -----------------------------
hw_model = ExponentialSmoothing(
    train,
    trend='add',
    seasonal='add',
    seasonal_periods=52
)

hw_fit = hw_model.fit()
hw_forecast = hw_fit.forecast(len(test))

# -----------------------------
# SARIMA MODEL
# -----------------------------
sarima_model = SARIMAX(
    train,
    order=(1, 0, 1),
    seasonal_order=(1, 0, 1, 52),
    enforce_stationarity=False,
    enforce_invertibility=False
)

sarima_fit = sarima_model.fit(disp=False)
sarima_forecast = sarima_fit.forecast(len(test))

# -----------------------------
# ERROR METRICS
# -----------------------------
def get_metrics(true, pred):
    mae = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    return mae, rmse

hw_mae, hw_rmse = get_metrics(test, hw_forecast)
sarima_mae, sarima_rmse = get_metrics(test, sarima_forecast)

# -----------------------------
# DISPLAY ACTUAL VS FORECASTS
# -----------------------------
st.subheader("Actual vs Forecast Comparison")

fig, ax = plt.subplots()

ax.plot(train.index, train, label="Train Data", alpha=0.5)
ax.plot(test.index, test, label="Actual Test Data", color="black")

ax.plot(test.index, hw_forecast, label="Holt-Winters Forecast")
ax.plot(test.index, sarima_forecast, label="SARIMA Forecast")

ax.legend()
st.pyplot(fig)

# -----------------------------
# METRICS TABLE
# -----------------------------
st.subheader("Model Performance")

metrics_df = pd.DataFrame({
    "Model": ["Holt-Winters", "SARIMA"],
    "MAE": [hw_mae, sarima_mae],
    "RMSE": [hw_rmse, sarima_rmse]
})

st.dataframe(metrics_df)

# -----------------------------
# INSIGHT
# -----------------------------
best_model = metrics_df.loc[metrics_df["RMSE"].idxmin(), "Model"]

st.success(f"Best performing model for Store {store_choice}: {best_model}")







