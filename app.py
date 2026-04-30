import streamlit as st
import pandas as pd
import numpy as np

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.graphics.tsaplots import plot_acf

from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

import matplotlib.pyplot as plt

# Loading Data

@st.cache_data
def load_data():
    df = pd.read_csv("walmart-sales-dataset-of-45stores.csv")
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    df = df.sort_values('Date')
    return df

df = load_data()

total_sales_series = df.groupby('Date')['Weekly_Sales'].sum()

def get_store_series(df, store_id):
    temp = df[df['Store'] == store_id].copy()
    temp = temp.sort_values('Date')
    temp.set_index('Date', inplace=True)
    temp = temp.asfreq('W-FRI')
    return temp['Weekly_Sales']

st.title("Walmart Sales Forecasting Dashboard")

st.header("Global Sales Overview")

st.write("""
Before analyzing individual stores, we look at the total system-wide sales. 
This helps identify macro-trends and seasonal patterns (like Black Friday) 
that impact all locations simultaneously.
""")

st.line_chart(total_sales_series)

total_mae_baseline = total_sales_series.mean() * 0.1
st.info(f"System-wide average weekly revenue: ${total_sales_series.mean():,.2f}")

stores = sorted(df['Store'].unique())
selected_store = st.selectbox("Select Store", stores)

series = get_store_series(df, selected_store)

# Full Dataset Time Series

st.line_chart(series)

# Choose Forecast Horizon

horizon = st.slider("Forecast Horizon (weeks)", 4, 104, 52)

# SARIMA Forecast

def sarima_forecast(series, steps=horizon):
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

# ACF To Determine What Model to Use

st.subheader("Model Identification")
col_acf, col_text = st.columns([2, 1])

with col_acf:
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_acf(series.diff(52).dropna(), ax=ax, lags=52)
    st.pyplot(fig)

with col_text:
    st.write("""
    **Autocorrelation Analysis:**
    The ACF plot shows significant spikes at lag 52, confirming 
    strong **yearly seasonality**. This justifies our use of 
    Seasonal ARIMA (SARIMA) and Holt-Winters.
    """)

# Run SARIMA Forecast Button Section

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

    st.subheader("Residual Diagnostics (SARIMA)")

    residuals = results.resid

    fig2, ax2 = plt.subplots()
    ax2.plot(residuals)
    ax2.set_title("Residuals")
    st.pyplot(fig2)

def holt_winters_forecast(series, steps=horizon):
    model = ExponentialSmoothing(
        series,
        trend='add',
        seasonal='add',
        seasonal_periods=52
    )

    fit = model.fit()
    forecast = fit.forecast(steps)

    return fit, forecast

    st.subheader("Residual Diagnostics")
    fig_resid, ax_resid = plt.subplots(figsize=(8, 4))
    plot_acf(results.resid, ax=ax_resid)
    st.pyplot(fig_resid)
    st.write("If these bars are inside the blue, the SARIMA model has successfully captured the sales patterns!")

# Holt-Winters Forecast Button

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

def run_ml_models(df, store_id):
    X, y = create_ml_features(df, store_id)

    split = int(len(X) * 0.8)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

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

# Machine Learning Model Section

st.subheader("Machine Learning Forecast (Random Forest + Neural Net)")

if st.button("Run ML Models"):
    results = run_ml_models(df, selected_store)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(results["y_test"], label="Actual")
    ax.plot(results["rf_pred"], label="Random Forest")
    ax.plot(results["nn_pred"], label="Neural Net")

    ax.legend()
    st.pyplot(fig)

    st.write("### Model Performance Analysis")
    m1, m2, m3, m4 = st.columns(4)
    
    m1.metric("RF MAE", f"${results['rf_mae']:,.0f}", delta_color="inverse")
    m2.metric("NN MAE", f"${results['nn_mae']:,.0f}", delta_color="inverse")
    m3.metric("RF RMSE", f"${results['rf_rmse']:,.0f}")
    m4.metric("NN RMSE", f"${results['nn_rmse']:,.0f}")
    
    st.caption("Lower MAE (Mean Absolute Error) indicates a more reliable forecast for daily operations.")

def evaluate_models(series, ml_results):
    sarima_res, sarima_pred, _ = sarima_forecast(series, steps=len(ml_results["y_test"]))
    sarima_actual = series[-len(sarima_pred):]
    sarima_mae = mean_absolute_error(sarima_actual, sarima_pred)
    sarima_rmse = np.sqrt(mean_squared_error(sarima_actual, sarima_pred))

    _, hw_forecast = holt_winters_forecast(series, steps=len(ml_results["y_test"]))
    hw_actual = series[-len(hw_forecast):]
    hw_mae = mean_absolute_error(hw_actual, hw_forecast)
    hw_rmse = np.sqrt(mean_squared_error(hw_actual, hw_forecast))

    rf_mae = ml_results["rf_mae"]
    rf_rmse = ml_results["rf_rmse"]

    nn_mae = ml_results["nn_mae"]
    nn_rmse = ml_results["nn_rmse"]

    results_table = pd.DataFrame({
        "Model": ["SARIMA", "Holt-Winters", "Random Forest", "Neural Net"],
        "MAE": [sarima_mae, hw_mae, rf_mae, nn_mae],
        "RMSE": [sarima_rmse, hw_rmse, rf_rmse, nn_rmse]
    })

    return results_table

# Comparison Between all Models

st.subheader("Model Comparison")

selected_stores = st.multiselect(
    "Select 3 stores for comparison",
    stores,
    default=[20, 34, 33]
)

if st.button("Run Model Comparison"):

    if len(selected_stores) != 3:
        st.warning("Please select exactly 3 stores.")
    else:
        for store in selected_stores:

            st.markdown(f"### Store {store}")

            store_series = get_store_series(df, store)

            ml_results = run_ml_models(df, store)
            results_table = evaluate_models(store_series, ml_results)
    
            st.header("Final Model Recommendation")
            best_model = results_table.loc[results_table['MAE'].idxmin(), 'Model']
            st.success(f"Based on the lowest MAE, the recommended model for this store is: **{best_model}**")
            
            st.write(results_table)
    
        st.divider()
    














