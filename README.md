# Time Series Forecasting Project

## Walmart Sales Time Series Forecasting Dashboard

This project performs a complete time series analysis on Walmart weekly sales data across multiple stores. The goal is to forecast future sales and compare different modeling approaches in a real-world retail forecasting context.

The use of an interactive Streamlit dashboard allows users to explore sales patterns, generate forecasts, and compare model performance across different stores.

Retail businesses must accurately forecast demand to optimize inventory, staffing, and logistics. Sales data is influenced by trends, seasonality, and randomness, making time series forecasting essential for decision-making.

The question driving this project: 
How can we best forecast weekly sales across different Walmart stores using statistical and machine learning models?

## Dataset

- Source: Walmart Store Sales Dataset (45 stores)
- Frequency: Weekly sales data
- Key variables:
  - Store ID
  - Date
  - Weekly Sales

## Models Implemented

This project compares three major forecasting approaches:

### 1. Exponential Smoothing
Holt-Winters method: captures trend and seasonality

### 2. Box-Jenkins Methodology
SARIMA (Seasonal ARIMA): models autocorrelation and seasonal structure

### 3. Machine Learning Models 
Random Forest Regressor and Neural Network (MLP Regressor): Uses lag features and rolling statistics

## Streamlit Dashboard

Using Streamlit, this app includes:

- Interactive store selector
- Historical sales visualization
- SARIMA forecasting with confidence intervals
- Holt-Winters forecasting
- Machine learning forecasts
- Model performance comparison (MAE & RMSE)
- Visual comparison of predictions
