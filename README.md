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

## Conclusion
This project explores time series forecasting for Walmart retail sales using both statistical and machine learning approaches. The Streamlit dashboard provides an interactive way to analyze store-level sales behavior and compare different forecasting methods.

By combining classical time series models (SARIMA and Holt-Winters) with machine learning models (Random Forest and Neural Networks), the project demonstrates different approaches to capturing trend, seasonality, and nonlinear patterns in real-world retail data.

The dashboard is designed as a decision-support tool that allows users to explore how different models behave under varying store conditions.
