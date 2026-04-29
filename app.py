import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

model_choice = st.selectbox(
    "Select Model",
    ["Holt-Winters", "SARIMA", "Random Forest", "Neural Net"]
)

horizon = st.slider("Forecast Horizon (weeks)", 4, 52, 12)

if model_choice == "Holt-Winters":
    forecast = results[f"Store_{store_choice}"]["forecast_hw"][:horizon]

results[store] = {
    "hw": ...,
    "sarima": ...,
    "rf": ...,
    "nn": ...
}

