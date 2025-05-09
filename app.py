
import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import pandas as pd

# Load model
model = joblib.load("xgb_model.pkl")

st.title("Energy Load Prediction and Analysis (XGBoost Model)")
st.markdown("## Predict energy load and explore historical consumption patterns")

# --- Prediction Section ---
st.subheader("🔮 Predict Future Load")

hour = st.slider("Hour of Day", 0, 23, 12)
dayofweek = st.slider("Day of Week (0=Monday, 6=Sunday)", 0, 6, 3)
month = st.slider("Month", 1, 12, 6)

# Lag features
lags = []
for i in range(1, 25):
    lag = st.number_input(f"Lag {i} (MW {i} hours ago)", value=3000.0 - i * 10)
    lags.append(lag)

features = [hour, dayofweek, month] + lags
features_array = np.array(features).reshape(1, -1)

if st.button("Predict Load"):
    prediction = model.predict(features_array)[0]
    st.success(f"Predicted Load: {prediction:.2f} MW")

    # Plot lag trend and prediction
    st.markdown("### Load Trend with Prediction")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(1, 25), lags, marker='o', label="Past 24 Hours")
    ax.axhline(y=prediction, color='r', linestyle='--', label=f"Predicted ({prediction:.2f} MW)")
    ax.set_xlabel("Lag Hour")
    ax.set_ylabel("Load (MW)")
    ax.set_title("Energy Load Trend")
    ax.legend()
    st.pyplot(fig)

# --- Forecast Section ---
st.subheader("📅 30-Day Forecast Monthly Average (Simulated)")
# Simulated average data for each month
monthly_avg = {1: 3050, 2: 3000, 3: 3100, 4: 3200, 5: 3300, 6: 3400, 7: 3500, 8: 3450,
               9: 3350, 10: 3250, 11: 3150, 12: 3100}
df_monthly = pd.DataFrame({
    "Month": list(monthly_avg.keys()),
    "Avg Load (MW)": list(monthly_avg.values())
})
fig2, ax2 = plt.subplots()
ax2.bar(df_monthly["Month"], df_monthly["Avg Load (MW)"], color='skyblue')
ax2.set_xlabel("Month")
ax2.set_ylabel("Avg Load (MW)")
ax2.set_title("Simulated Monthly Avg Energy Consumption")
st.pyplot(fig2)

# --- Holiday vs Non-Holiday ---
st.subheader("🎉 Energy Consumption: Holiday vs Non-Holiday (Simulated)")
holiday_data = {"Day Type": ["Holiday", "Non-Holiday"], "Avg Load (MW)": [2950, 3250]}
df_holiday = pd.DataFrame(holiday_data)
fig3, ax3 = plt.subplots()
ax3.bar(df_holiday["Day Type"], df_holiday["Avg Load (MW)"], color=["green", "orange"])
ax3.set_ylabel("Avg Load (MW)")
ax3.set_title("Simulated Energy Use: Holiday vs Non-Holiday")
st.pyplot(fig3)
