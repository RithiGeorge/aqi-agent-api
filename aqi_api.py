from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

import matplotlib.pyplot as plt
import io
import base64
from fastapi.responses import JSONResponse

app = FastAPI()

model = joblib.load("xgb_aqi_model.pkl")

@app.get("/")
def home():
    return {"message": "AQI Prediction API Running"}


@app.get("/predict")
def predict(city: str):

    try:
        df = pd.read_csv("historical_data.csv")

        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

        # Filter by city
        df_city = df[df["City"] == city].copy()

        if len(df_city) < 30:
            return {"error": "Not enough historical data for this city"}

        df_city = df_city.sort_values("Date")

        # Recreate lag features
        for lag in [1,2,3,7,14,30]:
            df_city[f"AQI_lag{lag}"] = df_city["AQI"].shift(lag)

        for lag in [1,3,7]:
            df_city[f"PM25_lag{lag}"] = df_city["PM2.5"].shift(lag)
            df_city[f"PM10_lag{lag}"] = df_city["PM10"].shift(lag)

        df_city["AQI_roll3"] = df_city["AQI"].rolling(3).mean()
        df_city["AQI_roll7"] = df_city["AQI"].rolling(7).mean()
        df_city["AQI_roll14"] = df_city["AQI"].rolling(14).mean()

        df_city["Month"] = df_city["Date"].dt.month
        df_city["DayOfWeek"] = df_city["Date"].dt.dayofweek

        df_city = df_city.dropna()

        features = [col for col in df_city.columns if "lag" in col or "roll" in col] + ["Month","DayOfWeek"]

        last_row = df_city.iloc[-1]

        X_input = np.array([last_row[features]])

        prediction = model.predict(X_input)

        return {
            "city": city,
            "predicted_next_aqi": float(prediction[0])
        }

    except Exception as e:
        return {"error": str(e)}
# Get last 7 days AQI for chart
weekly_data = df_city.tail(7)

plt.figure(figsize=(6,4))
plt.plot(weekly_data["Date"], weekly_data["AQI"], marker='o')
plt.xticks(rotation=45)
plt.title(f"Weekly AQI Trend - {city}")
plt.tight_layout()

# Save to buffer
buf = io.BytesIO()
plt.savefig(buf, format="png")
buf.seek(0)

# Convert to base64
chart_base64 = base64.b64encode(buf.read()).decode("utf-8")
plt.close()

