from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use("Agg")  # Required for cloud servers like Render
import matplotlib.pyplot as plt
import io
import base64

app = FastAPI()

# Load model once at startup
model = joblib.load("xgb_aqi_model.pkl")

@app.get("/")
def home():
    return {"message": "AQI Prediction API Running"}

@app.get("/predict")
def predict():

    try:
        # Load historical data
        df = pd.read_csv("historical_data.csv")

        # Clean column names
        df.columns = df.columns.str.strip()

        # Convert date
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)

        # 🔥 FORCE NUMERIC CONVERSION
        df["AQI"] = pd.to_numeric(df["AQI"], errors="coerce")
        df["PM2.5"] = pd.to_numeric(df["PM2.5"], errors="coerce")
        df["PM10"] = pd.to_numeric(df["PM10"], errors="coerce")

        # Drop rows where AQI is missing
        df = df.dropna(subset=["AQI"])
      

        # Filter city
        df_city = df.copy()

        if len(df_city) < 30:
            return {"error": "Not enough historical data for this city"}

        df_city = df_city.sort_values("Date")

        # -------- Feature Engineering --------
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
        predicted_value = float(prediction[0])

        # -------- Create Professional Weekly Chart --------
        weekly_data = df_city.tail(7)

        plt.style.use("seaborn-v0_8-darkgrid")
        fig, ax = plt.subplots(figsize=(8,5))

        ax.plot(
            weekly_data["Date"],
            weekly_data["AQI"],
            marker="o",
            linewidth=3,
            markersize=8
        )

        ax.set_title(f"Weekly AQI Trend - {city}", fontsize=16, weight="bold")
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("AQI Level", fontsize=12)

        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, linestyle="--", alpha=0.6)

        # Add AQI category color zones
        ax.axhspan(0, 50, alpha=0.1)
        ax.axhspan(51, 100, alpha=0.1)
        ax.axhspan(101, 200, alpha=0.1)
        ax.axhspan(201, 300, alpha=0.1)
        ax.axhspan(301, 500, alpha=0.1)

        plt.tight_layout()

        # Convert to Base64
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100)        
        buf.seek(0)
        chart_base64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close()

        return {
            "city": Bengaluru,
            "date": str(datetime.today().date()),
            "predicted_next_aqi": round(predicted_value, 2),
            "weekly_chart": chart_base64
        }

    except Exception as e:
        return {"error": str(e)}
