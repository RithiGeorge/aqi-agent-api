from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
import base64

app = FastAPI()

# Load model once at startup
model = joblib.load("xgb_aqi_model.pkl")


@app.get("/")
def home():
    return {"message": "AQI Prediction API Running - Updated Version"}


@app.get("/predict")
def predict():
    try:
        # -------------------------------
        # Load historical data
        # -------------------------------
        df = pd.read_csv("historical_data.csv")

        # Clean column names
        df.columns = df.columns.str.strip()

        # Convert Date
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)

        # Force numeric conversion (VERY IMPORTANT)
        df["AQI"] = pd.to_numeric(df["AQI"], errors="coerce")
        df["PM2.5"] = pd.to_numeric(df["PM2.5"], errors="coerce")
        df["PM10"] = pd.to_numeric(df["PM10"], errors="coerce")

        # Drop rows where AQI is missing
        df = df.dropna(subset=["AQI"])

        if len(df) < 35:
            return {"error": "Not enough historical data in dataset"}

        # Sort by date
        df = df.sort_values("Date")

        # -------------------------------
        # Feature Engineering
        # -------------------------------

        for lag in [1, 2, 3, 7, 14, 30]:
            df[f"AQI_lag{lag}"] = df["AQI"].shift(lag)

        for lag in [1, 3, 7]:
            df[f"PM25_lag{lag}"] = df["PM2.5"].shift(lag)
            df[f"PM10_lag{lag}"] = df["PM10"].shift(lag)

        df["AQI_roll3"] = df["AQI"].rolling(3).mean()
        df["AQI_roll7"] = df["AQI"].rolling(7).mean()
        df["AQI_roll14"] = df["AQI"].rolling(14).mean()

        df["Month"] = df["Date"].dt.month
        df["DayOfWeek"] = df["Date"].dt.dayofweek

        # Drop NaN rows created by lag/rolling
        df = df.dropna()

        if len(df) == 0:
            return {"error": "Not enough usable data after feature engineering"}

        # -------------------------------
        # Prediction
        # -------------------------------

        features = [col for col in df.columns if "lag" in col or "roll" in col] + ["Month", "DayOfWeek"]

        last_row = df.iloc[-1]
        X_input = np.array([last_row[features]])

        prediction = model.predict(X_input)
        predicted_value = float(prediction[0])

        # -------------------------------
        # Weekly Chart
        # -------------------------------

        weekly_data = df.tail(7)

        plt.style.use("seaborn-v0_8-darkgrid")
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(
            weekly_data["Date"],
            weekly_data["AQI"],
            marker="o",
            linewidth=3,
            markersize=7
        )

        ax.set_title("Weekly AQI Trend - Bengaluru", fontsize=16, weight="bold")
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("AQI Level", fontsize=12)

        ax.set_xticks(weekly_data["Date"])
        ax.set_xticklabels(
            weekly_data["Date"].dt.strftime("%d %b"),
            rotation=45
        )

        # AQI category background zones
        ax.axhspan(0, 50, color="#2ecc71", alpha=0.08)
        ax.axhspan(51, 100, color="#27ae60", alpha=0.08)
        ax.axhspan(101, 200, color="#f39c12", alpha=0.08)
        ax.axhspan(201, 300, color="#e74c3c", alpha=0.08)
        ax.axhspan(301, 500, color="#8e44ad", alpha=0.08)

        ax.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=120)  # reduced DPI for email reliability
        buf.seek(0)

        chart_base64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)

        # -------------------------------
        # Return Response
        # -------------------------------

        return {
            "city": "Bengaluru",
            "date": str(datetime.today().date()),
            "predicted_next_aqi": round(predicted_value, 2),
            "weekly_chart": chart_base64
        }

    except Exception as e:
        return {"error": str(e)}
