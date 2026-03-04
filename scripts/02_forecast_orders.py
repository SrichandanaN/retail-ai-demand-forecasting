import os
import sqlite3
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Connect to SQLite
DB_PATH = r"C:\Users\nsric\OneDrive\Desktop\AI_Sales_Dashboard_Project\data\raw\olist.sqlite"
EXPORT_DIR = r"data/exports"
os.makedirs(EXPORT_DIR, exist_ok=True)

con = sqlite3.connect(DB_PATH)
df = pd.read_sql_query("SELECT * FROM vw_daily_orders;", con)
con.close()

df["order_date"] = pd.to_datetime(df["order_date"])
df = df.sort_values("order_date")

# Fill missing dates
full_idx = pd.date_range(df["order_date"].min(), df["order_date"].max(), freq="D")
df = df.set_index("order_date").reindex(full_idx).rename_axis("order_date").reset_index()
df["orders"] = df["orders"].fillna(0)

series = df.set_index("order_date")["orders"]

# -------- Backtest (last 60 days) --------
# -------- Future Forecast (with confidence interval) --------
future_days = 30

model_full = SARIMAX(
    series,
    order=(1,1,1),
    seasonal_order=(1,1,1,7),
    enforce_stationarity=False,
    enforce_invertibility=False
)
fit_full = model_full.fit(disp=False)

forecast_res = fit_full.get_forecast(steps=future_days)

pred_mean = forecast_res.predicted_mean
conf_int = forecast_res.conf_int()

future_dates = pd.date_range(
    series.index.max() + pd.Timedelta(days=1),
    periods=future_days,
    freq="D"
)

future_out = pd.DataFrame({
    "order_date": future_dates,
    "predicted_orders": pred_mean.values,
    "lower_ci": conf_int.iloc[:, 0].values,
    "upper_ci": conf_int.iloc[:, 1].values
})

future_out.to_csv(
    os.path.join(EXPORT_DIR, "orders_future_with_ci.csv"),
    index=False
)

print("Saved: orders_future_with_ci.csv")