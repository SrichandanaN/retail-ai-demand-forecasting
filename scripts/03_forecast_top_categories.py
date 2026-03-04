import os
import sqlite3
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

DB_PATH = r"data/raw/olist.sqlite"
EXPORT_DIR = r"data/exports"
os.makedirs(EXPORT_DIR, exist_ok=True)

TOP_CATEGORIES = [
    "cama_mesa_banho",
    "beleza_saude",
    "esporte_lazer",
    "informatica_acessorios",
    "moveis_decoracao"
]

con = sqlite3.connect(DB_PATH)
df = pd.read_sql_query(
    "SELECT * FROM vw_daily_orders_by_category;",
    con
)
con.close()

df["order_date"] = pd.to_datetime(df["order_date"])
df = df[df["category"].isin(TOP_CATEGORIES)]

results = []

for category in TOP_CATEGORIES:

    cat_df = df[df["category"] == category].copy()
    cat_df = cat_df.sort_values("order_date")

    full_idx = pd.date_range(
        cat_df["order_date"].min(),
        cat_df["order_date"].max(),
        freq="D"
    )

    cat_df = (
        cat_df.set_index("order_date")
              .reindex(full_idx)
              .rename_axis("order_date")
              .reset_index()
    )

    cat_df["orders"] = cat_df["orders"].fillna(0)
    series = cat_df.set_index("order_date")["orders"]

    model = SARIMAX(
        series,
        order=(1,1,1),
        seasonal_order=(1,1,1,7),
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    fit = model.fit(disp=False)

    future_days = 30
    forecast = fit.forecast(steps=future_days)

    future_dates = pd.date_range(
        series.index.max() + pd.Timedelta(days=1),
        periods=future_days,
        freq="D"
    )

    out = pd.DataFrame({
        "order_date": future_dates,
        "category": category,
        "predicted_orders": forecast.values
    })

    results.append(out)

final_forecast = pd.concat(results)
final_forecast.to_csv(
    os.path.join(EXPORT_DIR, "top5_category_future_forecast.csv"),
    index=False
)

print("Saved: top5_category_future_forecast.csv")