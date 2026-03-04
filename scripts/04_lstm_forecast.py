import os
import sqlite3
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras import layers

DB_PATH = r"data/raw/olist.sqlite"
EXPORT_DIR = r"data/exports"
os.makedirs(EXPORT_DIR, exist_ok=True)

con = sqlite3.connect(DB_PATH)
df = pd.read_sql_query("SELECT * FROM vw_daily_orders;", con)
con.close()

df["order_date"] = pd.to_datetime(df["order_date"])
df = df.sort_values("order_date")

full_idx = pd.date_range(df["order_date"].min(), df["order_date"].max(), freq="D")
df = df.set_index("order_date").reindex(full_idx).rename_axis("order_date").reset_index()
df["orders"] = df["orders"].fillna(0)

values = df["orders"].values.reshape(-1,1)

scaler = MinMaxScaler()
scaled = scaler.fit_transform(values)

def create_sequences(data, lookback=30):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])
        y.append(data[i])
    return np.array(X), np.array(y)

LOOKBACK = 30
X, y = create_sequences(scaled, LOOKBACK)

test_days = 60
split = len(X) - test_days

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = tf.keras.Sequential([
    layers.LSTM(64, input_shape=(LOOKBACK,1)),
    layers.Dense(1)
])

model.compile(optimizer="adam", loss="mse")

model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

pred_scaled = model.predict(X_test)
pred = scaler.inverse_transform(pred_scaled)
actual = scaler.inverse_transform(y_test)

mae = mean_absolute_error(actual, pred)
rmse = np.sqrt(mean_squared_error(actual, pred))

print(f"LSTM MAE: {mae:.2f}")
print(f"LSTM RMSE: {rmse:.2f}")

dates = df["order_date"].iloc[-test_days:].values

out = pd.DataFrame({
    "order_date": dates,
    "actual_orders": actual.flatten(),
    "lstm_predicted": pred.flatten()
})

out.to_csv(os.path.join(EXPORT_DIR, "lstm_backtest.csv"), index=False)

print("Saved: lstm_backtest.csv")