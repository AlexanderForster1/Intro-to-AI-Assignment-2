'''
Simple baseline model for comparison (using Linear Regression)
'''
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv(
  Path(__file__).parent.parent.parent / "data" / "model_data.csv",
  dtype={"SCATS Number": str}
)

df.columns = df.columns.str.strip()
df["SCATS Number"] = df["SCATS Number"].str.zfill(4)
df = df.sort_values(["Date", "SCATS Number", "hour"]).reset_index(drop=True)
df = pd.get_dummies(df, columns=["SCATS Number"], prefix="SCATS")

features = [
  "hour_sin",
  "hour_cos",
  "day_sin",
  "day_cos",
  "is_weekend",
  "lat_scaled",
  "lon_scaled"
] + [col for col in df.columns if col.startswith("SCATS_")]

X = df[features].values
y = df["traffic_volume"].values

tscv = TimeSeriesSplit()

mae_scores = []
mse_scores = []
rmse_scores = []
r2_scores = []

scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
  X_train, X_test = X[train_idx], X[test_idx]
  y_train, y_test = y_scaled[train_idx], y_scaled[test_idx]

  model = LinearRegression()
  model.fit(X_train, y_train)

  preds_scaled = model.predict(X_test)
  preds = scaler_y.inverse_transform(preds_scaled.reshape(-1, 1))
  y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1))

  # Metrics
  mae = mean_absolute_error(y_true, preds)
  mse = mean_squared_error(y_true, preds)
  rmse = np.sqrt(mse)
  r2 = r2_score(y_true, preds)

  mae_scores.append(mae)
  mse_scores.append(mse)
  rmse_scores.append(rmse)
  r2_scores.append(r2)

  print(f"Fold {fold}: MAE={mae:.4f}, MSE={mse:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")

# Summary
print(f"MAE:  {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}")
print(f"MSE: {np.mean(mse_scores):.4f} ± {np.std(mse_scores):.4f}")
print(f"RMSE: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")
print(f"R2:   {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")