import numpy as np
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.optimizers import Adam
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List

@dataclass
class Config:
  gru_layers: List[int]
  time_step: int = 24
  loss: str = "mean_squared_error"
  lr: float = 0.001
  batch_size: int = 32
  epochs: int = 20
  dropout: float= 0.2

def create_dataset(df: pd.DataFrame, time_step: int):
  '''Create sliding windows of time_step length'''
  X, y = [], []
  input  = df[features].values
  output = df["traffic_volume"].values
  for i in range(len(df.values) - time_step):
    X.append(input[i:(i + time_step)])
    y.append(output[i + time_step])
  return np.array(X), np.array(y)

def load_data(df: pd.DataFrame, features: list[int], time_step=1, test_size=0.2):
  '''Load, scale, and split data'''
  train_size = int(len(df) * (1 - test_size))
  train_df = df[:train_size].copy()
  test_df  = df[train_size:].copy()

  scaler = StandardScaler()
  scaler.fit(train_df[["traffic_volume"]])
  joblib.dump(scaler, Path(__file__).parent / "runs" / "traffic_volume_scaler.pkl")

  # Z-score standardisation on traffic flow
  train_df["traffic_volume"] = scaler.transform(train_df[["traffic_volume"]])
  test_df["traffic_volume"]  = scaler.transform(test_df[["traffic_volume"]])

  # X.shape = (samples, time_steps, features)
  # y.shape = (samples, )
  X_train, y_train = create_dataset(train_df, time_step)
  X_test, y_test   = create_dataset(test_df, time_step)

  return X_train, y_train, X_test, y_test, scaler

def build_model(config: Config):
  '''Build the GRU'''
  model = Sequential()

  gru_layers = config["gru_layers"]

  for i, units in enumerate(gru_layers):
    # Needs to be True if GRU layers are to be stacked
    return_sequences = (i < len(gru_layers) - 1)

    model.add(GRU(units=units,
                  return_sequences=return_sequences,
                  dropout=config["dropout"]
    ))
  
  model.add(Dense(units=1))

  model.compile(optimizer=Adam(learning_rate=config["lr"]),
                loss=config['loss'])
  
  return model

def run(config, X_train, y_train, X_test, y_test, scaler):
  run_id  = datetime.now().strftime("%Y%m%d%H%M%S")
  run_dir = parent_dir / "runs"

  model   = build_model(config)
  history = model.fit(X_train, y_train,
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            validation_split=0.2)
  
  predictions = model.predict(X_test)

  y_test_inversed      = scaler.inverse_transform(y_test.reshape(-1, 1))
  predictions_inversed = scaler.inverse_transform(predictions)

  # Metrics: Root Mean Squared Error and Mean Absolute Error
  rmse = np.sqrt(mean_squared_error(y_test_inversed, predictions_inversed))
  mae  = mean_absolute_error(y_test_inversed, predictions_inversed)

  # Save plots
  # Save prediction plot
  step = 50

  y_true = y_test_inversed[::step]
  y_pred = predictions_inversed[::step]

  x = np.arange(0, len(y_test_inversed), step)

  fig, ax = plt.subplots()

  ax.plot(x[:len(y_true)], y_true, label="Actual")
  ax.plot(x[:len(y_pred)], y_pred, label="Predicted")

  ax.set_title(f"Run {run_id}")
  ax.set_xlabel("Time Step")
  ax.set_ylabel("Traffic Volume")
  ax.legend()

  fig.savefig(run_dir / f"preds_{run_id}.png")
  plt.close(fig)

  # Save loss curve over time
  fig, ax = plt.subplots()
  epochs = np.arange(1, len(history.history["loss"]) + 1)
  ax.plot(epochs, history.history["loss"], label="train")
  ax.plot(epochs, history.history["val_loss"], label="val")
  ax.set_title("Loss Curve")
  ax.set_xlabel("Epoch")
  ax.set_xticks(np.arange(1, config["epochs"]+1, 5))
  ax.set_ylabel("Loss")
  ax.legend()
  fig.savefig(run_dir / f"loss_{run_id}.png")
  plt.close(fig)

  # Save model
  model.save(run_dir / f"model_{run_id}.keras")

  return {
    "run_id"        : run_id,
    **config,
    "rmse"          : rmse,
    "mae"           : mae,
    "final_val_loss": history.history["val_loss"][-1]
  }

# ------------------------------ #

os.system('cls')
parent_dir = Path(__file__).resolve().parent
time_step = 24

df = pd.read_csv(parent_dir / "data" / "model_data.csv", dtype={"SCATS Number": str})
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
joblib.dump(features, Path(__file__).parent / "runs" / "feature_columns.pkl")

df[features] = df[features].astype(np.float32)

X_train, y_train, X_test, y_test, scaler = load_data(df, features, time_step=time_step)

configs = [
  Config([32], dropout=0.1, time_step=time_step, epochs=50, loss='mean_squared_error'),
]

results = []

for config in configs:
  config = asdict(config)
  result = run(config, X_train, y_train, X_test, y_test, scaler)
  results.append(result)

df = pd.DataFrame(results)
df.to_csv(parent_dir / "runs" / "results.csv", mode="a", header=False, index=False)