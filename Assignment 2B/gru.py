import numpy as np
import pandas as pd
import os
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

'''
NOTES

normalizing data for the RNNs
https://arxiv.org/abs/1510.01378

would incorporate TimeSeriesSplit but current model already encodes time structure so it's prob not that necessary and there are a bunch of data leakage issues i don't wanna think about

- early stopping?
- use exact location?
'''

PARENT_DIR = Path(__file__).resolve().parent
TEST_SIZE  = 0.2
TIME_STEP  = 96  # 24 hours

def create_dataset(data, time_step=1):
  '''Create sliding windows of time_step length'''
  X, y = [], []
  print(len(data))
  for i in range(len(data) - time_step):
    X.append(data[i:(i + time_step), :])
    y.append(data[i + time_step, -1])  # Last feature = traffic flow
  return np.array(X), np.array(y)

def load_data():
  '''Load, scale, and split data'''
  df = pd.read_csv(PARENT_DIR / "data" / "model_data.csv")

  train_size = int(len(df) * (1 - TEST_SIZE))
  train_df = df[:train_size].copy()
  test_df  = df[train_size:].copy()

  scaler = StandardScaler()
  scaler.fit(train_df[["traffic_volume"]])

  # Z-score standardisation on traffic flow
  train_df["traffic_volume"] = scaler.transform(train_df[["traffic_volume"]])
  test_df["traffic_volume"]  = scaler.transform(test_df[["traffic_volume"]])

  # X.shape = (samples, time_steps, features)
  # y.shape = (samples, )
  X_train, y_train = create_dataset(train_df.values, TIME_STEP)
  X_test, y_test   = create_dataset(test_df.values, TIME_STEP)

  return X_train, y_train, X_test, y_test, scaler

def build_model(config):
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
  run_dir = PARENT_DIR / "runs"

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

  # Save plot
  plt.plot(y_test_inversed[:200], label="Actual")
  plt.plot(predictions_inversed[:200], label="Predicted")
  plt.legend()
  plt.title(f"Run {run_id}")
  plt.savefig(run_dir / f"preds_{run_id}.png")
  plt.close()

  # Save loss curve over time
  plt.figure()
  plt.plot(history.history["loss"], label="train")
  plt.plot(history.history["val_loss"], label="val")
  plt.legend()
  plt.title("Loss Curve")
  plt.savefig(run_dir / f"loss_{run_id}.png")

  # Save model
  model.save(run_dir / f"model_{run_id}.keras")

  return {
    "run_id"        : run_id,
    **config,
    "rmse"          : rmse,
    "mae"           :  mae,
    "final_val_loss": history.history["val_loss"][-1]
  }

@dataclass
class Config:
  gru_layers: List[int]
  lr: float = 0.001
  batch_size: int = 32
  epochs: int = 20
  dropout: float= 0.2

def main():
  X_train, y_train, X_test, y_test, scaler = load_data()

  configs = [
    Config([32], dropout=0.0),
    Config([32]),
    Config([32], lr=0.0005),
    Config([32, 32]),
    Config([64, 32])
  ]

  results = []

  for config in configs:
    config = asdict(config)
    result = run(config, X_train, y_train, X_test, y_test, scaler)
    results.append(result)
  
  df = pd.DataFrame(results)
  df.to_csv(PARENT_DIR / "results.csv", index=False)

if __name__ == '__main__':
  os.system('cls')
  main()