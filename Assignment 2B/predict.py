import numpy as np
import joblib
import pandas as pd
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
from graph_builder import _load_sites
from pathlib import Path

coord_scaler = joblib.load(Path(__file__).parent / "data" / "coord_scaler.pkl")
traffic_volume_scaler = joblib.load(Path(__file__).parent / "runs" / "traffic_volume_scaler.pkl")
feature_columns = joblib.load(Path(__file__).parent / "runs" / "feature_columns.pkl")

def predict(
    time: datetime, 
    model, 
    time_step=24) -> dict[int, float]:
  '''Predicts the hourly traffic flow at the given time across all SCATS sites in Boroondara, returning a dictionary mapping each SCATS site number to its predicted traffic flow'''

  sites = _load_sites(Path(__file__).resolve().parent / "data" / "map_data.csv")
  site_ids = list(sites.keys())

  X = []

  for scats_number, site_info in sites.items():
    scats_number = f"{scats_number:04d}"
    sequence = []
    # Create history of 24 previous timesteps
    for i in range(time_step, 0, -1):
      t = time - timedelta(hours=i)

      features = _build_features(
        time=t,
        scats_number=scats_number,
        sites=sites,
      )

      sequence.append(features)
    X.append(sequence)
    
  X = np.array(X, dtype=np.float32)

  # Shape check
  assert X.shape[-1] == model.input_shape[-1], \
    f"Feature mismatch: {X.shape[-1]} vs {model.input_shape[-1]}"
  
  preds = model.predict(X, verbose=0)
  preds_real = traffic_volume_scaler.inverse_transform(preds)

  return {
    site_ids[i]: float(preds_real[i][0])
    for i in range(len(site_ids))
  }

def _build_features(
    time: datetime, 
    scats_number: int, 
    sites: dict[int, dict]) -> np.ndarray:

  hour = time.hour
  day = time.weekday()

  site_info = sites[int(scats_number)]

  return np.array([
    np.sin(2 * np.pi * hour / 24),  # hour_sin
    np.cos(2 * np.pi * hour / 24),  # hour_cos
    np.sin(2 * np.pi * day / 7),    # day_sin
    np.cos(2 * np.pi * day / 7),    # day_cos
    int(day >= 5),
    *coord_scaler.transform(
      pd.DataFrame([[site_info["lat"], site_info["lon"]]], 
                   columns=["NB_LATITUDE", "NB_LONGITUDE"])
    )[0],
    *_scats_one_hot(scats_number)
  ], dtype=np.float32)

def _scats_one_hot(scats_number):
  scats_columns = [col for col in feature_columns if col.startswith("SCATS_")]
  vec = np.zeros(len(scats_columns), dtype=np.float32)

  col_name = f"SCATS_{scats_number}"

  if col_name not in scats_columns:
    raise ValueError(f"Unknown SCATS: {scats_number}")
  idx = scats_columns.index(col_name)
  vec[idx] = 1.0
  return vec

# Test run
model = load_model(Path(__file__).parent / "runs" / "model_20260518162915.keras")
preds = predict(datetime.now(), model)
for key, value in preds.items():
  print(f"{key}: {value}")