import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from graph_builder import _load_sites

_BASE = Path(__file__).parent
_DATA = _BASE / "data"
_MODELS = _BASE / "models"

_coord_scaler    = joblib.load(_DATA / "coord_scaler.pkl")
_feature_columns = joblib.load(_MODELS / "feature_columns.pkl")

def _load_models() -> dict:
  """load available trained models. missing model files are skipped"""
  from tensorflow.keras.models import load_model

  registry = {}

  lstm_path = _MODELS / "lstm" / "lstm_traffic_model.keras"
  lstm_scaler_path = _MODELS / "lstm_traffic_volume_scaler.pkl"
  if lstm_path.exists():
    registry["lstm"] = {
      "model" : load_model(lstm_path),
      "scaler": joblib.load(lstm_scaler_path) if lstm_scaler_path.exists() else None,
    }

  gru_path = _MODELS / "gru" / "gru_traffic_model.keras"
  gru_scaler_path = _MODELS / "gru_traffic_volume_scaler.pkl"
  if gru_path.exists():
    registry["gru"] = {
      "model" : load_model(gru_path),
      "scaler": joblib.load(gru_scaler_path) if gru_scaler_path.exists() else None,
    }

  rnn_path = _MODELS / "rnn" / "rnn_traffic_model.keras"
  if rnn_path.exists():
    registry["rnn"] = {
      "model" : load_model(rnn_path),
      "scaler": None,
    }

  return registry 

_models = _load_models()
_sites   = _load_sites(_DATA / "map_data.csv")


def predict(time: datetime, model_name: str = "rnn", time_step: int = 24) -> dict[int, float]:
  """
  predict hourly traffic flow for all SCATS sites at the given datetime
  returns a dict mapping each SCATS site number to its predicted flow (vehicles/hour)
  falls back to 0.0 for all sites if the requested model is unavailable
  """
  flow_dict = {}
  for scats_number, _ in _sites.items():
    flow = predict_single(scats_number, time, model_name, time_step)
    flow_dict[scats_number] = flow
  return flow_dict

def predict_single(
  scats_number: int, 
  time: datetime, 
  model_name: str = "rnn",
  time_step: int = 24
) -> float:
  '''predict hourly traffic flow for a single SCATS site at the given datetime
  returns predicted flow (vehicles/hour), falls back to 0.0 
  if the requested model is unavailable'''
  entry = _models.get(model_name)
  if entry is None or entry["model"] is None:
    return 0.0

  model = entry["model"]

  X = []

  for i in range(time_step - 1, -1, -1):
    t = time - timedelta(hours=i)
    X.append(_build_features(t, scats_number))

  X = np.array(X, dtype=np.float32)
  X = np.expand_dims(X, axis=0)

  assert X.shape[-1] == model.input_shape[-1], (
    f"Feature mismatch: got {X.shape[-1]}, model expects {model.input_shape[-1]}"
  )

  preds = model.predict(X, verbose=0)

  scaler = entry["scaler"]
  if scaler is not None:
    preds = scaler.inverse_transform(preds.reshape(-1, 1))

  if model_name == "gru":
    return float(preds[i][0])
  elif model_name == "rnn":
    return float(preds[i][-1][0])
  else:
    return float(preds[i].flat[0])


def _build_features(time: datetime, scats: int, use_dummies: bool=True) -> np.ndarray:
  """build the feature vector for one site at one timestep"""
  hour = time.hour
  day  = time.weekday()
  info = _sites[scats]

  coords = _coord_scaler.transform(
    pd.DataFrame([[info["lat"], info["lon"]]], columns=["NB_LATITUDE", "NB_LONGITUDE"])
  )[0]

  return np.array([
    np.sin(2 * np.pi * hour / 24),
    np.cos(2 * np.pi * hour / 24),
    np.sin(2 * np.pi * day  / 7),
    np.cos(2 * np.pi * day  / 7),
    int(day >= 5),
    *coords,
    *(_scats_one_hot(scats) if use_dummies else []),
  ], dtype=np.float32)
  

def _scats_one_hot(scats: int) -> np.ndarray:
  """return the one-hot encoding for a SCATS site"""
  scats_columns = [col for col in _feature_columns if col.startswith("SCATS_")]
  col_name = f"SCATS_{scats:04d}"
  if col_name not in scats_columns:
    raise ValueError(f"Unknown SCATS site: {scats}")
  vec = np.zeros(len(scats_columns), dtype=np.float32)
  vec[scats_columns.index(col_name)] = 1.0
  return vec


def available_models() -> list[str]:
  """Return the names of models that loaded successfully."""
  return list(_models.keys())


if __name__ == "__main__":
  preds = predict(datetime.now(), model_name="lstm")
  for sid, flow in preds.items():
    print(f"{sid}: {flow:1f} veh/hr")

  print(predict_single(970, datetime.now(), model_name="lstm"))