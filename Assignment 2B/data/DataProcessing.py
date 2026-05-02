import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('Scats Data October 2006.csv', dtype={"SCATS Number": str})

df_map = df[["SCATS Number", "Location", "NB_LATITUDE", "NB_LONGITUDE"]].copy()
df_map = df_map.drop_duplicates(subset=["SCATS Number"])
df_map.to_csv('map_data.csv', index=False)

coord_scaler = StandardScaler()
df[["lat_scaled", "lon_scaled"]] = coord_scaler.fit_transform(df[["NB_LATITUDE", "NB_LONGITUDE"]])
df_temp = df.drop(columns=["Location", "NB_LATITUDE", "NB_LONGITUDE", "CD_MELWAY", "HF VicRoads Internal", "VR Internal Stat", "VR Internal Loc", "NB_TYPE_SURVEY"])

df_model = df_temp.melt(
    id_vars=["SCATS Number", "Date", "lat_scaled", "lon_scaled"],
    var_name="time",
    value_name="traffic_volume"

)


df_model["Date"] = pd.to_datetime(df_model["Date"], format="%d/%m/%Y")
df_model["Day"] = df_model["Date"].dt.dayofweek

df_model["datetime"] = pd.to_datetime(df_model["Date"].astype(str) + ' ' + df_model["time"].astype(str))

df_model["minutes_since_midnight"] = (df_model["datetime"].dt.hour * 60 + df_model["datetime"].dt.minute)
df_model["time_index"] = df_model["minutes_since_midnight"] // 15
df_model["time_sin"] = np.sin(2 * np.pi * df_model["time_index"] / 96)
df_model["time_cos"] = np.cos(2 * np.pi * df_model["time_index"] / 96)
df_model["day_sin"] = np.sin(2 * np.pi * df_model["Day"] / 7)
df_model["day_cos"] = np.cos(2 * np.pi * df_model["Day"] / 7)

df_model["is_weekend"] = df_model["Day"].isin([5, 6]).astype(int)
df_model["SCATS_id"] = df_model["SCATS Number"].astype('category').cat.codes
df_id = df_model[["SCATS Number", "SCATS_id", "lat_scaled", "lon_scaled"]].copy()
df_id = df_id.drop_duplicates(subset=["SCATS Number"])
df_id.to_csv('id_mapping.csv', index=False)
df_model = df_model[["SCATS_id", "time_sin", "time_cos", "day_sin", "day_cos", "is_weekend", "lat_scaled", "lon_scaled", "traffic_volume"]]
df_model.to_csv('model_data.csv', index=False)