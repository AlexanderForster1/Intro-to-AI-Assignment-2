import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('Scats Data October 2006.csv', dtype={"SCATS Number": str})

df.columns = df.columns.str.strip()
df["SCATS Number"] = df["SCATS Number"].str.zfill(4)

# Save map data
df_map = df[["SCATS Number", "Location", "NB_LATITUDE", "NB_LONGITUDE"]].copy()
df_map = df_map.drop_duplicates(subset=["SCATS Number"])
df_map.to_csv("map_data.csv", index=False)

coord_scaler = StandardScaler()
df[["lat_scaled", "lon_scaled"]] = coord_scaler.fit_transform(df[["NB_LATITUDE", "NB_LONGITUDE"]])
df_temp = df.drop(columns=["Location", "NB_LATITUDE", "NB_LONGITUDE", "CD_MELWAY", "HF VicRoads Internal", "VR Internal Stat", "VR Internal Loc", "NB_TYPE_SURVEY"])
time_cols = [col for col in df_temp.columns if ":" in col]
df_model = df_temp.melt(
    id_vars=["SCATS Number", "Date", "lat_scaled", "lon_scaled"],
    var_name="time",
    value_vars=time_cols,
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
df_model["hour"] = df_model["datetime"].dt.hour
df_model = df_model.sort_values(by=["Date", "SCATS Number", "time_index"]).reset_index(drop=True)
df_hourly = (
    df_model.groupby(["SCATS Number", "Date", "hour"], as_index=False)
    .agg({
        "traffic_volume": "sum",
        "lat_scaled": "mean",
        "lon_scaled": "mean",
        "day_sin": "first",
        "day_cos": "first",
        "is_weekend": "first"
    })
)
df_hourly["hour_sin"] = np.sin(2 * np.pi * df_hourly["hour"] / 24)
df_hourly["hour_cos"] = np.cos(2 * np.pi * df_hourly["hour"] / 24)

df_id = df_hourly[["SCATS Number", "lat_scaled", "lon_scaled"]].copy()
df_id = df_id.drop_duplicates(subset=["SCATS Number"])
df_id.to_csv('id_mapping.csv', index=False)


df_hourly = df_hourly[["SCATS Number", "Date", "hour", "hour_sin", "hour_cos", "day_sin", "day_cos", "is_weekend", "lat_scaled", "lon_scaled", "traffic_volume"]]
df_hourly.to_csv('model_data.csv', index=False)