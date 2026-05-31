from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

target_folder = Path("Updated Data")
df_id_mapping = pd.read_csv('id_mapping.csv', dtype={"SCATS Number": str})
df_id_mapping.columns = df_id_mapping.columns.str.strip()
df_id_mapping["SCATS Number"] = df_id_mapping["SCATS Number"].str.zfill(4)
list_SCATS_numbers = df_id_mapping["SCATS Number"].tolist()

def v_col_to_time(col):
    if col.startswith("V") and col[1:].isdigit():
        i = int(col[1:])
        hour = i // 4
        minute = (i % 4) * 15
        return f"{hour:02d}:{minute:02d}"
    return col

dfs = []
for path in target_folder.rglob("*.csv"):
    print(f"Processing file: {path}")
    df = pd.read_csv(path, dtype={"NB_SCATS_SITE": str})
    df.columns = df.columns.str.strip()
    df["NB_SCATS_SITE"] = df["NB_SCATS_SITE"].str.zfill(4)
    df_filtered = df[df["NB_SCATS_SITE"].isin(list_SCATS_numbers)].copy()
    df_filtered = df_filtered.rename(columns={
        "NB_SCATS_SITE": "SCATS Number",
        "QT_INTERVAL_COUNT": "Date"
    })
    df_filtered = df_filtered.drop(columns=["CT_ALARM_24HOUR", "QT_VOLUME_24HOUR", "CT_RECORDS", "NM_REGION", "NB_DETECTOR"], errors='ignore')
    df_filtered = df_filtered.rename(columns=v_col_to_time)
    df_filtered = df_filtered.merge(df_id_mapping[["SCATS Number", "lon_scaled", "lat_scaled"]], on="SCATS Number", how="left")
    dfs.append(df_filtered)
    print(f"Finished processing file: {path}")

complete_df = pd.concat(dfs, ignore_index=True)
time_cols = [col for col in complete_df.columns if ":" in col]

df_model = complete_df.melt(
    id_vars=["SCATS Number", "Date", "lat_scaled", "lon_scaled"],
    var_name="time",
    value_vars=time_cols,
    value_name="traffic_volume"
)

df_model["Date"] = pd.to_datetime(df_model["Date"])
df_model["Day"] = df_model["Date"].dt.dayofweek
df_model["Month"] = df_model["Date"].dt.month

df_model["datetime"] = pd.to_datetime(df_model["Date"].astype(str) + ' ' + df_model["time"].astype(str))

df_model["minutes_since_midnight"] = (df_model["datetime"].dt.hour * 60 + df_model["datetime"].dt.minute)
df_model["time_index"] = df_model["minutes_since_midnight"] // 15
df_model["time_sin"] = np.sin(2 * np.pi * df_model["time_index"] / 96)
df_model["time_cos"] = np.cos(2 * np.pi * df_model["time_index"] / 96)
df_model["day_sin"] = np.sin(2 * np.pi * df_model["Day"] / 7)
df_model["day_cos"] = np.cos(2 * np.pi * df_model["Day"] / 7)
df_model["month_sin"] = np.sin(2 * np.pi * df_model["Month"] / 12)
df_model["month_cos"] = np.cos(2 * np.pi * df_model["Month"] / 12)

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
        "month_sin": "first",
        "month_cos": "first",
        "is_weekend": "first"
    })
)
df_hourly["hour_sin"] = np.sin(2 * np.pi * df_hourly["hour"] / 24)
df_hourly["hour_cos"] = np.cos(2 * np.pi * df_hourly["hour"] / 24)

df_hourly = df_hourly[["SCATS Number", "Date", "hour", "hour_sin", "hour_cos", "day_sin", "day_cos", "month_sin", "month_cos", "is_weekend", "lat_scaled", "lon_scaled", "traffic_volume"]]
df_hourly.to_csv('updated_data.csv', index=False)
print("Data processing complete. Updated data saved to 'updated_data.csv'.")
