import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from pathlib import Path
data_dir = Path(__file__).parent.parent.parent / "data" / "updated_data.csv"

df = pd.read_csv(data_dir, dtype={"SCATS Number": str})
df.columns = df.columns.str.strip()
df["SCATS Number"] = df["SCATS Number"].str.zfill(4)
df = df.sort_values(["Date", "SCATS Number", "hour"]).reset_index(drop=True)
num_intersections = df["SCATS Number"].nunique()
df = pd.get_dummies(df, columns=["SCATS Number"], prefix="SCATS")
feature_cols = [
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
    "month_cos",
    "month_sin",
    "is_weekend",
    "lat_scaled",
    "lon_scaled"
] + [col for col in df.columns if col.startswith("SCATS_")]

target_col = "traffic_volume"

sequence_length = 24

X = []
y = []

num_sequences = len(df) // sequence_length
for i in range(num_sequences):
    start = i * sequence_length
    end = start + sequence_length
    sequence = df.iloc[start:end]
    X.append(sequence[feature_cols].values)
    y.append(sequence[[target_col]].values)
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

print("X shape:", X.shape)
print("y shape:", y.shape)

num_time_steps = X.shape[1]
num_features = X.shape[2]
def build_rnn_model(Number_of_time_steps, num_features):
    rnn_model = models.Sequential([
        layers.Input(shape=(Number_of_time_steps, num_features)),
        layers.SimpleRNN(128, return_sequences=True),
        layers.LayerNormalization(),
        layers.Dropout(0.2),
        layers.TimeDistributed(layers.Dense(64, activation='relu')),
        layers.TimeDistributed(layers.Dense(32, activation='relu')),
        layers.TimeDistributed(layers.Dense(1, activation='relu')),
    ])
    optimizer = keras.optimizers.Adam(learning_rate=0.0005, clipnorm=1.0)
    rnn_model.compile(optimizer=optimizer, loss=keras.losses.Huber(delta=50.0), metrics=['mae'])
    return rnn_model
spilt_index = int(0.8 * len(X))
seq_per_day = num_intersections
num_days = len(X) // seq_per_day
train_days = int(0.8 * num_days)
train_seqs = train_days * seq_per_day
x_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.15, random_state=42, shuffle=True)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_results = []
for fold, (train_index, val_index) in enumerate(kf.split(x_train_full), start=1):
    print(f"\nTraining fold {fold}...")
    X_train = x_train_full[train_index]
    y_train = y_train_full[train_index]
    X_val = x_train_full[val_index]
    y_val = y_train_full[val_index]
    # Create a fresh model for each fold
    model = build_rnn_model(num_time_steps, num_features)
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=[keras.callbacks.EarlyStopping(monitor='val_mae', patience=8, min_delta=0.05, restore_best_weights=True, verbose=1)],
        verbose=1
    )
    y_pred = model.predict(X_val, verbose=0)
    # Flatten because y is shaped like (samples, 96, 1)
    y_val_flat = y_val.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)

    fold_r2 = r2_score(y_val_flat, y_pred_flat)
    fold_mse = mean_squared_error(y_val_flat, y_pred_flat)
    fold_rmse = np.sqrt(fold_mse)
    fold_mae = mean_absolute_error(y_val_flat, y_pred_flat)

    fold_results.append({
        "Fold": fold,
        "R2": fold_r2,
        "MSE": fold_mse,
        "RMSE": fold_rmse,
        "MAE": fold_mae
    })

    print(f"Fold {fold} R2: {fold_r2:.4f}")
    print(f"Fold {fold} MAE: {fold_mae:.4f}")
    print(f"Fold {fold} RMSE: {fold_rmse:.4f}")

results_df = pd.DataFrame(fold_results)

print("\nCross-validation results:")
print(results_df)
print("\nMean CV results:")
print(results_df[["R2", "MSE", "RMSE", "MAE"]].mean())
print("\nStandard deviation:")
print(results_df[["R2", "MSE", "RMSE", "MAE"]].std())

final_model = build_rnn_model(num_time_steps, num_features)
final_history = final_model.fit(x_train_full, y_train_full, validation_split=0.15, epochs=50, batch_size=32, callbacks=[keras.callbacks.EarlyStopping(monitor='val_mae', patience=8, min_delta=0.05, restore_best_weights=True, verbose=1)], verbose=1)

y_final_pred = final_model.predict(X_test)
y_test_flat = y_test.reshape(-1)
y_final_pred_flat = y_final_pred.reshape(-1)
final_mse = mean_squared_error(y_test_flat, y_final_pred_flat)
final_mae = mean_absolute_error(y_test_flat, y_final_pred_flat)
final_rmse = np.sqrt(final_mse)
normalised_mae = final_mae / np.mean(y_test_flat)
print("Normalised MAE:", normalised_mae)
print("MAE as % of mean traffic:", normalised_mae * 100)
print(f"\nFinal Model - MSE: {final_mse:.4f}, MAE: {final_mae:.4f}, RMSE: {final_rmse:.4f}, MAPE: {normalised_mae:.4f}, R2: {r2_score(y_test_flat, y_final_pred_flat):.4f}, explained variance: {r2_score(y_test_flat, y_final_pred_flat, multioutput='variance_weighted'):.4f}")
plt.figure(figsize=(8, 5))
plt.plot(final_history.history["loss"], label="Training Loss")
plt.plot(final_history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Huber Loss")
plt.title("RNN Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(final_history.history["mae"], label="Training MAE")
plt.plot(final_history.history["val_mae"], label="Validation MAE")
plt.xlabel("Epoch")
plt.ylabel("Mean Absolute Error")
plt.title("RNN Training and Validation MAE")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(6, 6))
plt.scatter(y_test_flat, y_final_pred_flat, alpha=0.3)

min_val = min(y_test_flat.min(), y_final_pred_flat.min())
max_val = max(y_test_flat.max(), y_final_pred_flat.max())

plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

plt.xlabel("Actual Traffic Volume")
plt.ylabel("Predicted Traffic Volume")
plt.title("Actual vs Predicted Traffic Volume")
plt.grid(True)
plt.show()

sample_index = 0

actual = y_test[sample_index].reshape(-1)
predicted = y_final_pred[sample_index].reshape(-1)

plt.figure(figsize=(8, 5))
plt.plot(actual, label="Actual")
plt.plot(predicted, label="Predicted")
plt.title("RNN Actual vs Predicted Traffic Volume")
plt.xlabel("Time Step")
plt.ylabel("Traffic Volume")
plt.legend()
plt.grid(True)
plt.show()

final_model.save('updated_rnn_traffic_model.keras')