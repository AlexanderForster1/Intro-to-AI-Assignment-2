import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Input
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
import random

np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

df_model = pd.read_csv('Assignment 2B/data/model_data.csv')

print(df_model)

window_size = 24 # num of rows in a day
X = []
y = []

target = ['traffic_volume']

feature_cols = ['hour_sin', 'hour_cos','day_sin', 'day_cos',
                'is_weekend', 'lat_scaled', 'lon_scaled']

df_model = df_model.sort_values(["SCATS Number", "Date", "hour"])

train_cutoff = '2006-10-25'
train_rows = df_model[df_model['Date'] < '2006-10-18']

scaler_y = MinMaxScaler(feature_range=(0, 1))
scaler_y.fit(train_rows[target])

X, y, dates = [], [], []

for scats_num, group in df_model.groupby("SCATS Number"):
    group = group.reset_index(drop=True)
    feature_values = group[feature_cols].values
    target_values = scaler_y.transform(group[target]).flatten()
    group_dates = group['Date'].values

    for i in range(window_size, len(group)):
        X.append(feature_values[i - window_size:i])
        y.append(target_values[i])
        dates.append(group_dates[i])

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)
dates = np.array(dates)

train_mask = dates < '2006-10-18'                                # Oct 01 - Oct 17
validate_mask   = (dates >= '2006-10-18') & (dates < '2006-10-25')    # Oct 18 - Oct 24
test_mask  = dates >= '2006-10-25'                               # Oct 25 - Oct 31

X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val     = X[validate_mask], y[validate_mask]
X_test, y_test   = X[test_mask], y[test_mask]

# Shuffle SCATS sites for validation
shuffle_idx = np.random.permutation(len(X_train))
X_train = X_train[shuffle_idx]
y_train = y_train[shuffle_idx]

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

def get_lstm(units):
    """LSTM(Long Short-Term Memory)
    Build LSTM Model.

    # Arguments
        units: List(int), number of input, output and hidden units.
    # Returns
        model: Model, nn model.
    """

    model = Sequential()
    model.add(Input(shape=(X_train.shape[1], X_train.shape[2]))) # Scale up layers to 128/64 if underfitting
    model.add(LSTM(units[0], return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units[1]))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(units[2], activation='linear'))

    model.compile(optimizer=Adam(learning_rate=0.0005), loss='huber', metrics=['mae'])

    es = EarlyStopping(
    monitor='val_loss',
    patience=15,              # 15 epochs of no improvement
    restore_best_weights=True, # restore best epoch's weights
    )

    history = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_val, y_val), callbacks=[es])

    predictions = model.predict(X_test)
    predictions = scaler_y.inverse_transform(predictions).flatten()
    y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1,1)).flatten()

    mse  = mean_squared_error(y_test_actual, predictions)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_test_actual, predictions)
    r2   = r2_score(y_test_actual, predictions)
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R2 Score: {r2:.4f}")

    # Training History
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history['loss'], label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_title('Model Loss (Huber Loss)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    axes[1].plot(history.history['mae'], label='Train MAE')
    axes[1].plot(history.history['val_mae'], label='Val MAE')
    axes[1].set_title('Model MAE')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].legend()

    plt.suptitle('LSTM Training History', fontsize=14)
    plt.tight_layout()
    plt.show()

    # Plot Actual vs Predicted Line Graph
    samples_per_day = 24
    days_to_plot = 14
    plot_slice = slice(0, samples_per_day * days_to_plot)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(y_test_actual[plot_slice], label='Actual', alpha=0.7)
    ax.plot(predictions[plot_slice], label='Predicted', alpha=0.7)
    ax.set_title(f'First 2 Weeks — Actual vs Predicted')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Traffic Volume')
    ax.legend()
    plt.tight_layout()
    plt.show()

    # Scatter Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_test_actual, predictions, alpha=0.4)

    min_val = min(y_test_actual.min(), predictions.min())
    max_val = max(y_test_actual.max(), predictions.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect prediction')

    ax.set_title('Actual vs Predicted — Scatter')
    ax.set_xlabel('Actual Traffic Volume')
    ax.set_ylabel('Predicted Traffic Volume')
    ax.legend()
    plt.tight_layout()
    plt.show()

    return model

model = get_lstm([64, 32, 1])