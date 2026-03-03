import json
from pathlib import Path
from io import BytesIO
from typing import List

import numpy as np
import pandas as pd
from fastapi import UploadFile, HTTPException
from sklearn.preprocessing import StandardScaler
import joblib

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


# =========================
# CONFIG
# =========================
BUILD_THRESHOLD = 409
LOOKBACK = 12
FORECAST_MONTHS = 36
LSTM_EPOCHS = 8
LSTM_BATCH = 16
PATIENCE = 3

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "xgb_model.pkl"
FEATURES_PATH = BASE_DIR / "features.json"

XGB_MODEL = joblib.load(MODEL_PATH)

with open(FEATURES_PATH, "r") as f:
    UNIVERSAL_FEATURES = json.load(f)


# =========================
# HELPERS
# =========================
def create_sequences(arr, lookback):
    X, y = [], []
    for i in range(len(arr) - lookback):
        X.append(arr[i:i+lookback])
        y.append(arr[i+lookback])
    return np.array(X), np.array(y)


def build_lstm():
    model = Sequential([
        LSTM(16, activation='tanh', input_shape=(LOOKBACK, 1)),
        Dropout(0.2),
        Dense(8, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(0.001), loss='mse')
    return model


# =========================
# MAIN FUNCTION
# =========================
async def predict_from_uploaded_csvs(files: List[UploadFile]):

    if len(files) == 0:
        raise HTTPException(status_code=400, detail="No files uploaded")

    all_dfs = []

    # -------- 1) READ ALL FILES --------
    for file in files:
        if not file.filename.lower().endswith(".csv"):
            raise HTTPException(status_code=400, detail=f"{file.filename} is not CSV")

        content = await file.read()
        df = pd.read_csv(BytesIO(content), skiprows=2)
        df.columns = [str(c).strip() for c in df.columns]
        all_dfs.append(df)

    # -------- 2) MERGE YEARS --------
    df = pd.concat(all_dfs, ignore_index=True)

    if "GHI" not in df.columns:
        raise HTTPException(status_code=400, detail="Column 'GHI' missing")

    # -------- 3) CLEANING --------
    df = df[df["GHI"] > 0].copy()
    df.drop(columns=["Wind Speed", "Wind Direction"], errors="ignore", inplace=True)

    present_feats = [f for f in UNIVERSAL_FEATURES if f in df.columns]

    if len(present_feats) < 6:
        raise HTTPException(status_code=400, detail="Not enough required features in dataset")

    keep_cols = ["Year", "Month", "GHI"] + present_feats
    df = df[keep_cols].dropna()

    if len(df) < 200:
        raise HTTPException(status_code=400, detail="Not enough valid rows")

    # -------- 4) MONTHLY AGGREGATION --------
    monthly = (
        df.groupby(["Year", "Month"], as_index=False)
        .mean(numeric_only=True)
        .sort_values(["Year", "Month"])
        .reset_index(drop=True)
    )

    if len(monthly) < 24:
        raise HTTPException(status_code=400, detail="Not enough monthly rows for forecasting")

    # -------- 5) NORMALIZATION --------
    split_idx = int(len(monthly) * 0.8)
    train = monthly.iloc[:split_idx].copy()
    test = monthly.iloc[split_idx:].copy()

    norm_cols = ["GHI"] + present_feats

    scaler = StandardScaler()
    scaler.fit(train[norm_cols])

    train[norm_cols] = scaler.transform(train[norm_cols])
    test[norm_cols] = scaler.transform(test[norm_cols])

    monthly_norm = pd.concat([train, test], ignore_index=True)

    ghi_mean = float(scaler.mean_[0])
    ghi_std = float(scaler.scale_[0])

    # -------- 6) FORECAST FEATURES --------
    future = pd.DataFrame({
        "Year": [2025 + (i // 12) for i in range(FORECAST_MONTHS)],
        "Month": [(i % 12) + 1 for i in range(FORECAST_MONTHS)]
    })

    for feat in present_feats:

        data = monthly_norm[[feat]].values.astype(np.float32)

        X_seq, y_seq = create_sequences(data, LOOKBACK)

        if len(X_seq) < 20:
            raise HTTPException(status_code=400, detail=f"Not enough data for feature {feat}")

        split = int(len(X_seq) * 0.8)
        Xtr, ytr = X_seq[:split], y_seq[:split]
        Xval, yval = X_seq[split:], y_seq[split:]

        model = build_lstm()
        early = EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)

        model.fit(
            Xtr, ytr,
            validation_data=(Xval, yval),
            epochs=LSTM_EPOCHS,
            batch_size=LSTM_BATCH,
            verbose=0,
            callbacks=[early]
        )

        preds = []
        last_seq = data[-LOOKBACK:].reshape(1, LOOKBACK, 1)

        for _ in range(FORECAST_MONTHS):
            nxt = model.predict(last_seq, verbose=0)[0, 0]
            preds.append(nxt)
            last_seq = np.append(last_seq[:, 1:, :], [[[nxt]]], axis=1)

        future[feat] = preds

        del model
        tf.keras.backend.clear_session()

    # -------- 7) XGBOOST PREDICTION --------
    missing_model_feats = [f for f in UNIVERSAL_FEATURES if f not in future.columns]

    if missing_model_feats:
        raise HTTPException(status_code=400, detail=f"Missing features for model: {missing_model_feats}")

    X_future = future[UNIVERSAL_FEATURES].values
    pred_norm = XGB_MODEL.predict(X_future)

    pred_real = pred_norm * ghi_std + ghi_mean
    future["Predicted_GHI_Wm2"] = pred_real

    yearly_avg = future.groupby("Year")["Predicted_GHI_Wm2"].mean().round(2).to_dict()
    avg_3y = round(float(np.mean(list(yearly_avg.values()))), 2)

    decision = "BUILD" if avg_3y >= BUILD_THRESHOLD else "DON'T BUILD"

    return {
        "yearly_avg": yearly_avg,
        "average_3_year": avg_3y,
        "threshold": BUILD_THRESHOLD,
        "decision": decision
    }
