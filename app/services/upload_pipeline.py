# app/services/upload_pipeline.py
# Custom-city upload pipeline (Phases 1–3 + Phase 4 with saved XGBoost)
# - Reads raw NSRDB CSV (skiprows=2)
# - Cleans (remove GHI=0, drop wind cols, keep universal 10 features)
# - Monthly aggregation + train/test split BEFORE normalization (no leakage)
# - Forecasts 10 features for 36 months (2025–2027) using lightweight LSTM per feature
# - Predicts normalized GHI using saved xgb_model.pkl, then denormalizes using GHI mean/std
# - Returns yearly averages + decision

import json
from pathlib import Path
from io import BytesIO

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

# API-safe training defaults (fast / low resource)
LSTM_EPOCHS = 8
LSTM_BATCH = 16
LSTM_UNITS = 16
PATIENCE = 3

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "xgb_model.pkl"
FEATURES_PATH = BASE_DIR / "features.json"

# Load model + features once
try:
    XGB_MODEL = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load XGBoost model at {MODEL_PATH}: {e}")

with open(FEATURES_PATH, "r") as f:
    UNIVERSAL_FEATURES = json.load(f)


# =========================
# HELPERS
# =========================
def _create_sequences(arr: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(len(arr) - lookback):
        X.append(arr[i : i + lookback])
        y.append(arr[i + lookback])
    return np.array(X), np.array(y)


def _build_lstm(lookback: int) -> tf.keras.Model:
    model = Sequential(
        [
            LSTM(LSTM_UNITS, activation="tanh", return_sequences=False, input_shape=(lookback, 1)),
            Dropout(0.2),
            Dense(8, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    return model


def _next_year_month(year: int, month: int) -> tuple[int, int]:
    month += 1
    if month == 13:
        year += 1
        month = 1
    return year, month


# =========================
# MAIN ENTRY
# =========================
async def predict_from_uploaded_csv(file: UploadFile):
    # -------- 0) Basic checks --------
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file")

    # -------- 1) Read CSV --------
    try:
        content = await file.read()
        df = pd.read_csv(BytesIO(content), skiprows=2)
        df.columns = [str(c).strip() for c in df.columns]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {str(e)}")

    required = {"Year", "Month", "GHI"}
    if not required.issubset(df.columns):
        missing = list(required - set(df.columns))
        raise HTTPException(status_code=400, detail=f"Missing required columns: {missing}")

    # -------- 2) Phase 1: Cleaning --------
    # Remove nighttime
    df = df[df["GHI"] > 0].copy()

    # Drop wind cols if exist
    df.drop(columns=["Wind Speed", "Wind Direction"], errors="ignore", inplace=True)

    # Keep only needed columns
    present_feats = [c for c in UNIVERSAL_FEATURES if c in df.columns]
    if len(present_feats) < 6:
        raise HTTPException(
            status_code=400,
            detail=f"Uploaded dataset missing many universal features. Found only: {present_feats}",
        )

    keep_cols = ["Year", "Month", "GHI"] + present_feats
    df = df[keep_cols].dropna().copy()

    if len(df) < 200:
        raise HTTPException(status_code=400, detail="Not enough valid rows after cleaning (need >= 200).")

    # Ensure numeric
    for c in ["Year", "Month", "GHI"] + present_feats:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Year", "Month", "GHI"] + present_feats)

    # -------- 3) Phase 2: Monthly aggregation + normalization --------
    monthly = (
        df.groupby(["Year", "Month"], as_index=False)
        .mean(numeric_only=True)
        .sort_values(["Year", "Month"])
        .reset_index(drop=True)
    )

    if len(monthly) < (LOOKBACK + 12):
        raise HTTPException(
            status_code=400,
            detail=f"Not enough monthly rows for forecasting. Need at least {LOOKBACK + 12} months.",
        )

    # Train/test split BEFORE scaler fit (no leakage)
    split_idx = int(len(monthly) * 0.8)
    train = monthly.iloc[:split_idx].copy()
    test = monthly.iloc[split_idx:].copy()

    # IMPORTANT: GHI must be first
    norm_cols = ["GHI"] + present_feats

    scaler = StandardScaler()
    scaler.fit(train[norm_cols])

    train[norm_cols] = scaler.transform(train[norm_cols])
    test[norm_cols] = scaler.transform(test[norm_cols])

    monthly_norm = pd.concat([train, test], ignore_index=True)

    ghi_mean = float(scaler.mean_[0])
    ghi_std = float(scaler.scale_[0])

    # -------- 4) Phase 3: Forecast 10 features (or available subset) 2025–2027 --------
    # Create Year/Month for 36 months starting Jan 2025
    future = pd.DataFrame(
        {
            "Year": [2025 + (i // 12) for i in range(FORECAST_MONTHS)],
            "Month": [(i % 12) + 1 for i in range(FORECAST_MONTHS)],
        }
    )

    # Forecast each feature independently (on normalized scale)
    for feat in present_feats:
        series = monthly_norm[[feat]].values.astype(np.float32)

        # Build sequences
        X_seq, y_seq = _create_sequences(series, LOOKBACK)
        if len(X_seq) < 20:
            raise HTTPException(
                status_code=400,
                detail=f"Not enough sequences to train LSTM for feature '{feat}'.",
            )

        # Split sequences chronologically
        s2 = int(len(X_seq) * 0.8)
        Xtr, ytr = X_seq[:s2], y_seq[:s2]
        Xva, yva = X_seq[s2:], y_seq[s2:]

        # Build & train model
        model = _build_lstm(LOOKBACK)
        early = EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)

        model.fit(
            Xtr,
            ytr,
            validation_data=(Xva, yva),
            epochs=LSTM_EPOCHS,
            batch_size=LSTM_BATCH,
            verbose=0,
            callbacks=[early],
        )

        # Recursive forecast
        preds = []
        last_seq = series[-LOOKBACK:].reshape(1, LOOKBACK, 1)
        for _ in range(FORECAST_MONTHS):
            nxt = float(model.predict(last_seq, verbose=0)[0, 0])
            preds.append(nxt)
            last_seq = np.append(last_seq[:, 1:, :], [[[nxt]]], axis=1)

        future[feat] = preds

        # Cleanup TF memory
        del model
        tf.keras.backend.clear_session()

    # -------- 5) Phase 4: XGBoost GHI prediction (using saved model) --------
    # Ensure correct feature order for XGB input:
    # Your saved features.json is the universal order used in training
    # But uploaded dataset might miss some columns -> must fail (model expects fixed feature count)
    missing_for_model = [f for f in UNIVERSAL_FEATURES if f not in future.columns]
    if missing_for_model:
        raise HTTPException(
            status_code=400,
            detail=f"Uploaded dataset missing required model features: {missing_for_model}",
        )

    X_future = future[UNIVERSAL_FEATURES].values
    pred_norm = XGB_MODEL.predict(X_future)

    # Denormalize
    pred_real = pred_norm * ghi_std + ghi_mean
    future["Predicted_GHI_Wm2"] = pred_real

    yearly_avg = future.groupby("Year")["Predicted_GHI_Wm2"].mean().round(2).to_dict()
    avg_3y = round(float(np.mean(list(yearly_avg.values()))), 2)
    decision = "BUILD" if avg_3y >= BUILD_THRESHOLD else "DON'T BUILD"

    return {
        "message": "Custom dataset processed successfully (Phases 1–3 + XGBoost prediction).",
        "yearly_avg": yearly_avg,
        "average_3_year": avg_3y,
        "threshold": BUILD_THRESHOLD,
        "decision": decision,
        "notes": {
            "features_used": UNIVERSAL_FEATURES,
            "features_present_in_upload": present_feats,
            "lstm_epochs": LSTM_EPOCHS,
            "lookback_months": LOOKBACK,
        },
    }

from typing import List

async def predict_from_uploaded_csvs(files: List[UploadFile]):

    if len(files) == 0:
        raise HTTPException(status_code=400, detail="No files uploaded")

    all_dfs = []

    for file in files:
        if not file.filename.lower().endswith(".csv"):
            raise HTTPException(status_code=400, detail=f"{file.filename} is not a CSV")

        content = await file.read()
        df = pd.read_csv(BytesIO(content), skiprows=2)
        df.columns = [str(c).strip() for c in df.columns]

        all_dfs.append(df)

    # 🔥 Merge all years into one dataset
    df = pd.concat(all_dfs, ignore_index=True)

    # Now continue using your existing pipeline logic
    return await process_combined_dataframe(df)
