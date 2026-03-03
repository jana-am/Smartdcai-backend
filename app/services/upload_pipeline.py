import json
import traceback
from pathlib import Path
from io import BytesIO
from typing import List

import numpy as np
import pandas as pd
from fastapi import UploadFile, HTTPException
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam


# =========================
# CONFIG
# =========================
BUILD_THRESHOLD = 409
LOOKBACK = 12
FORECAST_MONTHS = 36
LSTM_EPOCHS = 8
LSTM_BATCH = 16

BASE_DIR = Path(__file__).resolve().parents[2]
FEATURES_PATH = BASE_DIR / "features.json"

XGB_MODEL = xgb.XGBRegressor()
XGB_MODEL.load_model(str(BASE_DIR / "xgb_model_new.json"))

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
    try:
        if len(files) == 0:
            raise HTTPException(status_code=400, detail="No files uploaded")

        all_dfs = []

        # -------- 1) READ ALL FILES --------
        print(f"[1] Reading {len(files)} files...")
        for file in files:
            if not file.filename.lower().endswith(".csv"):
                raise HTTPException(
                    status_code=400,
                    detail=f"'{file.filename}' is not a CSV file."
                )
            content = await file.read()
            try:
                df = pd.read_csv(BytesIO(content), skiprows=2, encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(BytesIO(content), skiprows=2, encoding="latin-1")
            df.columns = [str(c).strip() for c in df.columns]
            for c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            all_dfs.append(df)
        print(f"[1] Done. Files read: {len(all_dfs)}")

        # -------- 2) MERGE --------
        print("[2] Merging files...")
        df = pd.concat(all_dfs, ignore_index=True)
        print(f"[2] Merged shape: {df.shape}")

        if "GHI" not in df.columns:
            raise HTTPException(status_code=400, detail="Column 'GHI' is missing.")

        # -------- 3) CLEANING --------
        print("[3] Cleaning...")
        df = df[df["GHI"] > 0].copy()
        df.drop(columns=["Wind Speed", "Wind Direction"], errors="ignore", inplace=True)
        present_feats = [f for f in UNIVERSAL_FEATURES if f in df.columns]
        print(f"[3] Present features: {present_feats}")

        if len(present_feats) < 6:
            raise HTTPException(status_code=400, detail=f"Only {len(present_feats)} features found.")

        keep_cols = ["Year", "Month", "GHI"] + present_feats
        df = df[keep_cols].dropna()
        print(f"[3] After cleaning: {len(df)} rows")

        if len(df) < 200:
            raise HTTPException(status_code=400, detail=f"Only {len(df)} valid rows.")

        # -------- 4) MONTHLY AGGREGATION --------
        print("[4] Monthly aggregation...")
        monthly = (
            df.groupby(["Year", "Month"], as_index=False)
            .mean(numeric_only=True)
            .sort_values(["Year", "Month"])
            .reset_index(drop=True)
        )
        print(f"[4] Monthly rows: {len(monthly)}")

        if len(monthly) < 24:
            raise HTTPException(status_code=400, detail=f"Only {len(monthly)} monthly records.")

        # -------- 5) NORMALIZATION --------
        print("[5] Normalizing...")
        split_idx = int(len(monthly) * 0.8)
        train = monthly.iloc[:split_idx].copy()
        test  = monthly.iloc[split_idx:].copy()
        norm_cols = ["GHI"] + present_feats
        scaler = StandardScaler()
        scaler.fit(train[norm_cols])
        train[norm_cols] = scaler.transform(train[norm_cols])
        test[norm_cols]  = scaler.transform(test[norm_cols])
        monthly_norm = pd.concat([train, test], ignore_index=True)
        ghi_mean = float(scaler.mean_[0])
        ghi_std  = float(scaler.scale_[0])
        print(f"[5] GHI mean={ghi_mean:.2f}, std={ghi_std:.2f}")

        # -------- 6) LSTM FORECAST --------
        print("[6] LSTM forecasting...")
        future = pd.DataFrame({
            "Year":  [2025 + (i // 12) for i in range(FORECAST_MONTHS)],
            "Month": [(i % 12) + 1     for i in range(FORECAST_MONTHS)]
        })

        for feat in present_feats:
            print(f"[6] Forecasting feature: {feat}")
            data = monthly_norm[[feat]].values.astype(np.float32)
            X_seq, y_seq = create_sequences(data, LOOKBACK)

            if len(X_seq) < 5:
                future[feat] = float(np.mean(data))
                print(f"[6] {feat}: not enough sequences, using mean")
                continue

            model = build_lstm()
            model.fit(X_seq, y_seq, epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH, verbose=0)

            preds    = []
            last_seq = data[-LOOKBACK:].reshape(1, LOOKBACK, 1)
            for _ in range(FORECAST_MONTHS):
                nxt = model.predict(last_seq, verbose=0)[0, 0]
                preds.append(nxt)
                last_seq = np.append(last_seq[:, 1:, :], [[[nxt]]], axis=1)

            future[feat] = preds
            del model
            tf.keras.backend.clear_session()
            print(f"[6] {feat}: done")

        # -------- 7) XGBOOST --------
        print("[7] XGBoost prediction...")
        missing_model_feats = [f for f in UNIVERSAL_FEATURES if f not in future.columns]
        if missing_model_feats:
            raise HTTPException(status_code=400, detail=f"Missing features: {missing_model_feats}")

        X_future  = future[UNIVERSAL_FEATURES].values
        pred_norm = XGB_MODEL.predict(X_future)
        pred_real = pred_norm * ghi_std + ghi_mean
        future["Predicted_GHI_Wm2"] = pred_real

        yearly_avg = future.groupby("Year")["Predicted_GHI_Wm2"].mean().round(2).to_dict()
        avg_3y     = round(float(np.mean(list(yearly_avg.values()))), 2)
        decision   = "BUILD" if avg_3y >= BUILD_THRESHOLD else "DON'T BUILD"

        print(f"[7] Decision: {decision}, avg_3y={avg_3y}")

        return {
            "yearly_avg":     {int(k): v for k, v in yearly_avg.items()},
            "average_3_year": avg_3y,
            "threshold":      BUILD_THRESHOLD,
            "decision":       decision,
            "files_uploaded": len(files),
            "months_of_data": len(monthly)
        }

    except HTTPException:
        raise
    except Exception as e:
        # Print full traceback to Render logs
        print("=== FULL ERROR TRACEBACK ===")
        traceback.print_exc()
        print("=== END TRACEBACK ===")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")
