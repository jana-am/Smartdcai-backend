import json
import traceback
import warnings
from pathlib import Path
from io import BytesIO
from typing import List

import numpy as np
import pandas as pd
from fastapi import UploadFile, HTTPException
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

# =========================
# CONFIG
# =========================
BUILD_THRESHOLD = 409
FORECAST_MONTHS = 36

BASE_DIR = Path(__file__).resolve().parents[2]
FEATURES_PATH = BASE_DIR / "features.json"

XGB_MODEL = xgb.XGBRegressor()
XGB_MODEL.load_model(str(BASE_DIR / "solar_ghi_predictor.json"))

with open(FEATURES_PATH, "r") as f:
    UNIVERSAL_FEATURES = json.load(f)


# =========================
# SARIMA FORECAST
# =========================
def sarima_forecast(series: np.ndarray, steps: int = 36) -> np.ndarray:
    try:
        model = SARIMAX(
            series,
            order=(1, 0, 1),
            seasonal_order=(1, 0, 1, 12),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        fit = model.fit(disp=False, maxiter=50)
        forecast = fit.forecast(steps=steps)
        return np.array(forecast, dtype=np.float32)
    except Exception:
        cycle = series[-12:] if len(series) >= 12 else series
        repeated = np.tile(cycle, steps // len(cycle) + 1)[:steps]
        return repeated.astype(np.float32)


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
                    detail=f"'{file.filename}' is not a CSV file. Please upload NSRDB CSV files only."
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
            raise HTTPException(
                status_code=400,
                detail="Column 'GHI' is missing. Make sure you are uploading a standard NSRDB CSV file."
            )

        # -------- 3) CLEANING --------
        print("[3] Cleaning...")
        df = df[df["GHI"] > 0].copy()
        df.drop(columns=["Wind Speed", "Wind Direction"], errors="ignore", inplace=True)

        present_feats = [f for f in UNIVERSAL_FEATURES if f in df.columns]
        missing_feats = [f for f in UNIVERSAL_FEATURES if f not in df.columns]
        print(f"[3] Present features: {present_feats}")
        print(f"[3] Missing features (will use 0): {missing_feats}")

        if len(present_feats) < 6:
            raise HTTPException(
                status_code=400,
                detail=f"Only {len(present_feats)} required features found. "
                       f"Required at least 6 of: {UNIVERSAL_FEATURES}"
            )

        keep_cols = ["Year", "Month", "GHI"] + present_feats
        df = df[keep_cols].dropna()
        print(f"[3] After cleaning: {len(df)} rows")

        if len(df) < 200:
            raise HTTPException(
                status_code=400,
                detail=f"Only {len(df)} valid rows found. "
                       "Please upload at least one full year of hourly NSRDB data."
            )

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
            raise HTTPException(
                status_code=400,
                detail=f"Only {len(monthly)} monthly records found. "
                       "Please upload at least 2 years of data for reliable forecasting."
            )

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

        # -------- 6) SARIMA FEATURE FORECAST --------
        print("[6] SARIMA forecasting features...")
        future = pd.DataFrame({
            "Year":  [2025 + (i // 12) for i in range(FORECAST_MONTHS)],
            "Month": [(i % 12) + 1     for i in range(FORECAST_MONTHS)]
        })

        # Forecast features that are present
        for feat in present_feats:
            print(f"[6] Forecasting: {feat}")
            series = monthly_norm[feat].values.astype(np.float32)
            future[feat] = sarima_forecast(series, steps=FORECAST_MONTHS)

        # Fill missing features with 0 (neutral normalized value)
        for feat in missing_feats:
            print(f"[6] Filling missing feature with 0: {feat}")
            future[feat] = 0.0

        print("[6] SARIMA forecasting complete")

        # -------- 7) XGBOOST PREDICTION --------
        print("[7] XGBoost prediction...")
        X_future  = future[UNIVERSAL_FEATURES].values
        pred_norm = XGB_MODEL.predict(X_future)
        pred_real = pred_norm * ghi_std + ghi_mean

        future["Predicted_GHI_Wm2"] = pred_real

        yearly_avg = (
            future.groupby("Year")["Predicted_GHI_Wm2"]
            .mean()
            .round(2)
            .to_dict()
        )
        avg_3y   = round(float(np.mean(list(yearly_avg.values()))), 2)
        decision = "BUILD" if avg_3y >= BUILD_THRESHOLD else "DON'T BUILD"

        print(f"[7] Decision: {decision}, avg_3y={avg_3y}")

        return {
            "yearly_avg":     {int(k): v for k, v in yearly_avg.items()},
            "average_3_year": avg_3y,
            "threshold":      BUILD_THRESHOLD,
            "decision":       decision,
            "files_uploaded": len(files),
            "months_of_data": len(monthly),
            "missing_features": missing_feats  # inform the frontend which were missing
        }

    except HTTPException:
        raise
    except Exception as e:
        print("=== FULL ERROR TRACEBACK ===")
        traceback.print_exc()
        print("=== END TRACEBACK ===")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")
