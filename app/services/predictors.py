import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

BUILD_THRESHOLD = 409

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = BASE_DIR / "xgb_model.pkl"
FEATURES_PATH = BASE_DIR / "features.json"
DATASETS_DIR = BASE_DIR / "datasets"
GHI_PARAMS_DIR = BASE_DIR / "ghi_params"
CITIES_PATH = BASE_DIR / "cities.csv"

# Load model once at startup
model = joblib.load(MODEL_PATH)

# Load feature list
with open(FEATURES_PATH, "r") as f:
    FEATURES = json.load(f)

# Load supported cities
cities_df = pd.read_csv(CITIES_PATH)
SUPPORTED_CITIES = cities_df["city_name"].tolist()


def get_supported_cities():
    return SUPPORTED_CITIES


def predict_city(city_name: str):
    city_name = city_name.strip()

    if city_name not in SUPPORTED_CITIES:
        raise ValueError(f"City '{city_name}' is not supported")

    dataset_path = DATASETS_DIR / f"{city_name}_forecast_2025_2027.csv"
    ghi_path = GHI_PARAMS_DIR / f"{city_name}_ghi_params.json"

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found for {city_name}")

    if not ghi_path.exists():
        raise FileNotFoundError(f"GHI params not found for {city_name}")

    df = pd.read_csv(dataset_path)

    # Ensure correct feature order
    X = df[FEATURES].values

    # Predict normalized GHI
    pred_norm = model.predict(X)

    # Load normalization parameters
    with open(ghi_path, "r") as f:
        ghi_params = json.load(f)

    mean = ghi_params["mean"]
    std = ghi_params["scale"]

    # Denormalize
    pred_real = pred_norm * std + mean
    df["Predicted_GHI_Wm2"] = pred_real

    # Yearly averages
    yearly_avg = (
        df.groupby("Year")["Predicted_GHI_Wm2"]
        .mean()
        .round(2)
        .to_dict()
    )

    avg_3y = round(np.mean(list(yearly_avg.values())), 2)

    decision = "BUILD" if avg_3y >= BUILD_THRESHOLD else "DON'T BUILD"

    return {
        "city": city_name,
        "yearly_avg": yearly_avg,
        "average_3_year": avg_3y,
        "threshold": BUILD_THRESHOLD,
        "decision": decision
    }
