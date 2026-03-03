from fastapi import FastAPI, HTTPException, UploadFile, File
from typing import List

from app.services.predictors import predict_city, get_supported_cities
from app.services.upload_pipeline import predict_from_uploaded_csvs

app = FastAPI(
    title="SmartDCAI API",
    description="AI-Driven Smart Data Center Decision System",
    version="1.0"
)

@app.get("/")
def home():
    return {"message": "SmartDCAI API is running"}


@app.get("/cities")
def cities():
    return {"supported_cities": get_supported_cities()}


@app.get("/predict/{city_name}")
def predict(city_name: str):
    try:
        return predict_city(city_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")


from fastapi import UploadFile, File
from typing import List

@app.post("/predict/upload")
async def predict_upload(
    files: List[UploadFile] = File(...)
):
    return await predict_from_uploaded_csvs(files)
