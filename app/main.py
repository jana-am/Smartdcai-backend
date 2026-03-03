from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from typing import List
from app.services.predictors import predict_city, get_supported_cities
from app.services.upload_pipeline import predict_from_uploaded_csvs

app = FastAPI(
    title="SmartDCAI API",
    description="AI-Driven Smart Data Center Decision System",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
    except Exception as e:
        # Show the real error so we can debug
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/predict/upload")
async def predict_upload(
    files: List[UploadFile] = File(...)
):
    """
    Upload one or more raw NSRDB CSV files.
    Each file = one year of hourly solar data.
    More years = better accuracy.
    """
    try:
        return await predict_from_uploaded_csvs(files)
    except HTTPException:
        raise  # re-raise our own clean errors
    except Exception as e:
        # Show the real error message instead of hiding it
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")


# ── Fix Swagger UI to show a real file picker ────────────────────────────────
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    upload_path = "/predict/upload"
    if upload_path in schema.get("paths", {}):
        post = schema["paths"][upload_path].get("post", {})
        content = post.get("requestBody", {}).get("content", {})
        if "multipart/form-data" in content:
            content["multipart/form-data"]["schema"] = {
                "type": "object",
                "properties": {
                    "files": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "format": "binary"
                        }
                    }
                },
                "required": ["files"]
            }

    app.openapi_schema = schema
    return app.openapi_schema


app.openapi = custom_openapi
