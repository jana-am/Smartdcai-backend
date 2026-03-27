from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import List
from app.services.predictors import predict_city, get_supported_cities
from app.services.upload_pipeline import predict_from_uploaded_csvs
from app.database import get_db, create_tables, User
from app.auth import hash_password, verify_password, create_token, get_current_user

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

# Create database tables on startup
@app.on_event("startup")
def startup():
    create_tables()

# Request body model for login and signup
class AuthRequest(BaseModel):
    email: str
    username: str
    password: str

@app.post("/auth/signup")
def signup(data: AuthRequest, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == data.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    user = User(email=data.email, hashed_password=hash_password(data.password))
    db.add(user)
    db.commit()
    return {"message": "Account created successfully"}

@app.post("/auth/login")
def login(data: AuthRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == data.email).first()
    if not user or not verify_password(data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = create_token(user.email)
    return {"access_token": token, "token_type": "bearer"}

@app.get("/")
def home():
    return {"message": "SmartDCAI API is running"}

@app.get("/cities")
def cities():
    return {"supported_cities": get_supported_cities()}

@app.get("/predict/{city_name}")
def predict(city_name: str, current_user=Depends(get_current_user)):
    try:
        return predict_city(city_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/predict/upload")
async def predict_upload(
    files: List[UploadFile] = File(...),
    current_user=Depends(get_current_user)
):
    """
    Upload one or more raw NSRDB CSV files.
    Each file = one year of hourly solar data.
    More years = better accuracy.
    """
    try:
        return await predict_from_uploaded_csvs(files)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

# Fix Swagger UI to show a real file picker
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
