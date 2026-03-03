import pandas as pd
from fastapi import UploadFile, HTTPException

REQUIRED_COLUMNS = {"Year", "Month", "GHI"}

async def predict_from_uploaded_csv(file: UploadFile):
    # 1) Basic file checks
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file")

    # 2) Read file into dataframe
    try:
        contents = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(contents), skiprows=2)  # keep your NSRDB skiprows
        df.columns = [str(c).strip() for c in df.columns]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {str(e)}")

    # 3) Validate minimum columns
    if not REQUIRED_COLUMNS.issubset(df.columns):
        missing = REQUIRED_COLUMNS - set(df.columns)
        raise HTTPException(status_code=400, detail=f"Missing required columns: {list(missing)}")

    # TEMP response (Step 1 success)
    return {
        "message": "Upload received and CSV parsed successfully",
        "rows": int(len(df)),
        "columns_sample": df.columns[:15].tolist()
    }
