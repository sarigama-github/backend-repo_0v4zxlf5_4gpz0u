import os
import io
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Attempt optional image libs. If unavailable, the API will still run with a fallback.
try:
    from PIL import Image, ImageFilter  # type: ignore
    import numpy as np  # type: ignore
    PIL_AVAILABLE = True
except Exception:
    Image = None  # type: ignore
    ImageFilter = None  # type: ignore
    np = None  # type: ignore
    PIL_AVAILABLE = False

# Database helpers
from database import db, create_document, get_documents

app = FastAPI(title="FaceCare AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalysisResponse(BaseModel):
    image_url: str
    skin_type: Optional[str]
    detected_issues: List[str]
    recommendations: List[str]


@app.get("/")
def root():
    return {
        "message": "FaceCare AI backend is running",
        "image_processing": "enabled" if PIL_AVAILABLE else "fallback",
    }


@app.get("/test")
def test_database():
    """Verify DB connectivity"""
    resp = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "image_processing": "enabled" if PIL_AVAILABLE else "fallback",
    }
    try:
        _ = db.list_collection_names()
        resp["database"] = "✅ Connected & Working"
    except Exception as e:
        resp["database"] = f"⚠️ {str(e)[:80]}"
    return resp


# Simple heuristic skin analysis (when PIL/numpy available)
def analyze_image_with_cv(image):
    img = image.convert("RGB")
    arr = np.array(img)

    brightness = arr.mean()
    edges = np.array(img.convert("L").filter(ImageFilter.FIND_EDGES))
    texture_score = edges.mean()

    h, w, _ = arr.shape
    tzone = arr[:, w // 3 : 2 * w // 3, :].astype(np.float32)
    tzone_var = tzone.var()

    r, g, b = (
        arr[:, :, 0].astype(np.float32),
        arr[:, :, 1].astype(np.float32),
        arr[:, :, 2].astype(np.float32),
    )
    redness = (r - (g + b) / 2).mean()

    issues: List[str] = []

    if brightness < 90:
        issues.append("dullness")
    if texture_score > 25:
        issues.append("texture/pores")
    if tzone_var > 4000:
        issues.append("oiliness in T-zone")
    if redness > 15:
        issues.append("redness/irritation")

    if "oiliness in T-zone" in issues and brightness > 110:
        skin_type = "combination"
    elif tzone_var > 4500:
        skin_type = "oily"
    elif brightness < 95 and texture_score < 20:
        skin_type = "dry"
    else:
        skin_type = "normal"

    recs: List[str] = []
    if "dullness" in issues:
        recs.append("Introduce gentle chemical exfoliation 2-3x/week (AHA like lactic acid).")
        recs.append("Add Vitamin C serum in the morning to boost radiance.")
    if "texture/pores" in issues:
        recs.append("Use BHA (salicylic acid) 2-3x/week for pores and texture.")
        recs.append("Avoid harsh physical scrubs; focus on SPF daily.")
    if "oiliness in T-zone" in issues or skin_type == "oily":
        recs.append("Use gel-based, non-comedogenic moisturizer.")
        recs.append("Consider niacinamide 5% serum to regulate sebum.")
        recs.append("Use blotting papers or setting powder during the day.")
    if "redness/irritation" in issues:
        recs.append("Simplify routine; avoid fragrance and alcohol in products.")
        recs.append("Look for soothing ingredients: centella, panthenol, aloe, green tea.")
    if skin_type == "dry":
        recs.append("Use a hydrating cleanser and thicker moisturizer with ceramides.")
        recs.append("Add hyaluronic acid serum on damp skin and seal with moisturizer.")
    recs.append("Apply broad-spectrum SPF 30+ every morning.")

    return {
        "skin_type": skin_type,
        "detected_issues": list(dict.fromkeys(issues)),
        "recommendations": list(dict.fromkeys(recs)),
    }


# Fallback analysis when image libs are not installed
def analyze_image_fallback():
    recs = [
        "Cleanse twice daily with a gentle, pH-balanced cleanser.",
        "Moisturize morning and night; choose non-comedogenic formulas.",
        "Apply broad-spectrum SPF 30+ every morning.",
        "Introduce actives gradually (niacinamide, AHA/BHA) 2-3x/week.",
    ]
    return {
        "skin_type": None,
        "detected_issues": [],
        "recommendations": recs,
    }


@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_face(file: UploadFile = File(...)):
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=400, detail="Please upload a JPG, PNG, or WEBP image.")

    content = await file.read()
    pseudo_url = f"uploaded://{file.filename}"

    if PIL_AVAILABLE:
        try:
            image = Image.open(io.BytesIO(content))  # type: ignore
            result = analyze_image_with_cv(image)
        except Exception:
            # If decoding or analysis fails, fall back
            result = analyze_image_fallback()
    else:
        result = analyze_image_fallback()

    # Persist analysis (best-effort)
    doc = {"image_url": pseudo_url, **result}
    try:
        _ = create_document("analysis", doc)
    except Exception:
        pass

    return AnalysisResponse(image_url=pseudo_url, **result)


class SeedProduct(BaseModel):
    issue: str
    product_name: str
    category: str
    ingredients: Optional[List[str]] = None
    notes: Optional[str] = None


@app.post("/api/seed-product")
async def seed_product(item: SeedProduct):
    try:
        _id = create_document("productsuggestion", item.dict())
        return {"ok": True, "id": str(_id)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/recommendations")
async def list_recommendations(issue: Optional[str] = None):
    try:
        filter_dict = {"issue": issue} if issue else {}
        docs = get_documents("productsuggestion", filter_dict, limit=50)
        return {"items": docs}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
