from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ml_models.assistant.orchestrator import (
    FarmerParcel,
    FarmerProfile,
    chat_with_assistant,
    load_farmer_profile,
    save_farmer_profile,
)

ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = ROOT / "frontend"


class ParcelIn(BaseModel):
    id: str
    area_hectares: float
    crop: Optional[str] = None
    variety: Optional[str] = None
    sowing_date: Optional[str] = Field(
        default=None,
        description="ISO date string (YYYY-MM-DD)",
    )
    irrigation_type: Optional[str] = None


class FarmerProfileIn(BaseModel):
    farmer_id: str
    name: str
    district: str
    soil_type: str
    water_source: str
    # Pydantic v2 removed `regex` kwarg; use `pattern` instead.
    preferred_language: str = Field(default="ml", pattern="^(ml|en)$")
    prefers_organic: bool = True
    parcels: List[ParcelIn] = Field(default_factory=list)


class ChatRequest(BaseModel):
    farmer_id: str
    message: str
    language: Optional[str] = None


app = FastAPI(title="Krishi Mitra Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/farmers", response_model=Dict[str, Any])
def create_or_update_farmer(profile_in: FarmerProfileIn):
    parcels = [
        FarmerParcel(
            id=p.id,
            area_hectares=p.area_hectares,
            crop=p.crop,
            variety=p.variety,
            sowing_date=p.sowing_date,
            irrigation_type=p.irrigation_type,
        )
        for p in profile_in.parcels
    ]

    profile = FarmerProfile(
        farmer_id=profile_in.farmer_id,
        name=profile_in.name,
        district=profile_in.district,
        soil_type=profile_in.soil_type,
        water_source=profile_in.water_source,
        preferred_language=profile_in.preferred_language,
        prefers_organic=profile_in.prefers_organic,
        parcels=parcels,
    )
    save_farmer_profile(profile)
    return {"status": "ok", "farmer_id": profile.farmer_id}


@app.get("/api/farmers/{farmer_id}", response_model=Dict[str, Any])
def get_farmer(farmer_id: str):
    profile = load_farmer_profile(farmer_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="Farmer not found")
    from dataclasses import asdict

    return asdict(profile)


@app.post("/api/assistant/chat", response_model=Dict[str, Any])
def assistant_chat(req: ChatRequest):
    try:
        out = chat_with_assistant(req.farmer_id, req.message, language=req.language)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return out


# Serve frontend
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/", include_in_schema=False)
def index():
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend index.html not found")
    return FileResponse(str(index_path))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)

