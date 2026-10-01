from __future__ import annotations

"""
High-level orchestration for the Krishi Mitra personal farming assistant.

This module:
- Manages a simple per-farmer profile (JSON on disk)
- Calls the trained ML models (crop, yield, pest) for a given context
- Uses the RAG chatbot pipeline to turn model outputs into conversational advice

It is designed to be imported from:
- a Colab notebook, or
- a thin CLI / backend later on.
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..utils.utils import (
    calculate_soil_quality,
    get_crop_stage,
    get_district_climate,
    load_model,
)


ROOT = Path(__file__).resolve().parents[1]
ASSISTANT_DIR = ROOT / "assistant"
FARMERS_DIR = ASSISTANT_DIR / "farmers"
FARMERS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class FarmerParcel:
    id: str
    area_hectares: float
    crop: Optional[str] = None
    variety: Optional[str] = None
    sowing_date: Optional[str] = None  # ISO date string
    irrigation_type: Optional[str] = None


@dataclass
class FarmerProfile:
    farmer_id: str
    name: str
    district: str
    soil_type: str
    water_source: str
    preferred_language: str = "ml"  # "ml" or "en"
    prefers_organic: bool = True
    parcels: List[FarmerParcel] = field(default_factory=list)
    last_updated: Optional[str] = None


@dataclass
class Advisory:
    today: List[str]
    this_week: List[str]
    season_plan: List[str]
    meta: Dict[str, Any] = field(default_factory=dict)


def _farmer_path(farmer_id: str) -> Path:
    return FARMERS_DIR / f"{farmer_id}.json"


def load_farmer_profile(farmer_id: str) -> Optional[FarmerProfile]:
    path = _farmer_path(farmer_id)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    parcels = [FarmerParcel(**p) for p in data.get("parcels", [])]
    data["parcels"] = parcels
    return FarmerProfile(**data)


def save_farmer_profile(profile: FarmerProfile) -> None:
    profile.last_updated = datetime.utcnow().isoformat()
    serialized = asdict(profile)
    _farmer_path(profile.farmer_id).write_text(json.dumps(serialized, indent=2), encoding="utf-8")


def _load_rag_chat():
    """
    Load RAG chatbot function from 5_chatbot/rag_pipeline.py via importlib.
    We avoid treating '5_chatbot' as a Python package because of the digit.
    """
    import sys
    import importlib.util

    rag_path = ROOT / "5_chatbot" / "rag_pipeline.py"
    spec = importlib.util.spec_from_file_location("km_rag_pipeline", rag_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # Required for dataclasses / type inspection during module execution.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.chat


def _default_model_paths() -> Dict[str, Path]:
    return {
        "crop_model": ROOT / "1_crop_recommender" / "model" / "crop_model.pkl",
        "crop_le": ROOT / "1_crop_recommender" / "model" / "label_encoder.pkl",
        "yield_model": ROOT / "3_yield_predictor" / "model" / "yield_model.pkl",
        "pest_model": ROOT / "4_pest_risk_forecaster" / "model" / "pest_model.pkl",
    }


def _days_since(date_str: Optional[str]) -> Optional[int]:
    if not date_str:
        return None
    try:
        dt = datetime.fromisoformat(date_str)
        return (datetime.utcnow().date() - dt.date()).days
    except Exception:
        return None


def _build_farmer_context(profile: FarmerProfile) -> Dict[str, Any]:
    """Flatten key farm context into a single dict used by models and RAG."""
    district_climate = get_district_climate(profile.district)
    ctx: Dict[str, Any] = {
        "farmer_id": profile.farmer_id,
        "name": profile.name,
        "district": profile.district,
        "soil_type": profile.soil_type,
        "water_source": profile.water_source,
        "preferred_language": profile.preferred_language,
        "prefers_organic": profile.prefers_organic,
        "district_climate": district_climate,
        "parcels": [],
    }
    for p in profile.parcels:
        days = _days_since(p.sowing_date)
        stage = get_crop_stage(p.crop or "", days or 0) if p.crop and days is not None else None
        ctx["parcels"].append(
            {
                "id": p.id,
                "area_hectares": p.area_hectares,
                "crop": p.crop,
                "variety": p.variety,
                "sowing_date": p.sowing_date,
                "days_since_sowing": days,
                "growth_stage": stage,
                "irrigation_type": p.irrigation_type,
            }
        )
    return ctx


def _run_crop_model(profile: FarmerProfile, paths: Dict[str, Path]) -> List[Dict[str, Any]]:
    """Run crop recommender on the largest parcel (or first) to propose crops."""
    try:
        crop_model = load_model(str(paths["crop_model"]))
        import joblib

        le = joblib.load(paths["crop_le"])
    except FileNotFoundError:
        # Trained artifacts not shipped / not downloaded yet.
        return []

    if not profile.parcels:
        return []

    parcel = profile.parcels[0]

    # simple synthetic agronomic inputs anchored on district climate
    climate = get_district_climate(profile.district)
    N, P, K = 100, 40, 40
    ph = 6.5
    rainfall = climate["avg_rainfall"]
    temperature = climate["avg_temp"]
    humidity = climate["humidity"]
    season = "Kharif"

    import pandas as pd

    row = {
        "district": profile.district,
        "soil_type": profile.soil_type,
        "N": N,
        "P": P,
        "K": K,
        "ph": ph,
        "rainfall": rainfall,
        "temperature": temperature,
        "humidity": humidity,
        "season": season,
        "water_source": profile.water_source,
    }
    X = pd.DataFrame([row])
    proba = crop_model.predict_proba(X)[0]
    idx = np.argsort(proba)[::-1][:5]

    out: List[Dict[str, Any]] = []
    for i in idx:
        crop = le.inverse_transform([i])[0]
        score = float(proba[i])
        out.append({"crop": crop, "score": score})
    return out


def _run_yield_model(profile: FarmerProfile, paths: Dict[str, Path]) -> Optional[Dict[str, Any]]:
    """Predict yield for the main crop/parcel if available."""
    if not profile.parcels or not profile.parcels[0].crop:
        return None

    parcel = profile.parcels[0]
    try:
        yield_model = load_model(str(paths["yield_model"]))
    except FileNotFoundError:
        # Trained artifacts not shipped / not downloaded yet.
        return None

    climate = get_district_climate(profile.district)
    rainfall_mm = climate["avg_rainfall"] * 10  # crude annualization
    temperature = climate["avg_temp"]
    soil_quality = calculate_soil_quality(6.5, 2.0, 100, 40, 40)

    import pandas as pd

    row = {
        "crop_name": parcel.crop,
        "district": profile.district,
        "year": datetime.utcnow().year,
        "area_hectares": parcel.area_hectares,
        "annual_rainfall_mm": rainfall_mm,
        "avg_temperature": temperature,
        "soil_quality_index": soil_quality,
        "fertilizer_kg_per_hectare": 140,
        "pesticide_kg_per_hectare": 3.0,
        "irrigation_type": parcel.irrigation_type or "Rainfed",
        "season": "Kharif",
        "previous_year_yield": 3.5,
        "avg_3yr_yield": 3.6,
        "drought_index": max(0.0, 1.0 - rainfall_mm / 3500.0),
        "flood_risk_score": min(1.0, rainfall_mm / 3500.0 + 0.2),
    }
    X = pd.DataFrame([row])
    yhat = float(yield_model.predict(X)[0])
    return {"parcel_id": parcel.id, "crop": parcel.crop, "predicted_tph": yhat}


def _run_pest_risk(profile: FarmerProfile, paths: Dict[str, Path]) -> Optional[Dict[str, Any]]:
    """Get a simple pest risk classification for the main parcel."""
    if not profile.parcels or not profile.parcels[0].crop:
        return None

    parcel = profile.parcels[0]
    try:
        pest_model = load_model(str(paths["pest_model"]))
    except FileNotFoundError:
        # Trained artifacts not shipped / not downloaded yet.
        return None

    climate = get_district_climate(profile.district)
    temperature = climate["avg_temp"]
    humidity = climate["humidity"]
    rainfall_7 = climate["avg_rainfall"]  # very rough

    import pandas as pd

    days = _days_since(parcel.sowing_date) or 30
    stage = get_crop_stage(parcel.crop, days)

    row = {
        "district": profile.district,
        "crop": parcel.crop,
        "month": datetime.utcnow().month,
        "avg_temperature": temperature,
        "humidity_percent": humidity,
        "rainfall_last_7days_mm": rainfall_7,
        "wind_speed_kmh": 9.0,
        "previous_pest_incident": 0,
        "crop_growth_stage": stage,
        "cloudy": 1,
    }
    X = pd.DataFrame([row])
    risk = pest_model.predict(X)[0]
    proba = pest_model.predict_proba(X)[0]
    classes = list(pest_model.named_steps["clf"].classes_)
    risk_prob = float(proba[classes.index(risk)])
    return {"risk_level": str(risk), "probability": risk_prob, "growth_stage": stage}


def generate_weekly_advisory(profile: FarmerProfile) -> Advisory:
    """
    Combine pest risk + basic crop growth stage insights into a weekly checklist.
    """
    paths = _default_model_paths()
    pest = _run_pest_risk(profile, paths)

    today: List[str] = []
    this_week: List[str] = []
    season_plan: List[str] = []

    if pest:
        level = str(pest["risk_level"]).lower()
        if level == "high":
            today.append(
                "Scout your main field for pest/damage symptoms today, focusing on lower leaves and tillers."
            )
            this_week.append(
                "If confirmed pest levels are high, consult local extension for IPM spray schedule; avoid unnecessary broad-spectrum sprays."
            )
        elif level == "medium":
            this_week.append(
                "Check 10–20 random plants in the field for early pest signs; use traps or light monitoring where available."
            )
        else:
            this_week.append("Pest risk currently low; continue regular field scouting once this week.")
    else:
        # Keep the assistant usable even without trained pest model artifacts.
        this_week.append("Pest risk model not available yet; rely on field scouting and local extension guidance for IPM timing.")

    if profile.parcels:
        parcel = profile.parcels[0]
        days = _days_since(parcel.sowing_date) or 0
        stage = get_crop_stage(parcel.crop or "", days)
        if stage == "Seedling":
            season_plan.append("Protect seedlings from damping-off and early insect damage; avoid waterlogging.")
        elif stage == "Vegetative":
            season_plan.append("Focus on balanced nitrogen and weed control during vegetative stage.")
        elif stage == "Flowering":
            season_plan.append("Avoid moisture stress during flowering; monitor for diseases on upper canopy.")
        elif stage == "Maturity":
            season_plan.append("Plan harvest window considering forecast; avoid lodging and shattering losses.")

    return Advisory(today=today, this_week=this_week, season_plan=season_plan, meta={"pest": pest})


def generate_season_plan(profile: FarmerProfile) -> Advisory:
    """
    Use crop + yield models to suggest cropping and yield expectations for the coming season.
    """
    paths = _default_model_paths()
    crop_suggestions = _run_crop_model(profile, paths)
    yield_info = _run_yield_model(profile, paths)

    today: List[str] = []
    this_week: List[str] = []
    season_plan: List[str] = []

    if crop_suggestions:
        top = crop_suggestions[0]
        season_plan.append(
            f"Top suggested crop for your main parcel is {top['crop']} (suitability score ~{top['score']:.2f})."
        )
        if len(crop_suggestions) > 1:
            alts = ", ".join([c["crop"] for c in crop_suggestions[1:3]])
            season_plan.append(f"Alternative options to consider: {alts}.")

    if yield_info:
        season_plan.append(
            f"Expected yield for {yield_info['crop']} on parcel {yield_info['parcel_id']} is about "
            f"{yield_info['predicted_tph']:.1f} tons/ha under typical management."
        )

    this_week.append("Finalize crop choice and seed/planting material for the coming season.")
    this_week.append("Arrange inputs (seed, organic manures, fertilizers) based on chosen crop and area.")

    return Advisory(
        today=today,
        this_week=this_week,
        season_plan=season_plan,
        meta={"crop_suggestions": crop_suggestions, "yield_info": yield_info},
    )


def chat_with_assistant(
    farmer_id: str,
    message: str,
    language: Optional[str] = None,
) -> Dict[str, Any]:
    """
    High-level entry point for a conversational assistant.
    - Loads farmer profile
    - Builds weekly + seasonal advisory context
    - Calls RAG chatbot to generate a natural language answer
    """
    profile = load_farmer_profile(farmer_id)
    if profile is None:
        raise ValueError(f"No profile found for farmer_id={farmer_id}. Create one first.")

    lang = language or profile.preferred_language or "ml"
    ctx = _build_farmer_context(profile)

    weekly = generate_weekly_advisory(profile)
    seasonal = generate_season_plan(profile)

    # Render advisory summaries as plain text to feed into the RAG context
    advisory_snippets = []
    if weekly.today:
        advisory_snippets.append("Today: " + "; ".join(weekly.today))
    if weekly.this_week:
        advisory_snippets.append("This week: " + "; ".join(weekly.this_week))
    if seasonal.season_plan:
        advisory_snippets.append("Season plan: " + "; ".join(seasonal.season_plan))

    advisory_text = "\n".join(advisory_snippets)

    rag_chat = _load_rag_chat()
    kb_dir = ROOT / "5_chatbot" / "knowledge_base"
    chroma_dir = ROOT / "5_chatbot" / ".chroma"

    # Extend user message with structured context for the LLM
    enriched_message = (
        f"Farmer context: district={profile.district}, soil_type={profile.soil_type}, "
        f"water_source={profile.water_source}, prefers_organic={profile.prefers_organic}. "
        f"Parcels={len(profile.parcels)}. Advisory summary: {advisory_text}\n\n"
        f"Farmer says: {message}"
    )

    resp = rag_chat(
        message=enriched_message,
        language=lang,
        chat_history=[],
        knowledge_base_dir=str(kb_dir),
        chroma_dir=str(chroma_dir),
    )

    return {
        "farmer_id": farmer_id,
        "profile": asdict(profile),
        "advisory": {
            "weekly": asdict(weekly),
            "seasonal": asdict(seasonal),
        },
        "assistant_response": resp,
    }


if __name__ == "__main__":
    # Minimal CLI smoke test (requires an existing farmer profile + trained models)
    fid = "demo_farmer"
    prof = load_farmer_profile(fid)
    if prof is None:
        prof = FarmerProfile(
            farmer_id=fid,
            name="Demo Farmer",
            district="Palakkad",
            soil_type="Laterite",
            water_source="Canal",
            parcels=[FarmerParcel(id="P1", area_hectares=1.5, crop="Paddy", sowing_date="2025-07-01")],
        )
        save_farmer_profile(prof)

    out = chat_with_assistant(fid, "What should I focus on this week?")
    print(json.dumps(out, indent=2, ensure_ascii=False))

