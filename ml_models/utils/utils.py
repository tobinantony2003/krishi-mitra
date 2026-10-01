from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any, Dict, Optional


KERALA_DISTRICT_CLIMATE = {
    "Palakkad": {"avg_temp": 29.0, "avg_rainfall": 180.0, "humidity": 78.0},
    "Thrissur": {"avg_temp": 28.0, "avg_rainfall": 210.0, "humidity": 82.0},
    "Wayanad": {"avg_temp": 24.5, "avg_rainfall": 260.0, "humidity": 88.0},
    "Malappuram": {"avg_temp": 27.5, "avg_rainfall": 230.0, "humidity": 84.0},
    "Kannur": {"avg_temp": 27.0, "avg_rainfall": 240.0, "humidity": 86.0},
    "Kozhikode": {"avg_temp": 27.0, "avg_rainfall": 245.0, "humidity": 87.0},
    "Idukki": {"avg_temp": 23.5, "avg_rainfall": 250.0, "humidity": 86.0},
    "Ernakulam": {"avg_temp": 28.0, "avg_rainfall": 220.0, "humidity": 85.0},
    "Alappuzha": {"avg_temp": 28.2, "avg_rainfall": 235.0, "humidity": 88.0},
    "Kottayam": {"avg_temp": 27.2, "avg_rainfall": 240.0, "humidity": 87.0},
}


def get_district_climate(district: str) -> Dict[str, float]:
    """
    Kerala district → climate priors (synthetic defaults).
    Replace with IMD/KAU data if available.
    """
    d = (district or "").strip()
    if d in KERALA_DISTRICT_CLIMATE:
        return dict(KERALA_DISTRICT_CLIMATE[d])
    return {"avg_temp": 27.5, "avg_rainfall": 230.0, "humidity": 85.0}


def get_crop_stage(crop: str, days_since_sowing: int) -> str:
    """
    Simple stage heuristic for a few common crops.
    For pests/diseases you typically only need a coarse stage bucket.
    """
    d = int(days_since_sowing)
    c = (crop or "").strip().lower()

    if c in {"paddy", "rice"}:
        if d <= 14:
            return "Seedling"
        if d <= 45:
            return "Vegetative"
        if d <= 75:
            return "Flowering"
        return "Maturity"

    if c in {"banana"}:
        if d <= 45:
            return "Vegetative"
        if d <= 150:
            return "Flowering"
        return "Maturity"

    if d <= 20:
        return "Seedling"
    if d <= 60:
        return "Vegetative"
    if d <= 90:
        return "Flowering"
    return "Maturity"


def calculate_soil_quality(ph: float, organic_matter: float, n: float, p: float, k: float) -> float:
    """
    Composite soil quality score on [1, 10] using rough agronomic heuristics.
    - pH ideal near 6.5
    - organic matter ideal around 2-3% for many Kerala systems
    - NPK scaled softly
    """
    ph = float(ph)
    organic_matter = float(organic_matter)
    n, p, k = float(n), float(p), float(k)

    ph_score = max(0.0, 1.0 - abs(ph - 6.5) / 2.0)  # 0..1
    om_score = max(0.0, min(1.0, organic_matter / 3.0))
    npk_score = max(0.0, min(1.0, (n + p + k) / 300.0))

    score01 = 0.45 * ph_score + 0.35 * om_score + 0.20 * npk_score
    return max(1.0, min(10.0, 1.0 + 9.0 * score01))


@functools.lru_cache(maxsize=16)
def load_model(model_path: str) -> Any:
    """
    Load a model from disk with in-memory caching.
    Supports:
    - .pkl/.joblib (joblib)
    - .h5/.keras (tf.keras)
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(str(path))

    suffix = path.suffix.lower()
    if suffix in {".pkl", ".joblib"}:
        import joblib

        return joblib.load(path)

    if suffix in {".h5", ".keras"}:
        import tensorflow as tf

        return tf.keras.models.load_model(path)

    raise ValueError(f"Unsupported model extension: {suffix}")


def _generate_soil_advice(farmer_input: Dict[str, Any]) -> Dict[str, Any]:
    ph = farmer_input.get("pH", farmer_input.get("ph", 6.5))
    n = farmer_input.get("N", 80)
    p = farmer_input.get("P", 40)
    k = farmer_input.get("K", 40)
    om = farmer_input.get("organic_matter", 2.0)

    score = calculate_soil_quality(ph, om, n, p, k)
    tips = []
    if ph < 5.8:
        tips.append("Consider liming (based on soil test) to raise pH toward 6.0–6.8.")
    if ph > 7.4:
        tips.append("Avoid over-liming; consider organic matter additions to buffer pH.")
    if om < 1.5:
        tips.append("Increase organic matter using compost/green manures (Kerala monsoon-friendly).")
    return {"soil_quality_score": score, "tips": tips or ["Maintain balanced nutrients and organic matter."]}


def predict_all(
    farmer_input: Dict[str, Any],
    crop_model_path: Optional[str] = None,
    yield_model_path: Optional[str] = None,
    pest_model_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Unified interface that can call the 3 tabular models (crop/yield/pest) if paths are provided.
    Disease model is image-based and not included here.
    """
    out: Dict[str, Any] = {"soil_advice": _generate_soil_advice(farmer_input)}

    if crop_model_path:
        model = load_model(crop_model_path)
        # Expect a sklearn pipeline supporting predict_proba; caller can post-process.
        out["crop_model_ready"] = True
        out["crop_model_type"] = type(model).__name__

    if yield_model_path:
        model = load_model(yield_model_path)
        out["yield_model_ready"] = True
        out["yield_model_type"] = type(model).__name__

    if pest_model_path:
        model = load_model(pest_model_path)
        out["pest_model_ready"] = True
        out["pest_model_type"] = type(model).__name__

    return out

