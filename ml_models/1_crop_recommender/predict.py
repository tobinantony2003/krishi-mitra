from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd


def _season_hint(crop: str) -> str:
    mapping = {
        "Paddy": "Kharif",
        "Banana": "Year-round (avoid peak flood weeks)",
        "Coconut": "Year-round",
        "Black_Pepper": "Kharif/Rabi (with drainage)",
        "Ginger": "Kharif",
        "Turmeric": "Kharif",
        "Tapioca": "Rabi/Summer",
        "Rubber": "Monsoon planting",
    }
    return mapping.get(crop, "Season depends on local practice")


def _avg_yield_hint(crop: str) -> str:
    yields = {
        "Paddy": "3–5 t/ha",
        "Banana": "25–40 t/ha",
        "Coconut": "8,000–12,000 nuts/ha",
        "Tapioca": "20–30 t/ha",
        "Ginger": "10–15 t/ha (green)",
        "Turmeric": "20–30 t/ha (green)",
        "Black_Pepper": "0.3–0.8 t/ha (dry)",
        "Rubber": "1.2–1.8 t/ha (dry rubber)",
    }
    return yields.get(crop, "Varies by variety & management")


def _reason(inp: Dict[str, Any], crop: str, score: float) -> str:
    reasons: List[str] = []
    if inp.get("season") == "Kharif" and float(inp.get("rainfall", 0)) > 150:
        reasons.append("Good monsoon moisture")
    if inp.get("soil_type") in ("Laterite", "Red_Loam") and crop in ("Cashew", "Tapioca", "Mango", "Jackfruit"):
        reasons.append("Matches laterite/red loam adaptation")
    if float(inp.get("humidity", 0)) > 80 and crop in ("Paddy", "Banana", "Coconut", "Black_Pepper"):
        reasons.append("High humidity supports this crop (ensure disease management)")
    if crop == "Paddy" and inp.get("water_source") in ("Canal", "River", "Pond"):
        reasons.append("Reliable water source suits paddy")
    if not reasons:
        reasons.append("Overall feature match to training patterns")
    return f"Score {score:.2f}. " + "; ".join(reasons)


def predict(
    farmer_input: Dict[str, Any],
    model_path: str | Path,
    label_encoder_path: str | Path,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    model = joblib.load(model_path)
    le = joblib.load(label_encoder_path)

    row = {
        "district": farmer_input["district"],
        "soil_type": farmer_input["soil_type"],
        "N": farmer_input["N"],
        "P": farmer_input["P"],
        "K": farmer_input["K"],
        "ph": farmer_input["pH"],
        "rainfall": farmer_input["rainfall"],
        "temperature": farmer_input["temperature"],
        "humidity": farmer_input["humidity"],
        "season": farmer_input["season"],
        "water_source": farmer_input["water_source"],
    }
    X_one = pd.DataFrame([row])
    proba = model.predict_proba(X_one)[0]
    idx = np.argsort(proba)[::-1][:top_k]

    out: List[Dict[str, Any]] = []
    for i in idx:
        crop = le.inverse_transform([i])[0]
        score = float(proba[i])
        out.append(
            {
                "crop": crop,
                "suitability_score": score,
                "reason": _reason(farmer_input, crop, score),
                "best_season": _season_hint(crop),
                "avg_yield": _avg_yield_hint(crop),
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default=str(Path(__file__).parent / "model" / "crop_model.pkl"))
    ap.add_argument("--le", type=str, default=str(Path(__file__).parent / "model" / "label_encoder.pkl"))
    args = ap.parse_args()

    sample = {
        "district": "Palakkad",
        "soil_type": "Laterite",
        "N": 90,
        "P": 42,
        "K": 43,
        "pH": 6.5,
        "rainfall": 200,
        "temperature": 28,
        "humidity": 82,
        "season": "Kharif",
        "water_source": "Canal",
    }

    preds = predict(sample, args.model, args.le, top_k=5)
    for p in preds:
        print(p)


if __name__ == "__main__":
    main()

