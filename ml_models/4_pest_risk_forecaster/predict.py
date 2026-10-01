from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd


PRACTICAL_ADVICE = {
    "Brown_Plant_Hopper": {
        "preventive": "Avoid excess nitrogen; maintain proper water level; encourage natural predators.",
        "spray": "If high: neem oil 3 ml/L or recommended insecticide per local extension advice.",
    },
    "Leaf_Blast": {
        "preventive": "Ensure spacing and airflow; avoid late evening overhead watering.",
        "spray": "If high: tricyclazole-based fungicide as per label/KAU guidance.",
    },
    "Aphids": {
        "preventive": "Use yellow sticky traps; manage weeds; avoid water stress.",
        "spray": "If high: neem oil/soap spray; chemical only if infestation severe.",
    },
    "Rhinoceros_Beetle": {
        "preventive": "Destroy breeding sites (rotting logs); use pheromone traps.",
        "spray": "If high: apply recommended measures to crown as advised.",
    },
    "Stem_Borer": {
        "preventive": "Remove stubbles; synchronize planting; avoid excess nitrogen.",
        "spray": "If high: follow local IPM schedule; use pheromone traps where available.",
    },
    "Phytophthora": {
        "preventive": "Improve drainage; avoid collar wetness; prune for airflow.",
        "spray": "If high: metalaxyl+mancozeb or copper-based sprays as recommended.",
    },
    "Pollu_Beetle": {
        "preventive": "Maintain shade balance; avoid drought stress; sanitation.",
        "spray": "If medium/high: follow pepper IPM guidance from local extension.",
    },
    "None": {"preventive": "Continue weekly scouting; maintain field hygiene.", "spray": "No spray recommended."},
}


def _guess_pest(row: Dict[str, Any]) -> str:
    crop = row["crop"]
    temp = float(row["avg_temperature"])
    hum = float(row["humidity_percent"])
    rain7 = float(row["rainfall_last_7days_mm"])
    cloudy = int(row.get("cloudy", 0)) == 1
    stage = row.get("crop_growth_stage", "")

    if crop == "Paddy" and hum > 80 and 25 <= temp <= 30 and rain7 > 20:
        return "Brown_Plant_Hopper"
    if crop == "Paddy" and hum > 90 and 20 <= temp <= 25 and cloudy:
        return "Leaf_Blast"
    if temp > 30 and hum < 70 and rain7 < 20:
        return "Aphids"
    if crop == "Coconut":
        return "Rhinoceros_Beetle"
    if crop == "Paddy" and stage == "Vegetative" and rain7 > 30 and hum > 75:
        return "Stem_Borer"
    if crop == "Black_Pepper" and hum > 88 and rain7 > 60:
        return "Phytophthora"
    if crop == "Black_Pepper" and hum < 78 and rain7 < 25 and temp > 28:
        return "Pollu_Beetle"
    return "None"


def predict(farmer_input: Dict[str, Any], model_path: str | Path, top_k: int = 3) -> List[Dict[str, Any]]:
    model = joblib.load(model_path)

    row = {
        "district": farmer_input["district"],
        "crop": farmer_input["crop"],
        "month": int(farmer_input["month"]),
        "avg_temperature": float(farmer_input["temperature"]),
        "humidity_percent": float(farmer_input["humidity"]),
        "rainfall_last_7days_mm": float(farmer_input["rainfall_7days"]),
        "wind_speed_kmh": float(farmer_input.get("wind_speed", 9.0)),
        "previous_pest_incident": int(bool(farmer_input.get("previous_pest_incident", False))),
        "crop_growth_stage": farmer_input["crop_growth_stage"],
        "cloudy": int(bool(farmer_input.get("cloudy", False))),
    }

    X_one = pd.DataFrame([row])
    proba = model.predict_proba(X_one)[0]
    classes = list(model.named_steps["clf"].classes_)
    idx = np.argsort(proba)[::-1]

    pest = _guess_pest(row)
    advice = PRACTICAL_ADVICE.get(pest, PRACTICAL_ADVICE["None"])

    out: List[Dict[str, Any]] = []
    for j in idx[:top_k]:
        out.append(
            {
                "pest_name": pest,
                "risk_level": str(classes[j]).lower(),
                "probability": float(proba[j]),
                "peak_window": "next 3-5 days",
                "preventive_action": advice["preventive"],
                "spray_recommendation": advice["spray"],
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default=str(Path(__file__).parent / "model" / "pest_model.pkl"))
    args = ap.parse_args()

    sample = {
        "district": "Palakkad",
        "crop": "Paddy",
        "month": 7,
        "temperature": 28,
        "humidity": 88,
        "rainfall_7days": 90,
        "crop_growth_stage": "Vegetative",
        "previous_pest_incident": True,
        "cloudy": True,
    }
    out = predict(sample, args.model)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

