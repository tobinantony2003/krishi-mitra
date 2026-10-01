from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd


def predict(farmer_input: Dict[str, Any], model_path: str | Path) -> Dict[str, Any]:
    model = joblib.load(model_path)

    row = {
        "crop_name": farmer_input["crop"],
        "district": farmer_input["district"],
        "year": farmer_input.get("year", 2025),
        "area_hectares": farmer_input["area_hectares"],
        "annual_rainfall_mm": farmer_input["rainfall_mm"],
        "avg_temperature": farmer_input["temperature"],
        "soil_quality_index": farmer_input["soil_quality"],
        "fertilizer_kg_per_hectare": farmer_input["fertilizer_used"],
        "pesticide_kg_per_hectare": farmer_input.get("pesticide_used", 3.0),
        "irrigation_type": farmer_input["irrigation_type"],
        "season": farmer_input["season"],
    }
    yhat = float(model.predict(pd.DataFrame([row]))[0])
    return {"predicted_yield_tons_per_hectare": yhat}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default=str(Path(__file__).parent / "model" / "yield_model.pkl"))
    args = ap.parse_args()

    sample = {
        "crop": "Paddy",
        "district": "Palakkad",
        "year": 2025,
        "area_hectares": 1.5,
        "rainfall_mm": 2800,
        "temperature": 28,
        "soil_quality": 7,
        "fertilizer_used": 140,
        "irrigation_type": "Canal",
        "season": "Kharif",
    }

    out = predict(sample, args.model)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

