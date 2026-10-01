"""
Synthetic Kerala-focused datasets used across Krishi Mitra ML notebooks.

This module is intentionally standalone and Colab-friendly.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


KERALA_DISTRICTS: List[str] = [
    "Palakkad",
    "Thrissur",
    "Wayanad",
    "Malappuram",
    "Kannur",
    "Kozhikode",
    "Idukki",
    "Ernakulam",
    "Alappuzha",
    "Kottayam",
]

SOIL_TYPES: List[str] = ["Laterite", "Alluvial", "Sandy_Loam", "Red_Loam", "Forest"]
SEASONS: List[str] = ["Kharif", "Rabi", "Summer"]
WATER_SOURCES: List[str] = ["Rainfall", "Canal", "Borewell", "Pond", "River"]

KERALA_CROPS: List[str] = [
    "Paddy",
    "Coconut",
    "Banana",
    "Tapioca",
    "Rubber",
    "Black_Pepper",
    "Cardamom",
    "Ginger",
    "Turmeric",
    "Coffee",
    "Tea",
    "Arecanut",
    "Cashew",
    "Mango",
    "Jackfruit",
    "Maize",
    "Sugarcane",
    "Bitter_Gourd",
    "Okra",
    "Brinjal",
]


# District-wise climate priors (approximate; used for synthetic augmentation only).
# Values are chosen to be plausible for Kerala and to create separability for models.
DISTRICT_CLIMATE_PRIORS: Dict[str, Dict[str, float]] = {
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


def _rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


def _clip(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(x, lo, hi)


def get_district_climate(district: str) -> Dict[str, float]:
    if district not in DISTRICT_CLIMATE_PRIORS:
        # Fall back to Kerala-like averages if unknown.
        return {"avg_temp": 27.5, "avg_rainfall": 230.0, "humidity": 85.0}
    return dict(DISTRICT_CLIMATE_PRIORS[district])


def generate_pest_risk_dataset(
    n_rows: int = 10_000,
    seed: int | None = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic pest risk dataset using domain rules + noise.

    Target columns:
      - pest_risk_level: Low/Medium/High
      - pest_name
    """

    rng = _rng(seed)

    districts = rng.choice(KERALA_DISTRICTS, size=n_rows, replace=True)
    crops = rng.choice(["Paddy", "Coconut", "Banana", "Black_Pepper"], size=n_rows, replace=True)
    months = rng.integers(1, 13, size=n_rows)

    # Base climate sampled around district priors + seasonal effects.
    temps = np.zeros(n_rows, dtype=float)
    hums = np.zeros(n_rows, dtype=float)
    rains7 = np.zeros(n_rows, dtype=float)
    winds = np.zeros(n_rows, dtype=float)

    for i, d in enumerate(districts):
        pri = get_district_climate(d)
        mon = months[i]
        mon_phase = (mon - 1) / 12.0 * 2 * math.pi
        seasonal_temp = 1.2 * math.sin(mon_phase + 0.8)
        seasonal_hum = 2.5 * math.sin(mon_phase + 1.4)
        seasonal_rain = 25.0 * max(0.0, math.sin(mon_phase + 2.0))

        temps[i] = pri["avg_temp"] + seasonal_temp + rng.normal(0, 1.8)
        hums[i] = pri["humidity"] + seasonal_hum + rng.normal(0, 4.5)
        rains7[i] = max(0.0, (pri["avg_rainfall"] / 4.0) + seasonal_rain + rng.normal(0, 18.0))
        winds[i] = max(0.0, rng.normal(9.0, 3.0))

    temps = _clip(temps, 18.0, 36.0)
    hums = _clip(hums, 45.0, 98.0)
    rains7 = _clip(rains7, 0.0, 250.0)
    winds = _clip(winds, 0.0, 35.0)

    prev_incident = rng.random(n_rows) < 0.18

    growth_stages = rng.choice(
        ["Seedling", "Vegetative", "Flowering", "Maturity"],
        size=n_rows,
        replace=True,
        p=[0.2, 0.35, 0.25, 0.2],
    )

    cloudy = rng.random(n_rows) < (0.35 + 0.25 * (hums > 85))

    pest_name = np.array(["None"] * n_rows, dtype=object)
    risk_level = np.array(["Low"] * n_rows, dtype=object)

    # Domain rules (Kerala-oriented) — applied with priority.
    # Brown Plant Hopper (Paddy): humidity > 80% and temp 25-30
    cond_bph = (crops == "Paddy") & (hums > 80) & (temps >= 25) & (temps <= 30) & (rains7 > 20)
    pest_name[cond_bph] = "Brown_Plant_Hopper"
    risk_level[cond_bph] = "High"

    # Leaf Blast (Paddy): humidity > 90 + temp 20-25 + cloudy
    cond_blast = (crops == "Paddy") & (hums > 90) & (temps >= 20) & (temps <= 25) & cloudy
    pest_name[cond_blast] = "Leaf_Blast"
    risk_level[cond_blast] = "High"

    # Aphids (Veg + Banana): temp > 30 + dry-ish conditions
    cond_aphids = (crops != "Coconut") & (temps > 30) & (hums < 70) & (rains7 < 20)
    pest_name[cond_aphids] = "Aphids"
    risk_level[cond_aphids] = "High"

    # Rhinoceros Beetle (Coconut): moderate year-round; higher in coastal/humid
    coastal = np.isin(districts, ["Alappuzha", "Ernakulam", "Kannur", "Kozhikode"])
    cond_rhino = (crops == "Coconut") & (rng.random(n_rows) < (0.35 + 0.2 * coastal))
    pest_name[cond_rhino] = "Rhinoceros_Beetle"
    risk_level[cond_rhino] = np.where(hums > 85, "High", "Medium")

    # Stem borer (Paddy): high risk at tillering stage (approx. vegetative)
    cond_stem = (crops == "Paddy") & (growth_stages == "Vegetative") & (rains7 > 30) & (hums > 75)
    pest_name[cond_stem] = "Stem_Borer"
    risk_level[cond_stem] = "High"

    # Pepper: Pollu Beetle higher in dry months; Phytophthora in wet & humid
    cond_pepper_phyt = (crops == "Black_Pepper") & (hums > 88) & (rains7 > 60)
    pest_name[cond_pepper_phyt] = "Phytophthora"
    risk_level[cond_pepper_phyt] = "High"

    cond_pepper_pollu = (crops == "Black_Pepper") & (hums < 78) & (rains7 < 25) & (temps > 28)
    pest_name[cond_pepper_pollu] = "Pollu_Beetle"
    risk_level[cond_pepper_pollu] = "Medium"

    # Boost risk if previous incident.
    risk_level = np.where(
        prev_incident & (risk_level == "Low"),
        "Medium",
        risk_level,
    )
    risk_level = np.where(
        prev_incident & (risk_level == "Medium") & (pest_name != "None"),
        "High",
        risk_level,
    )

    # Small stochasticity: avoid perfect rules.
    flip = rng.random(n_rows) < 0.03
    risk_level = np.where(flip & (risk_level == "High"), "Medium", risk_level)
    flip2 = rng.random(n_rows) < 0.02
    risk_level = np.where(flip2 & (risk_level == "Low") & (pest_name != "None"), "Medium", risk_level)

    df = pd.DataFrame(
        {
            "district": districts,
            "crop": crops,
            "month": months,
            "avg_temperature": temps.round(2),
            "humidity_percent": hums.round(1),
            "rainfall_last_7days_mm": rains7.round(1),
            "wind_speed_kmh": winds.round(1),
            "previous_pest_incident": prev_incident.astype(int),
            "crop_growth_stage": growth_stages,
            "pest_name": pest_name,
            "pest_risk_level": risk_level,
            "cloudy": cloudy.astype(int),
        }
    )

    return df


def save_csv(df: pd.DataFrame, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pest_rows", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "processed"),
        help="Directory to write processed synthetic data",
    )
    args = parser.parse_args()

    pest_df = generate_pest_risk_dataset(n_rows=args.pest_rows, seed=args.seed)
    out_pest = os.path.join(args.out_dir, "pest_risk_synthetic.csv")
    save_csv(pest_df, out_pest)

    meta = {
        "kerala_districts": KERALA_DISTRICTS,
        "soil_types": SOIL_TYPES,
        "seasons": SEASONS,
        "water_sources": WATER_SOURCES,
        "kerala_crops": KERALA_CROPS,
    }
    with open(os.path.join(args.out_dir, "kerala_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote: {out_pest}")
    print(f"Wrote: {os.path.join(args.out_dir, 'kerala_metadata.json')}")


if __name__ == "__main__":
    main()
