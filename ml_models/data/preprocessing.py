"""
Centralized data preprocessing utilities for Krishi Mitra ML models.

This module focuses on tabular pipelines used by:
- Model 1 (crop recommender)
- Model 3 (yield predictor)
- Model 4 (pest risk forecaster, synthetic)

Each function is Colab-friendly and works with local paths as well.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .generate_synthetic_data import (
    KERALA_DISTRICTS,
    SOIL_TYPES,
    SEASONS,
    WATER_SOURCES,
    DISTRICT_CLIMATE_PRIORS,
    generate_pest_risk_dataset,
)


@dataclass
class SplitData:
    train: pd.DataFrame
    test: pd.DataFrame


def _rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# Model 1 — Crop recommendation preprocessing
# ---------------------------------------------------------------------------

def add_kerala_context_to_crop_df(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Add synthetic Kerala-specific categorical context to the Kaggle crop dataset:
    - district (10 Kerala districts)
    - soil_type (5 soil classes)
    - season (Kharif/Rabi/Summer)
    - water_source (Rainfall/Canal/Borewell/Pond/River)

    Also gently nudges temperature/humidity/rainfall towards district priors.
    """
    rng = _rng(seed)
    out = df.copy()

    districts = rng.choice(KERALA_DISTRICTS, size=len(out))
    soils = rng.choice(SOIL_TYPES, size=len(out), p=[0.38, 0.22, 0.12, 0.18, 0.10])
    months = rng.integers(1, 13, size=len(out))

    def season_for_month(m: int) -> str:
        if m in (6, 7, 8, 9):
            return "Kharif"
        if m in (10, 11, 12, 1, 2):
            return "Rabi"
        return "Summer"

    seasons = np.array([season_for_month(int(m)) for m in months])
    water: list[str] = []
    for s in seasons:
        if s == "Kharif":
            water.append(rng.choice(WATER_SOURCES, p=[0.55, 0.2, 0.05, 0.1, 0.1]))
        elif s == "Rabi":
            water.append(rng.choice(WATER_SOURCES, p=[0.2, 0.35, 0.15, 0.15, 0.15]))
        else:
            water.append(rng.choice(WATER_SOURCES, p=[0.15, 0.25, 0.25, 0.2, 0.15]))

    out["district"] = districts
    out["soil_type"] = soils
    out["season"] = seasons
    out["water_source"] = np.array(water)

    temp_prior = np.array([DISTRICT_CLIMATE_PRIORS[d]["avg_temp"] for d in out["district"]])
    rain_prior = np.array([DISTRICT_CLIMATE_PRIORS[d]["avg_rainfall"] for d in out["district"]])
    hum_prior = np.array([DISTRICT_CLIMATE_PRIORS[d]["humidity"] for d in out["district"]])

    # 15% pull towards Kerala priors, 85% retain original Kaggle numeric signal
    out["temperature"] = 0.85 * out["temperature"] + 0.15 * temp_prior
    out["rainfall"] = 0.85 * out["rainfall"] + 0.15 * rain_prior
    out["humidity"] = 0.85 * out["humidity"] + 0.15 * hum_prior

    return out


def preprocess_crop_recommendation(
    raw_csv: Path,
    out_dir: Path,
    seed: int = 42,
) -> SplitData:
    """
    Full preprocessing pipeline for the crop recommendation dataset:
    - load CSV and normalize column names
    - rename 'label' → 'crop'
    - add Kerala-specific categorical features
    - basic cleaning (drop obvious NaNs)
    - train/test split and save processed CSVs
    """
    from sklearn.model_selection import train_test_split

    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(raw_csv)
    df.columns = [c.strip() for c in df.columns]
    if "label" in df.columns:
        df = df.rename(columns={"label": "crop"})

    df = add_kerala_context_to_crop_df(df, seed=seed)
    df = df.dropna(subset=["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "crop"])

    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["crop"], random_state=seed)

    train_path = out_dir / "crop_recommendation_train.csv"
    test_path = out_dir / "crop_recommendation_test.csv"
    df_all_path = out_dir / "crop_recommendation_kerala_augmented.csv"

    df.to_csv(df_all_path, index=False)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    return SplitData(train=train_df, test=test_df)


# ---------------------------------------------------------------------------
# Model 3 — Yield predictor preprocessing
# ---------------------------------------------------------------------------

KERALA_YIELD_DISTRICTS = [
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


def _canonical_district(name: str) -> str:
    n = (name or "").strip().title()
    for d in KERALA_YIELD_DISTRICTS:
        if d.lower() in n.lower():
            return d
    # fallback to best guess if not matched
    return KERALA_YIELD_DISTRICTS[0]


def normalize_yield_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map Kaggle (and similar) crop yield datasets to a standard schema:
    - state, district, crop_name, year, area_hectares, production_tons_per_hectare
    """
    d = df.copy()
    cols = {c.lower().strip(): c for c in d.columns}

    def pick(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    state_col = pick("state_name", "state")
    district_col = pick("district_name", "district")
    crop_col = pick("crop", "crop_name", "cropname")
    year_col = pick("crop_year", "year")
    area_col = pick("area", "area_hectares", "area_in_hectares")
    prod_col = pick("production", "production_in_tonnes", "production_tonnes")

    if state_col is None or crop_col is None or year_col is None or area_col is None or prod_col is None:
        raise ValueError("Could not map required columns; inspect raw columns and update mapping.")

    if district_col is None:
        d["district_name"] = "Unknown"
        district_col = "district_name"

    out = pd.DataFrame(
        {
            "state": d[state_col].astype(str),
            "district": d[district_col].astype(str),
            "crop_name": d[crop_col].astype(str),
            "year": pd.to_numeric(d[year_col], errors="coerce"),
            "area_hectares": pd.to_numeric(d[area_col], errors="coerce"),
            "production_tonnes_total": pd.to_numeric(d[prod_col], errors="coerce"),
        }
    )
    out["production_tons_per_hectare"] = out["production_tonnes_total"] / (out["area_hectares"] + 1e-6)
    return out


def preprocess_yield_data(
    raw_csv: Path,
    out_dir: Path,
    seed: int = 7,
    drop_outliers: bool = True,
) -> pd.DataFrame:
    """
    Full preprocessing for Kerala yield modelling:
    - normalize schema
    - filter Kerala rows
    - canonicalize districts
    - add synthetic weather / input-use features
    - compute lag features
    - handle missing values with district-wise medians
    - optional IQR-based outlier removal on target
    - save processed CSV
    """
    rng = _rng(seed)

    out_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(raw_csv)
    base = normalize_yield_schema(raw)

    base = base[base["state"].str.lower().str.contains("kerala")].copy()
    base = base.replace([np.inf, -np.inf], np.nan)
    base = base.dropna(subset=["year", "area_hectares", "production_tons_per_hectare"])

    base["district"] = base["district"].apply(_canonical_district)

    # approximate annual rainfall and temp from priors (mm/year, °C)
    annual_rain = base["district"].map(lambda d: DISTRICT_CLIMATE_PRIORS.get(d, {}) .get("avg_rainfall", 230.0))
    annual_temp = base["district"].map(lambda d: DISTRICT_CLIMATE_PRIORS.get(d, {}) .get("avg_temp", 27.5))

    base["annual_rainfall_mm"] = (annual_rain * 10.0) * rng.normal(1.0, 0.08, size=len(base))
    base["avg_temperature"] = annual_temp + rng.normal(0, 1.2, size=len(base))

    base["soil_quality_index"] = rng.integers(4, 9, size=len(base)).astype(float)
    base["fertilizer_kg_per_hectare"] = rng.normal(140, 35, size=len(base)).clip(40, 280)
    base["pesticide_kg_per_hectare"] = rng.normal(3.2, 1.1, size=len(base)).clip(0.2, 8.0)
    base["irrigation_type"] = rng.choice(["Rainfed", "Canal", "Borewell", "Pond"], size=len(base), p=[0.55, 0.2, 0.15, 0.1])
    base["season"] = rng.choice(["Kharif", "Rabi", "Summer"], size=len(base), p=[0.55, 0.3, 0.15])

    base["drought_index"] = (1.0 - (base["annual_rainfall_mm"] / 3500.0)).clip(0, 1)
    base["flood_risk_score"] = ((base["annual_rainfall_mm"] / 3500.0) + (base["season"] == "Kharif").astype(int) * 0.2).clip(0, 1)

    base = base.sort_values(["district", "crop_name", "year"]).reset_index(drop=True)
    base["previous_year_yield"] = base.groupby(["district", "crop_name"])["production_tons_per_hectare"].shift(1)
    base["avg_3yr_yield"] = (
        base.groupby(["district", "crop_name"])["production_tons_per_hectare"]
        .rolling(3, min_periods=1)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )

    num_cols = [
        "year",
        "area_hectares",
        "annual_rainfall_mm",
        "avg_temperature",
        "soil_quality_index",
        "fertilizer_kg_per_hectare",
        "pesticide_kg_per_hectare",
        "previous_year_yield",
        "avg_3yr_yield",
        "drought_index",
        "flood_risk_score",
    ]

    for c in num_cols:
        base[c] = pd.to_numeric(base[c], errors="coerce")
        base[c] = base.groupby("district")[c].transform(lambda s: s.fillna(s.median()))

    if drop_outliers:
        y = base["production_tons_per_hectare"].astype(float)
        q1, q3 = y.quantile(0.25), y.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        base = base[(y >= lo) & (y <= hi)].copy()

    out_path = out_dir / "yield_kerala_processed.csv"
    base.to_csv(out_path, index=False)
    return base


# ---------------------------------------------------------------------------
# Model 4 — Pest risk synthetic data
# ---------------------------------------------------------------------------

def preprocess_pest_risk(
    out_dir: Path,
    n_rows: int = 10_000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Wrapper around generate_pest_risk_dataset:
    - generates synthetic pest risk data
    - writes CSV
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    df = generate_pest_risk_dataset(n_rows=n_rows, seed=seed)
    out_path = out_dir / "pest_risk_synthetic.csv"
    df.to_csv(out_path, index=False)
    return df


# ---------------------------------------------------------------------------
# Lightweight EDA helpers (for notebooks to call)
# ---------------------------------------------------------------------------

def summarize_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Return describe() plus missing counts for numeric columns."""
    desc = df.describe().T
    desc["missing"] = df.isna().sum()
    return desc


def summarize_categorical(df: pd.DataFrame, max_unique: int = 30) -> Dict[str, pd.Series]:
    """Return value_counts() for reasonably small-cardinality categorical columns."""
    out: Dict[str, pd.Series] = {}
    for c in df.select_dtypes(include=["object", "category"]).columns:
        vc = df[c].value_counts(dropna=False)
        if len(vc) <= max_unique:
            out[c] = vc
    return out

