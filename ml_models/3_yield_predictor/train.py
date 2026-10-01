from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor


def train(csv_path: Path, out_dir: Path) -> None:
    """
    Expects a pre-merged Kerala-focused CSV with at least these columns:
      crop_name, district, year, area_hectares, annual_rainfall_mm, avg_temperature,
      soil_quality_index, fertilizer_kg_per_hectare, pesticide_kg_per_hectare,
      irrigation_type, season, production_tons_per_hectare
    """
    df = pd.read_csv(csv_path)

    required = {
        "crop_name",
        "district",
        "year",
        "area_hectares",
        "annual_rainfall_mm",
        "avg_temperature",
        "soil_quality_index",
        "fertilizer_kg_per_hectare",
        "pesticide_kg_per_hectare",
        "irrigation_type",
        "season",
        "production_tons_per_hectare",
    }
    missing = sorted(list(required - set(df.columns)))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    feature_cols = [
        "crop_name",
        "district",
        "year",
        "area_hectares",
        "annual_rainfall_mm",
        "avg_temperature",
        "soil_quality_index",
        "fertilizer_kg_per_hectare",
        "pesticide_kg_per_hectare",
        "irrigation_type",
        "season",
    ]

    X = df[feature_cols].copy()
    y = df["production_tons_per_hectare"].astype(float).copy()

    cat_cols = ["crop_name", "district", "irrigation_type", "season"]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    scaler = StandardScaler()
    pre = ColumnTransformer(
        [
            ("num", scaler, num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )

    pipe = Pipeline([("pre", pre), ("reg", model)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    mae = float(mean_absolute_error(y_test, pred))
    rmse = float(mean_squared_error(y_test, pred, squared=False))
    r2 = float(r2_score(y_test, pred))

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out_dir / "yield_model.pkl")
    joblib.dump(scaler, out_dir / "scaler.pkl")
    (out_dir / "metrics.json").write_text(json.dumps({"MAE": mae, "RMSE": rmse, "R2": r2}, indent=2), encoding="utf-8")

    print(f"Saved: {out_dir / 'yield_model.pkl'}")
    print(f"Saved: {out_dir / 'scaler.pkl'}")
    print(f"MAE={mae:.4f} RMSE={rmse:.4f} R2={r2:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="Pre-processed yield CSV")
    ap.add_argument("--out_dir", type=str, default=str(Path(__file__).parent / "model"))
    args = ap.parse_args()

    train(Path(args.csv), Path(args.out_dir))


if __name__ == "__main__":
    main()

