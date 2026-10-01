from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ml_models.data.generate_synthetic_data import generate_pest_risk_dataset


def train(out_dir: Path, n_rows: int = 10_000, seed: int = 42) -> None:
    df = generate_pest_risk_dataset(n_rows=n_rows, seed=seed)

    feature_cols = [
        "district",
        "crop",
        "month",
        "avg_temperature",
        "humidity_percent",
        "rainfall_last_7days_mm",
        "wind_speed_kmh",
        "previous_pest_incident",
        "crop_growth_stage",
        "cloudy",
    ]

    X = df[feature_cols].copy()
    y = df["pest_risk_level"].astype(str).copy()

    cat_cols = ["district", "crop", "crop_growth_stage"]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    model = Pipeline(
        [
            ("pre", pre),
            ("clf", RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1, class_weight="balanced")),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    report = classification_report(y_test, pred, output_dict=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "pest_model.pkl")
    (out_dir / "metrics.json").write_text(json.dumps({"classification_report": report}, indent=2), encoding="utf-8")
    print(f"Saved: {out_dir / 'pest_model.pkl'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default=str(Path(__file__).parent / "model"))
    ap.add_argument("--rows", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    train(Path(args.out_dir), n_rows=args.rows, seed=args.seed)


if __name__ == "__main__":
    main()

