from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


KERALA_DISTRICTS = [
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
SOIL_TYPES = ["Laterite", "Alluvial", "Sandy_Loam", "Red_Loam", "Forest"]
SEASONS = ["Kharif", "Rabi", "Summer"]
WATER_SOURCES = ["Rainfall", "Canal", "Borewell", "Pond", "River"]

DISTRICT_PRIORS = {
    "Palakkad": {"temp": 29.0, "rain": 180.0, "hum": 78.0},
    "Thrissur": {"temp": 28.0, "rain": 210.0, "hum": 82.0},
    "Wayanad": {"temp": 24.5, "rain": 260.0, "hum": 88.0},
    "Malappuram": {"temp": 27.5, "rain": 230.0, "hum": 84.0},
    "Kannur": {"temp": 27.0, "rain": 240.0, "hum": 86.0},
    "Kozhikode": {"temp": 27.0, "rain": 245.0, "hum": 87.0},
    "Idukki": {"temp": 23.5, "rain": 250.0, "hum": 86.0},
    "Ernakulam": {"temp": 28.0, "rain": 220.0, "hum": 85.0},
    "Alappuzha": {"temp": 28.2, "rain": 235.0, "hum": 88.0},
    "Kottayam": {"temp": 27.2, "rain": 240.0, "hum": 87.0},
}


def _sample_season(month: int) -> str:
    if month in (6, 7, 8, 9):
        return "Kharif"
    if month in (10, 11, 12, 1, 2):
        return "Rabi"
    return "Summer"


def add_kerala_context(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()

    districts = rng.choice(KERALA_DISTRICTS, size=len(out))
    soils = rng.choice(SOIL_TYPES, size=len(out), p=[0.38, 0.22, 0.12, 0.18, 0.10])
    months = rng.integers(1, 13, size=len(out))
    seasons = np.array([_sample_season(int(m)) for m in months])

    water = []
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

    temp_prior = np.array([DISTRICT_PRIORS[d]["temp"] for d in out["district"]])
    rain_prior = np.array([DISTRICT_PRIORS[d]["rain"] for d in out["district"]])
    hum_prior = np.array([DISTRICT_PRIORS[d]["hum"] for d in out["district"]])

    out["temperature"] = 0.85 * out["temperature"] + 0.15 * temp_prior
    out["rainfall"] = 0.85 * out["rainfall"] + 0.15 * rain_prior
    out["humidity"] = 0.85 * out["humidity"] + 0.15 * hum_prior

    return out


def train(csv_path: Path, out_dir: Path, tune: bool = True) -> None:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    if "label" in df.columns:
        df = df.rename(columns={"label": "crop"})

    df = add_kerala_context(df, seed=42)

    X = df[["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "district", "soil_type", "season", "water_source"]]
    y = df["crop"].astype(str)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    num_cols = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    cat_cols = ["district", "soil_type", "season", "water_source"]

    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    base = Pipeline(
        [
            ("pre", pre),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=500,
                    random_state=42,
                    n_jobs=-1,
                    class_weight="balanced_subsample",
                ),
            ),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    model = base
    if tune:
        grid = GridSearchCV(
            base,
            param_grid={
                "clf__n_estimators": [300, 500],
                "clf__max_depth": [None, 18, 26],
                "clf__min_samples_split": [2, 5],
                "clf__min_samples_leaf": [1, 2],
                "clf__max_features": ["sqrt", 0.6],
            },
            cv=3,
            n_jobs=-1,
            verbose=1,
            scoring="accuracy",
        )
        grid.fit(X_train, y_train)
        model = grid.best_estimator_
    else:
        model.fit(X_train, y_train)

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "crop_model.pkl")
    joblib.dump(le, out_dir / "label_encoder.pkl")

    print(f"Saved: {out_dir / 'crop_model.pkl'}")
    print(f"Saved: {out_dir / 'label_encoder.pkl'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="Path to Crop_recommendation.csv")
    ap.add_argument(
        "--out_dir",
        type=str,
        default=str(Path(__file__).parent / "model"),
        help="Output directory for model files",
    )
    ap.add_argument("--no_tune", action="store_true", help="Skip GridSearchCV")
    args = ap.parse_args()

    train(Path(args.csv), Path(args.out_dir), tune=not args.no_tune)


if __name__ == "__main__":
    main()

