## Krishi Mitra — ML Models (Kerala Farming Assistant)

This repo contains the full **AI-powered personal farming assistant** stack:

- **ML layer** (Jupyter notebooks + saved models, Colab-ready)
- **Assistant/orchestrator** (per-farmer digital companion)
- **Backend API** (FastAPI)
- **Minimal frontend** (HTML/JS) served by the backend

### Project structure

```
ml_models/
├── 1_crop_recommender/        # Crop recommendation model + notebook/scripts
├── 2_disease_detector/        # CNN disease detector + image preprocessing
├── 3_yield_predictor/         # Yield regression model
├── 4_pest_risk_forecaster/    # Synthetic pest risk model
├── 5_chatbot/                 # RAG chatbot (Kerala farming KB)
├── assistant/                 # Orchestrator (per-farmer digital companion)
├── data/                      # Preprocessing, synthetic data, dataset downloader
├── utils/                     # Shared utilities (climate, soil, model loader)
└── evaluation/                # Unified evaluation notebook

backend/
└── main.py                    # FastAPI app exposing assistant + profile APIs

frontend/
└── index.html                 # Minimal web UI (served by FastAPI)
```

### Environment setup (Windows / local)

#### Option A: Python venv (recommended)

```bash
cd c:\krishi-mitra
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

#### Option B: Conda

```bash
cd c:\krishi-mitra
conda env create -f environment.yml
conda activate krishi-mitra-ml
```

### Datasets (download first)

Required sources:
- Crop Recommendation: `atharvaingle/crop-recommendation-dataset`
- PlantVillage: `emmarex/plantdisease`
- Rice diseases: `minhhuy2810/rice-diseases-image-dataset`
- Crop Yield India: `akshatgupta7/crop-yield-in-indian-states-dataset`

#### Colab downloader notebook (Kaggle API)

Use:
- `ml_models/data/download_datasets_colab.ipynb`

It supports:
- Uploading `kaggle.json`, **or**
- Entering Kaggle **username + API key** (it will generate `kaggle.json`).

All datasets are downloaded into your Google Drive:
- `DATA_PATH/raw/`

### Model overview + artifacts

#### Model 1 — Crop Recommendation (tabular classification)

- Notebook: `ml_models/1_crop_recommender/crop_recommender.ipynb`
- Scripts: `ml_models/1_crop_recommender/train.py`, `ml_models/1_crop_recommender/predict.py`
- Saved artifacts:
  - `ml_models/1_crop_recommender/model/crop_model.pkl`
  - `ml_models/1_crop_recommender/model/label_encoder.pkl`

**Inputs** (example):
`{district, soil_type, N, P, K, pH, rainfall, temperature, humidity, season, water_source}`

**Output**:
Top-5 crops with suitability score + reason.

#### Model 2 — Disease Detection (CNN, transfer learning)

- Notebook: `ml_models/2_disease_detector/disease_detector.ipynb`
- Scripts: `ml_models/2_disease_detector/train.py`, `ml_models/2_disease_detector/predict.py`
- Saved artifacts:
  - `ml_models/2_disease_detector/model/disease_model.h5`
  - `ml_models/2_disease_detector/model/class_names.json`

Dataset must be arranged as:

```
DATA_PATH/processed/disease_images/
  train/<class_name>/*.jpg
  val/<class_name>/*.jpg
  test/<class_name>/*.jpg
```

#### Model 3 — Yield Predictor (tabular regression)

- Notebook: `ml_models/3_yield_predictor/yield_predictor.ipynb`
- Scripts: `ml_models/3_yield_predictor/train.py`, `ml_models/3_yield_predictor/predict.py`
- Saved artifacts:
  - `ml_models/3_yield_predictor/model/yield_model.pkl`
  - `ml_models/3_yield_predictor/model/scaler.pkl`

#### Model 4 — Pest Risk Forecaster (synthetic + RF)

- Notebook: `ml_models/4_pest_risk_forecaster/pest_risk.ipynb`
- Scripts: `ml_models/4_pest_risk_forecaster/train.py`, `ml_models/4_pest_risk_forecaster/predict.py`
- Saved artifacts:
  - `ml_models/4_pest_risk_forecaster/model/pest_model.pkl`

Synthetic data generator:
- `ml_models/data/generate_synthetic_data.py`

#### Model 5 — Chatbot (RAG)

- Notebook: `ml_models/5_chatbot/chatbot.ipynb`
- Pipeline: `ml_models/5_chatbot/rag_pipeline.py`
- Knowledge base text files:
  - `ml_models/5_chatbot/knowledge_base/kerala_crops.txt`
  - `ml_models/5_chatbot/knowledge_base/pest_treatments.txt`
  - `ml_models/5_chatbot/knowledge_base/fertilizer_guide.txt`
  - `ml_models/5_chatbot/knowledge_base/govt_schemes.txt`
  - `ml_models/5_chatbot/knowledge_base/soil_guide.txt`

**Ollama (no API key)**
- If you run locally and have Ollama installed, the pipeline can call it for generation.
- In Colab (no Ollama), it falls back to a grounded context-based response.

### Assistant / orchestrator (personal farming companion)

- Module: `ml_models/assistant/orchestrator.py`
- Stores **per-farmer profiles** under `ml_models/assistant/farmers/<farmer_id>.json`
- Key dataclasses:
  - `FarmerProfile`: farmer metadata, soil/water context, language, preferences, parcels
  - `FarmerParcel`: per-parcel crop + area + sowing info
  - `Advisory`: `today`, `this_week`, `season_plan` lists
- Core functions:
  - `load_farmer_profile(farmer_id)`, `save_farmer_profile(profile)`
  - `generate_weekly_advisory(profile)` — uses **pest risk model + growth stage**
  - `generate_season_plan(profile)` — uses **crop recommender + yield predictor**
  - `chat_with_assistant(farmer_id, message, language=None)`:
    - Loads profile
    - Builds weekly + seasonal advisory
    - Calls RAG chatbot with enriched context
    - Returns a structured dict with advisory + natural language response

### Unified evaluation

- Notebook: `ml_models/evaluation/evaluation.ipynb`
- Loads saved models (local first, then Drive fallback), runs a sample Kerala input, and writes a PDF report to Drive.

### Backend API (FastAPI)

- File: `backend/main.py`
- Run locally:

```bash
cd c:\krishi-mitra
uvicorn backend.main:app --reload
```

- Main endpoints:
  - **`POST /api/farmers`**
    - Body: farmer profile (id, name, district, soil_type, water_source, parcels, preferences)
    - Behavior: creates/updates `FarmerProfile` via orchestrator.
  - **`GET /api/farmers/{farmer_id}`**
    - Returns current profile JSON (if exists).
  - **`POST /api/assistant/chat`**
    - Body: `{ farmer_id, message, language? }`
    - Behavior: calls `chat_with_assistant` and returns:
      - `assistant_response` (natural language)
      - `advisory.weekly` / `advisory.seasonal` (structured)

The FastAPI app also serves the frontend (see below) so you can just open `http://localhost:8000/`.

### Frontend (minimal web UI)

- File: `frontend/index.html`
- Served by FastAPI at `http://localhost:8000/`
- Features:
  - Simple form to **create/update a farmer profile** (id, name, district, soil, water source, main parcel).
  - Chat box to send messages to `/api/assistant/chat`.
  - Renders assistant response and shows key weekly/seasonal action items.

#### Running everything together

1. **Start backend** (from repo root):

```bash
uvicorn backend.main:app --reload
```

2. **Open the assistant UI**:

- Navigate to `http://localhost:8000/` in your browser.
- Create/update a farmer profile.
- Start chatting with your **personal farming assistant** (English or Malayalam). 

