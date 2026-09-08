"""
app.py
------
FastAPI service for the housing valuation tool.

    uvicorn app:app --reload --port 8000

Endpoints
    GET  /api/health     liveness plus whether artifacts loaded
    GET  /api/meta       form options, tier statistics, training row count
    POST /api/predict    property attributes in, price and tier out

Design notes worth keeping in mind if you extend this:

*   The model was trained on log1p(SalePrice), so every prediction is inverted
    with expm1 before it leaves the service. Forget that and the numbers come
    back around 12 instead of around 180,000.
*   The confidence interval is a real quantity, not decoration. It comes from
    the standard deviation of holdout residuals in log space, so the interval
    is asymmetric in dollars, which is correct for a log-transformed target.
*   Unasked fields are filled from the training median or mode. That is an
    honest limitation and the response says so in `assumptions`.
"""

from __future__ import annotations

import json
import io
from pathlib import Path
from typing import Literal, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from pipeline import (SEGMENT_FEATURES, TIER_DISPLAY, apply_winsor, assign_tier,
                      clean_data, encode_ordinals, engineer_features)

ART = Path(__file__).resolve().parent / "artifacts"

app = FastAPI(
    title="Ames Housing Valuation API",
    description="Free property valuation and market tier assignment.",
    version="1.0.0",
)

# During development the frontend is usually served from a different port.
# Narrow this before any public deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------
# Artifact loading
# --------------------------------------------------------------------------

STATE: dict = {"ready": False, "error": None}


def load_artifacts():
    try:
        STATE["preprocessor"] = joblib.load(ART / "preprocessor.joblib")
        STATE["model"] = joblib.load(ART / "model.joblib")
        seg = joblib.load(ART / "segmenter.joblib")
        STATE["seg_scaler"] = seg["scaler"]
        STATE["kmeans"] = seg["kmeans"]
        with open(ART / "meta.json") as f:
            STATE["meta"] = json.load(f)
        with open(ART / "segment_meta.json") as f:
            STATE["segment_meta"] = json.load(f)
        STATE["ready"] = True
        STATE["error"] = None
    except FileNotFoundError as exc:
        STATE["ready"] = False
        STATE["error"] = (
            f"Artifact missing: {exc.filename}. "
            "Run `python train_and_export.py --data-dir <folder with train.csv>` first."
        )


@app.on_event("startup")
def _startup():
    load_artifacts()
    print("Artifacts loaded" if STATE["ready"] else f"NOT READY: {STATE['error']}")


# --------------------------------------------------------------------------
# Request schema
# --------------------------------------------------------------------------

QualityLevel = Literal["Ex", "Gd", "TA", "Fa", "Po"]


class PropertyInput(BaseModel):
    """The subset of attributes a member of the public can reasonably supply."""

    # Size
    GrLivArea: float = Field(..., ge=200, le=6000,
                             description="Above grade living area, square feet")
    TotalBsmtSF: float = Field(0, ge=0, le=6000, description="Total basement area")
    FirstFlrSF: float = Field(..., ge=200, le=5000, alias="1stFlrSF",
                              description="First floor area")
    SecondFlrSF: float = Field(0, ge=0, le=3000, alias="2ndFlrSF",
                               description="Second floor area")
    LotArea: float = Field(..., ge=500, le=100000, description="Lot size")

    # Quality
    OverallQual: int = Field(..., ge=1, le=10, description="Overall material and finish")
    OverallCond: int = Field(5, ge=1, le=10, description="Overall condition")
    ExterQual: QualityLevel = "TA"
    KitchenQual: QualityLevel = "TA"
    BsmtQual: QualityLevel = "TA"
    HeatingQC: QualityLevel = "TA"

    # Age
    YearBuilt: int = Field(..., ge=1870, le=2030)
    YearRemodAdd: Optional[int] = Field(None, ge=1870, le=2030)
    YrSold: int = Field(2010, ge=2006, le=2030)

    # Rooms
    FullBath: int = Field(1, ge=0, le=6)
    HalfBath: int = Field(0, ge=0, le=4)
    BsmtFullBath: int = Field(0, ge=0, le=4)
    BedroomAbvGr: int = Field(3, ge=0, le=12)
    TotRmsAbvGrd: int = Field(6, ge=1, le=20)
    Fireplaces: int = Field(0, ge=0, le=5)

    # Garage
    GarageCars: int = Field(1, ge=0, le=5)
    GarageArea: float = Field(400, ge=0, le=2000)

    # Location and services
    Neighborhood: str = "NAmes"
    CentralAir: Literal["Y", "N"] = "Y"

    model_config = {"populate_by_name": True}

    @field_validator("YearRemodAdd")
    @classmethod
    def remod_not_before_build(cls, v, info):
        built = info.data.get("YearBuilt")
        if v is not None and built is not None and v < built:
            raise ValueError("YearRemodAdd cannot be earlier than YearBuilt")
        return v

    @field_validator("YrSold")
    @classmethod
    def sold_not_before_build(cls, v, info):
        built = info.data.get("YearBuilt")
        if built is not None and v < built:
            raise ValueError("YrSold cannot be earlier than YearBuilt")
        return v


class PredictionResponse(BaseModel):
    predicted_price: float
    price_low: float
    price_high: float
    confidence_level: float
    tier_code: str
    tier: str
    tier_source: str
    segment_tier: str
    tier_median_price: float
    tier_mean_price: float
    tier_p25: float
    tier_p75: float
    overall_median_price: float
    price_vs_tier_median_pct: float
    assumptions: list[str]
    model_version: str


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------

def build_raw_row(payload: PropertyInput) -> pd.DataFrame:
    """Overlay the supplied fields onto the training defaults template."""
    meta = STATE["meta"]
    row = dict(meta["raw_defaults"])

    supplied = payload.model_dump(by_alias=True)
    if supplied.get("YearRemodAdd") is None:
        supplied["YearRemodAdd"] = supplied["YearBuilt"]

    for k, v in supplied.items():
        if v is not None:
            row[k] = v

    # LotFrontage is neighbourhood dependent; reuse the training median map
    lf = meta["lotfrontage_map"].get(str(row.get("Neighborhood")))
    if lf is not None:
        row["LotFrontage"] = lf

    df = pd.DataFrame([row])
    df["MSSubClass"] = df["MSSubClass"].astype(str)
    df["MoSold"] = df["MoSold"].astype(str)
    return df


def predict_one(payload: PropertyInput) -> dict:
    meta = STATE["meta"]

    df = build_raw_row(payload)
    df = apply_winsor(df, {k: tuple(v) for k, v in meta["winsor_limits"].items()})
    df = engineer_features(df)

    # Segment assignment uses the raw engineered features, before encoding
    seg_vec = df[SEGMENT_FEATURES].astype(float)
    cluster = int(STATE["kmeans"].predict(STATE["seg_scaler"].transform(seg_vec))[0])
    c2t = meta["cluster_to_tier"]
    segment_tier = c2t.get(str(cluster), c2t.get(cluster, "Medium"))

    df = encode_ordinals(df)

    # Align to the training column order before transforming
    expected = meta["num_cols"] + meta["cat_cols"]
    for c in expected:
        if c not in df.columns:
            df[c] = meta["raw_defaults"].get(c, 0)
    df = df[expected]

    Xt = STATE["preprocessor"].transform(df)
    names = list(STATE["preprocessor"].get_feature_names_out())
    Xt = pd.DataFrame(Xt, columns=names)[meta["final_features"]]

    pred_log = float(STATE["model"].predict(Xt)[0])
    price = float(np.expm1(pred_log))

    # 95 per cent interval from holdout residual spread, inverted from log space
    sd = meta["residual_std_log"]
    low = float(np.expm1(pred_log - 1.96 * sd))
    high = float(np.expm1(pred_log + 1.96 * sd))

    tier_code = assign_tier(price, meta["tier_edges"])
    stats = meta["tier_stats"][tier_code]

    supplied_keys = set(payload.model_dump(by_alias=True, exclude_none=True).keys())
    n_defaulted = len([c for c in meta["raw_defaults"] if c not in supplied_keys])

    return {
        "predicted_price": round(price, 2),
        "price_low": round(low, 2),
        "price_high": round(high, 2),
        "confidence_level": 0.95,
        "tier_code": tier_code,
        "tier": TIER_DISPLAY[tier_code],
        "tier_source": "predicted price against training terciles",
        "segment_tier": TIER_DISPLAY[segment_tier],
        "tier_median_price": round(stats["median"], 2),
        "tier_mean_price": round(stats["mean"], 2),
        "tier_p25": round(stats["p25"], 2),
        "tier_p75": round(stats["p75"], 2),
        "overall_median_price": round(meta["overall_median_price"], 2),
        "price_vs_tier_median_pct": round(
            (price - stats["median"]) / stats["median"] * 100, 1),
        "assumptions": [
            f"{n_defaulted} attributes not requested by the form were set to the "
            f"training median or most common value.",
            f"Model trained on {meta['n_training_rows']} Ames, Iowa sales from 2006 to 2010. "
            "Estimates outside that market or period are not validated.",
            "This is a statistical estimate, not a professional valuation.",
        ],
        "model_version": "stacked-gb-rf-svr-lasso/ridge-meta v1.0",
    }


def segment_one(payload: PropertyInput) -> dict:
    """Assign one property to the exported K-Means market segment."""
    meta = STATE["meta"]
    segment_meta = STATE["segment_meta"]
    df = build_raw_row(payload)
    df = apply_winsor(df, {k: tuple(v) for k, v in meta["winsor_limits"].items()})
    engineered = engineer_features(df)
    features = engineered[SEGMENT_FEATURES].astype(float)
    cluster = int(
        STATE["kmeans"].predict(STATE["seg_scaler"].transform(features))[0]
    )
    tier_code = segment_meta["cluster_to_tier"].get(str(cluster), "Medium")
    stats = segment_meta["segment_stats"][tier_code]
    return {
        "cluster": cluster,
        "tier_code": tier_code,
        "tier": TIER_DISPLAY[tier_code],
        "count": stats["count"],
        "mean_price": round(stats["mean_price"], 2),
        "median_price": round(stats["median_price"], 2),
        "insight": (
            f"{tier_code} segment properties average "
            f"{stats['mean_living_area']:,.0f} sq ft and quality "
            f"{stats['mean_overall_quality']:.1f}/10."
        ),
    }


def transform_for_prediction(df: pd.DataFrame) -> np.ndarray:
    """Apply the exported notebook pipeline to a complete feature frame."""
    meta = STATE["meta"]
    cleaned, _ = clean_data(
        df,
        lotfrontage_map=pd.Series(meta["lotfrontage_map"]),
        drop_duplicates=False,
    )
    cleaned = apply_winsor(
        cleaned, {k: tuple(v) for k, v in meta["winsor_limits"].items()}
    )
    encoded = encode_ordinals(engineer_features(cleaned))
    expected = meta["num_cols"] + meta["cat_cols"]
    missing = [column for column in expected if column not in encoded.columns]
    if missing:
        raise ValueError(f"Missing transformed columns: {', '.join(missing)}")
    transformed = STATE["preprocessor"].transform(encoded[expected])
    names = list(STATE["preprocessor"].get_feature_names_out())
    transformed = pd.DataFrame(transformed, columns=names)[meta["final_features"]]
    return np.expm1(STATE["model"].predict(transformed))


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

@app.get("/api/health")
@app.get("/health")
def health():
    return {"status": "ok" if STATE["ready"] else "not_ready", "detail": STATE["error"]}


@app.get("/api/meta")
@app.get("/meta")
def get_meta():
    if not STATE["ready"]:
        raise HTTPException(status_code=503, detail=STATE["error"])
    m = STATE["meta"]
    return {
        "neighborhoods": m["neighborhoods"],
        "quality_levels": [
            {"value": "Ex", "label": "Excellent"},
            {"value": "Gd", "label": "Good"},
            {"value": "TA", "label": "Typical / Average"},
            {"value": "Fa", "label": "Fair"},
            {"value": "Po", "label": "Poor"},
        ],
        "tier_edges": [m["tier_edges"][1], m["tier_edges"][2]],
        "tier_stats": {TIER_DISPLAY[k]: v for k, v in m["tier_stats"].items()},
        "n_training_rows": m["n_training_rows"],
        "overall_median_price": m["overall_median_price"],
    }


@app.get("/api/segmentation")
@app.get("/segmentation")
def get_segmentation():
    if not STATE["ready"]:
        raise HTTPException(status_code=503, detail=STATE["error"])
    return STATE["segment_meta"]


@app.post("/api/segment")
@app.post("/segment")
def segment(payload: PropertyInput):
    if not STATE["ready"]:
        raise HTTPException(status_code=503, detail=STATE["error"])
    try:
        return segment_one(payload)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=400, detail=f"Could not assign this property: {exc}"
        ) from exc


@app.post("/api/predict", response_model=PredictionResponse)
@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PropertyInput):
    if not STATE["ready"]:
        raise HTTPException(status_code=503, detail=STATE["error"])
    try:
        return predict_one(payload)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400,
                            detail=f"Could not score this property: {exc}") from exc


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not STATE["ready"]:
        raise HTTPException(status_code=503, detail=STATE["error"])
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")
    try:
        data = await file.read()
        frame = pd.read_csv(io.BytesIO(data))
        required = STATE["meta"]["required_schema"]
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Uploaded dataset does not match required columns. "
                    "Please align your dataset with the training schema. "
                    f"Missing columns: {', '.join(missing)}"
                ),
            )
        # Ignore harmless metadata or target columns, such as Id and SalePrice.
        frame = frame[required]
        if frame.empty:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Uploaded CSV has the correct columns but no data rows. "
                    "Add at least one property row before uploading."
                ),
            )
        predictions = transform_for_prediction(frame)
        return {
            "predictions": [
                {"row": index, "predicted_price": round(float(price), 2)}
                for index, price in enumerate(predictions)
            ]
        }
    except HTTPException:
        raise
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Could not process CSV: {exc}") from exc


@app.get("/template")
def template(example: bool = False):
    if not STATE["ready"]:
        raise HTTPException(status_code=503, detail=STATE["error"])
    columns = STATE["meta"]["required_schema"]
    if example:
        defaults = STATE["meta"]["raw_defaults"]
        row = {column: defaults.get(column) for column in columns}
        content = pd.DataFrame([row], columns=columns).to_csv(index=False)
        filename = "house_example_template.csv"
    else:
        content = pd.DataFrame(columns=columns).to_csv(index=False)
        filename = "house_template.csv"
    return StreamingResponse(
        iter([content]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )