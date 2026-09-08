"""
pipeline.py
-----------
Preprocessing logic lifted directly from the research notebook.

Both the training script and the API import from this module, which is the whole
point: if serving used a re-typed copy of the transforms, the two could drift apart
and predictions would quietly become wrong.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "SalePrice"
ID_COL = "Id"
RANDOM_STATE = 42
N_FOLDS = 5
TOP_K_FEATURES = 80

# --------------------------------------------------------------------------
# Missing value policy
# --------------------------------------------------------------------------

NA_MEANS_NONE = [
    "Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "PoolQC", "Fence", "MiscFeature", "MasVnrType",
]

NA_MEANS_ZERO = [
    "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
    "BsmtFullBath", "BsmtHalfBath", "GarageCars", "GarageArea",
]

MODE_IMPUTE = [
    "MSZoning", "Electrical", "KitchenQual", "Exterior1st", "Exterior2nd",
    "SaleType", "Functional", "Utilities",
]

WINSOR_COLS = [
    "LotArea", "LotFrontage", "MasVnrArea", "BsmtFinSF1", "TotalBsmtSF",
    "1stFlrSF", "GrLivArea", "GarageArea", "WoodDeckSF", "OpenPorchSF",
    "EnclosedPorch", "MiscVal",
]


def clean_data(
    df: pd.DataFrame,
    lotfrontage_map: pd.Series | None = None,
    drop_duplicates: bool = True,
):
    """Clean a raw frame. Returns (cleaned frame, lotfrontage map)."""
    df = df.copy()

    if drop_duplicates:
        feature_cols = [c for c in df.columns if c != ID_COL]
        df = df.drop_duplicates(subset=feature_cols).reset_index(drop=True)

    for c in NA_MEANS_NONE:
        if c in df.columns:
            df[c] = df[c].astype(object).fillna("None")
    for c in NA_MEANS_ZERO:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    if lotfrontage_map is None:
        lotfrontage_map = df.groupby("Neighborhood")["LotFrontage"].median()
    df["LotFrontage"] = df["LotFrontage"].fillna(df["Neighborhood"].map(lotfrontage_map))
    df["LotFrontage"] = df["LotFrontage"].fillna(df["LotFrontage"].median())

    if "GarageYrBlt" in df.columns:
        df["GarageYrBlt"] = df["GarageYrBlt"].fillna(df["YearBuilt"])
        df["GarageYrBlt"] = df["GarageYrBlt"].clip(upper=df["YrSold"].max())

    for c in MODE_IMPUTE:
        if c in df.columns and df[c].isna().any():
            df[c] = df[c].fillna(df[c].mode()[0])

    for c in df.columns:
        if df[c].isna().any():
            if pd.api.types.is_numeric_dtype(df[c]):
                df[c] = df[c].fillna(df[c].median())
            else:
                df[c] = df[c].astype(object).fillna("None")

    df["MSSubClass"] = df["MSSubClass"].astype(str)
    df["MoSold"] = df["MoSold"].astype(str)
    return df, lotfrontage_map


def remove_domain_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the two documented partial-sale outliers. Training data only."""
    mask = (df["GrLivArea"] > 4000) & (df[TARGET] < 300000)
    return df.loc[~mask].reset_index(drop=True)


def fit_winsor_limits(df: pd.DataFrame, cols=WINSOR_COLS, lo=0.01, hi=0.99) -> dict:
    return {c: (float(df[c].quantile(lo)), float(df[c].quantile(hi)))
            for c in cols if c in df.columns}


def apply_winsor(df: pd.DataFrame, limits: dict) -> pd.DataFrame:
    df = df.copy()
    for c, (lo, hi) in limits.items():
        if c in df.columns:
            df[c] = df[c].clip(lo, hi)
    return df


# --------------------------------------------------------------------------
# Feature engineering
# --------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    df["TotalPorchSF"] = (df["OpenPorchSF"] + df["EnclosedPorch"] + df["3SsnPorch"]
                          + df["ScreenPorch"] + df["WoodDeckSF"])
    df["TotalBath"] = (df["FullBath"] + 0.5 * df["HalfBath"]
                       + df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"])
    df["AvgRoomSize"] = df["GrLivArea"] / df["TotRmsAbvGrd"].clip(lower=1)
    df["LivingAreaRatio"] = df["GrLivArea"] / df["LotArea"].clip(lower=1)

    df["HouseAge"] = (df["YrSold"] - df["YearBuilt"]).clip(lower=0)
    df["RemodAge"] = (df["YrSold"] - df["YearRemodAdd"]).clip(lower=0)
    df["GarageAge"] = (df["YrSold"] - df["GarageYrBlt"]).clip(lower=0)
    df["IsRemodeled"] = (df["YearRemodAdd"] != df["YearBuilt"]).astype(int)
    df["IsNew"] = (df["YrSold"] == df["YearBuilt"]).astype(int)

    df["OverallGrade"] = df["OverallQual"] * df["OverallCond"]
    df["QualitySF"] = df["OverallQual"] * df["GrLivArea"]
    df["GarageScore"] = df["GarageArea"] * df["GarageCars"]

    df["HasBasement"] = (df["TotalBsmtSF"] > 0).astype(int)
    df["Has2ndFloor"] = (df["2ndFlrSF"] > 0).astype(int)
    df["HasGarage"] = (df["GarageArea"] > 0).astype(int)
    df["HasFireplace"] = (df["Fireplaces"] > 0).astype(int)
    df["HasPool"] = (df["PoolArea"] > 0).astype(int)
    return df


# --------------------------------------------------------------------------
# Ordinal encoding
# --------------------------------------------------------------------------

QUAL_MAP = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}

ORDINAL_MAPS = {
    "ExterQual": QUAL_MAP, "ExterCond": QUAL_MAP, "BsmtQual": QUAL_MAP,
    "BsmtCond": QUAL_MAP, "HeatingQC": QUAL_MAP, "KitchenQual": QUAL_MAP,
    "FireplaceQu": QUAL_MAP, "GarageQual": QUAL_MAP, "GarageCond": QUAL_MAP,
    "PoolQC": QUAL_MAP,
    "BsmtExposure": {"None": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4},
    "BsmtFinType1": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "BsmtFinType2": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "GarageFinish": {"None": 0, "Unf": 1, "RFn": 2, "Fin": 3},
    "Functional": {"Sal": 0, "Sev": 1, "Maj2": 2, "Maj1": 3, "Mod": 4,
                   "Min2": 5, "Min1": 6, "Typ": 7},
    "CentralAir": {"N": 0, "Y": 1},
    "PavedDrive": {"N": 0, "P": 1, "Y": 2},
    "LotShape": {"IR3": 0, "IR2": 1, "IR1": 2, "Reg": 3},
    "LandSlope": {"Sev": 0, "Mod": 1, "Gtl": 2},
    "Fence": {"None": 0, "MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4},
    "Street": {"Grvl": 0, "Pave": 1},
}


def encode_ordinals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, mapping in ORDINAL_MAPS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(0).astype(int)
    df["TotalQualScore"] = (df["ExterQual"] + df["BsmtQual"] + df["HeatingQC"]
                            + df["KitchenQual"] + df["GarageQual"] + df["FireplaceQu"])
    return df


# --------------------------------------------------------------------------
# Segmentation features (matches the notebook's K-Means block)
# --------------------------------------------------------------------------

SEGMENT_FEATURES = ["TotalSF", "OverallQual", "HouseAge", "TotalBath", "GrLivArea"]

TIER_LABELS = ["Low", "Medium", "High"]

# Display names. The notebook uses Low/Medium/High; the paper uses the friendlier
# set. Keep both so the API can return whichever the caller wants.
TIER_DISPLAY = {
    "Low": "Entry level",
    "Medium": "Midmarket",
    "High": "Premium",
}


def assign_tier(price: float, tier_edges) -> str:
    """Map a dollar price onto the tier the training terciles define."""
    if price < tier_edges[1]:
        return "Low"
    if price < tier_edges[2]:
        return "Medium"
    return "High"