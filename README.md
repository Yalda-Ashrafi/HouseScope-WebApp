"""
train_and_export.py
-------------------
Run this once. It reproduces the notebook's winning configuration, then writes
everything the API needs into ./artifacts.

    python train_and_export.py --data-dir /path/to/folder/containing/train.csv

What gets saved and why:

  preprocessor.joblib   fitted ColumnTransformer (scaling + one-hot)
  model.joblib          the stacked regressor, refitted on all training rows
  segmenter.joblib      StandardScaler + KMeans for the market segment view
  meta.json             feature order, winsor limits, tier edges, defaults,
                        residual spread for the confidence interval, and the
                        per-tier median prices used by the comparison chart

The defaults template matters more than it looks. A web form cannot reasonably
ask a member of the public for all 80 selected features, so the API fills the
unasked columns with the training median (numeric) or mode (categorical) and
overlays whatever the user did supply.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (GradientBoostingRegressor, RandomForestRegressor,
                              StackingRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (MinMaxScaler, OneHotEncoder, RobustScaler,
                                   StandardScaler)
from sklearn.svm import SVR

from pipeline import (ID_COL, N_FOLDS, RANDOM_STATE, SEGMENT_FEATURES, TARGET,
                      TIER_LABELS, TOP_K_FEATURES, apply_winsor, clean_data,
                      encode_ordinals, engineer_features, fit_winsor_limits,
                      remove_domain_outliers)

ART = Path("artifacts")
ART.mkdir(exist_ok=True)


def make_svr():
    """MinMax scaling before the RBF kernel, as in the notebook."""
    return make_pipeline(MinMaxScaler(), SVR(C=20, epsilon=0.008, gamma=0.0003))


def main(data_dir: Path):
    print("Loading data")
    train_raw = pd.read_csv(data_dir / "train.csv")
    print(f"  train.csv: {train_raw.shape[0]} rows x {train_raw.shape[1]} columns")

    print("Cleaning")
    train_clean, lf_map = clean_data(train_raw)

    print("Outlier handling")
    train_clean = remove_domain_outliers(train_clean)
    winsor_limits = fit_winsor_limits(train_clean)
    train_clean = apply_winsor(train_clean, winsor_limits)
    print(f"  {train_clean.shape[0]} rows retained")

    print("Feature engineering and ordinal encoding")
    train_fe = engineer_features(train_clean)
    train_enc = encode_ordinals(train_fe)

    X_all = train_enc.drop(columns=[ID_COL, TARGET])
    y_all = np.log1p(train_enc[TARGET])

    num_cols = X_all.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_all.select_dtypes(exclude=[np.number]).columns.tolist()
    print(f"  {len(num_cols)} numeric, {len(cat_cols)} nominal columns")

    # ----------------------------------------------------------------------
    # Holdout split, used only to estimate the residual spread that the API
    # reports as a confidence interval. The deployed model is refitted on all
    # rows afterwards.
    # ----------------------------------------------------------------------
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_all, y_all, test_size=0.20, random_state=RANDOM_STATE,
        stratify=pd.qcut(y_all, q=5, labels=False),
    )

    def build_preprocessor():
        return ColumnTransformer(
            transformers=[
                ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                                  ("scale", RobustScaler())]), num_cols),
                ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                  ("ohe", OneHotEncoder(handle_unknown="ignore",
                                                        sparse_output=False,
                                                        min_frequency=5))]), cat_cols),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )

    print("Fitting preprocessor on the training split")
    pre_tr = build_preprocessor().fit(X_tr)
    names_tr = list(pre_tr.get_feature_names_out())
    Xtr_full = pd.DataFrame(pre_tr.transform(X_tr), columns=names_tr, index=X_tr.index)
    Xte_full = pd.DataFrame(pre_tr.transform(X_te), columns=names_tr, index=X_te.index)

    sel = GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                                    subsample=0.8, random_state=RANDOM_STATE)
    sel.fit(Xtr_full, y_tr)
    selected_tr = (pd.Series(sel.feature_importances_, index=names_tr)
                   .sort_values(ascending=False).head(TOP_K_FEATURES).index.tolist())

    print("Estimating residual spread on the holdout")
    probe = StackingRegressor(
        estimators=[
            ("gb", GradientBoostingRegressor(n_estimators=600, max_depth=3,
                                             learning_rate=0.03, subsample=0.8,
                                             random_state=RANDOM_STATE)),
            ("rf", RandomForestRegressor(n_estimators=500, max_features=0.5,
                                         n_jobs=-1, random_state=RANDOM_STATE)),
            ("svr", make_svr()),
            ("lasso", Lasso(alpha=0.0005, max_iter=5000, random_state=RANDOM_STATE)),
        ],
        final_estimator=Ridge(alpha=1.0),
        cv=KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1,
    )
    probe.fit(Xtr_full[selected_tr], y_tr)
    resid_log = y_te.values - probe.predict(Xte_full[selected_tr])
    residual_std_log = float(np.std(resid_log))
    print(f"  residual sd in log space: {residual_std_log:.4f}")

    # ----------------------------------------------------------------------
    # Final refit on every training row
    # ----------------------------------------------------------------------
    print("Refitting on all training rows")
    final_pre = build_preprocessor().fit(X_all)
    final_names = list(final_pre.get_feature_names_out())
    X_all_t = pd.DataFrame(final_pre.transform(X_all), columns=final_names)

    sel_full = GradientBoostingRegressor(n_estimators=300, max_depth=4,
                                         learning_rate=0.05, subsample=0.8,
                                         random_state=RANDOM_STATE).fit(X_all_t, y_all)
    final_features = (pd.Series(sel_full.feature_importances_, index=final_names)
                      .sort_values(ascending=False).head(TOP_K_FEATURES).index.tolist())

    final_model = StackingRegressor(
        estimators=[
            ("gb", GradientBoostingRegressor(n_estimators=600, max_depth=3,
                                             learning_rate=0.03, subsample=0.8,
                                             random_state=RANDOM_STATE)),
            ("rf", RandomForestRegressor(n_estimators=500, max_features=0.5,
                                         n_jobs=-1, random_state=RANDOM_STATE)),
            ("svr", make_svr()),
            ("lasso", Lasso(alpha=0.0005, max_iter=5000, random_state=RANDOM_STATE)),
        ],
        final_estimator=Ridge(alpha=1.0),
        cv=KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1,
    )
    final_model.fit(X_all_t[final_features], y_all)
    print(f"  fitted on {X_all_t.shape[0]} rows x {len(final_features)} features")

    # ----------------------------------------------------------------------
    # Market tiers, from training price terciles
    # ----------------------------------------------------------------------
    prices = np.expm1(y_all.values)
    edges = np.quantile(prices, [0, 1 / 3, 2 / 3, 1.0])
    tier_edges = [float(-np.inf), float(edges[1]), float(edges[2]), float(np.inf)]
    tier_series = pd.cut(prices, bins=tier_edges, labels=TIER_LABELS)
    tier_stats = {}
    for lab in TIER_LABELS:
        sub = prices[tier_series == lab]
        tier_stats[lab] = {
            "count": int(sub.size),
            "mean": float(np.mean(sub)),
            "median": float(np.median(sub)),
            "p25": float(np.percentile(sub, 25)),
            "p75": float(np.percentile(sub, 75)),
        }
    print(f"  tier cuts at ${tier_edges[1]:,.0f} and ${tier_edges[2]:,.0f}")

    # ----------------------------------------------------------------------
    # Segmentation model (the notebook's K-Means on five raw features)
    # ----------------------------------------------------------------------
    seg_raw = train_fe[SEGMENT_FEATURES].copy()
    seg_scaler = StandardScaler().fit(seg_raw)
    km = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10)
    seg_labels = km.fit_predict(seg_scaler.transform(seg_raw))

    # Order clusters by median price so cluster ids map onto readable names
    seg_med = (pd.DataFrame({"cluster": seg_labels, "price": train_fe[TARGET].values})
               .groupby("cluster")["price"].median().sort_values())
    cluster_to_tier = {int(c): TIER_LABELS[i] for i, c in enumerate(seg_med.index)}

    # ----------------------------------------------------------------------
    # Defaults template for unasked form fields
    # ----------------------------------------------------------------------
    raw_defaults = {}
    raw_source = train_clean.drop(columns=[ID_COL, TARGET])
    for c in raw_source.columns:
        if pd.api.types.is_numeric_dtype(raw_source[c]):
            raw_defaults[c] = float(raw_source[c].median())
        else:
            raw_defaults[c] = str(raw_source[c].mode()[0])

    neighborhoods = sorted(train_clean["Neighborhood"].unique().tolist())

    print("Saving artifacts")
    joblib.dump(final_pre, ART / "preprocessor.joblib")
    joblib.dump(final_model, ART / "model.joblib")
    joblib.dump({"scaler": seg_scaler, "kmeans": km}, ART / "segmenter.joblib")

    meta = {
        "final_features": final_features,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "winsor_limits": winsor_limits,
        "lotfrontage_map": {str(k): float(v) for k, v in lf_map.dropna().items()},
        "tier_edges": tier_edges,
        "tier_stats": tier_stats,
        "cluster_to_tier": cluster_to_tier,
        "residual_std_log": residual_std_log,
        "raw_defaults": raw_defaults,
        "neighborhoods": neighborhoods,
        "n_training_rows": int(X_all.shape[0]),
        "overall_median_price": float(np.median(prices)),
    }
    with open(ART / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Done. Artifacts written to {ART.resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("."),
                    help="folder containing train.csv")
    args = ap.parse_args()
    main(args.data_dir)