"""Modèle global LightGBM : un seul modèle appris sur toutes les séries.

Cible : variation logarithmique du prix à l'horizon h, log(y[t+h]) - log(y[t]).
Variables : variations passées (1, 3, 6, 12 mois), écart à la moyenne 12 mois,
volatilité récente, mois cible, horizon, marché et produit.
Aucune fuite : à chaque origine, tout (variables, cibles, interpolation) est
calculé sur les données antérieures à l'origine uniquement.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluate import mase_scale

NAME = "lightgbm_global"
FEATURES = ["r1", "r3", "r6", "r12", "dev12", "vol6", "h", "month_target", "market", "product"]

LGBM_PARAMS = dict(
    objective="l1",
    n_estimators=300,
    learning_rate=0.03,
    num_leaves=15,
    min_child_samples=50,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    random_state=0,
    n_jobs=1,
    deterministic=True,
    force_row_wise=True,
    verbose=-1,
)


def _panel(L: pd.DataFrame) -> pd.DataFrame:
    """Format long : une ligne par (série, mois) avec les variables explicatives."""
    markets = {m: i for i, m in enumerate(sorted({c.split("|")[0] for c in L.columns}))}
    products = {p: i for i, p in enumerate(sorted({c.split("|")[1] for c in L.columns}))}
    frames = []
    for j, col in enumerate(L.columns):
        l = L[col]
        m, p = col.split("|")
        frames.append(
            pd.DataFrame(
                {
                    "L": l.to_numpy(),
                    "r1": l.diff(1).to_numpy(),
                    "r3": l.diff(3).to_numpy(),
                    "r6": l.diff(6).to_numpy(),
                    "r12": l.diff(12).to_numpy(),
                    "dev12": (l - l.rolling(12, min_periods=6).mean()).to_numpy(),
                    "vol6": l.diff().rolling(6, min_periods=4).std().to_numpy(),
                    "month": l.index.month,
                    "pos": np.arange(len(l)),
                    "sid": j,
                    "market": markets[m],
                    "product": products[p],
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _stack_horizons(panel: pd.DataFrame, horizon: int, training: bool) -> pd.DataFrame:
    parts = []
    for h in range(1, horizon + 1):
        d = panel.copy()
        d["h"] = h
        d["month_target"] = (d["month"] - 1 + h) % 12 + 1
        if training:
            d["target"] = d.groupby("sid")["L"].shift(-h) - d["L"]
            d = d.dropna(subset=["target", "L"])
        else:
            d = d[d["pos"] == d["pos"].max()].dropna(subset=["L"])
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def predict_next(L_train: pd.DataFrame, horizon: int) -> dict[str, np.ndarray]:
    """Entraîne sur L_train (log-prix, lignes = mois passés) et prévoit `horizon` mois."""
    import lightgbm as lgb

    L = L_train.interpolate(limit=3, limit_direction="both")  # n'utilise que le passé
    panel = _panel(L)
    train = _stack_horizons(panel, horizon, training=True)
    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    model.fit(train[FEATURES], train["target"], categorical_feature=["market", "product"])

    test = _stack_horizons(panel, horizon, training=False)
    delta = model.predict(test[FEATURES])
    out: dict[str, np.ndarray] = {}
    for sid, col in enumerate(L.columns):
        rows = test["sid"] == sid
        if rows.any():
            t = test[rows].sort_values("h")
            preds = np.exp(t["L"].to_numpy() + delta[t.index.to_numpy()])
            out[col] = preds
    return out


def backtest_global(
    series: dict[tuple[str, str], pd.Series],
    horizon: int = 6,
    min_train: int = 96,
    step: int = 6,
) -> pd.DataFrame:
    """Même protocole et même format de sortie que evaluate.backtest."""
    wide = pd.DataFrame({f"{m}|{c}": s for (m, c), s in series.items()}).sort_index()
    logw = np.log(wide)
    rows = []
    for t in range(min_train, len(wide) - horizon + 1, step):
        preds = predict_next(logw.iloc[:t], horizon)
        test = wide.iloc[t : t + horizon]
        for col, pred in preds.items():
            if len(pred) < horizon:
                continue
            market, commodity = col.split("|")
            scale = mase_scale(wide[col].iloc[:t])
            for h, (date, actual) in enumerate(test[col].items(), start=1):
                if np.isnan(actual):
                    continue
                rows.append(
                    dict(marche=market, produit=commodity, origine=wide.index[t - 1],
                         date=date, modele=NAME, h=h, reel=actual,
                         prevu=float(pred[h - 1]), echelle=scale)
                )
    return pd.DataFrame(rows)
