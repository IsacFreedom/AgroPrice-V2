"""Métriques d'erreur et validation glissante (rolling origin)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .models import REGISTRY, Forecaster


def mase_scale(y_train: pd.Series) -> float:
    """Erreur absolue moyenne du modèle naïf (pas 1) sur l'apprentissage."""
    d = y_train.dropna().diff().abs().dropna()
    return float(d.mean()) if len(d) else np.nan


def backtest(
    series: dict[tuple[str, str], pd.Series],
    models: list[str] | None = None,
    horizon: int = 6,
    min_train: int = 96,
    step: int = 6,
) -> pd.DataFrame:
    """Validation à origine glissante. Une ligne par (série, origine, modèle, horizon).

    À chaque origine t, on entraîne UNIQUEMENT sur y[:t] puis on prédit t..t+h-1.
    On n'évalue que les mois où le prix réel existe.
    """
    models = models or list(REGISTRY)
    rows = []
    for (market, commodity), y in series.items():
        for t in range(min_train, len(y) - horizon + 1, step):
            train, test = y.iloc[:t], y.iloc[t : t + horizon]
            if train.tail(12).isna().mean() > 0.5:
                continue
            scale = mase_scale(train)
            for name in models:
                model: Forecaster = REGISTRY[name]()
                pred = model.fit(train).predict(horizon)
                for h, (date, actual) in enumerate(test.items(), start=1):
                    if np.isnan(actual):
                        continue
                    rows.append(
                        dict(marche=market, produit=commodity, origine=train.index[-1],
                             date=date, modele=name, h=h, reel=actual,
                             prevu=float(pred[h - 1]), echelle=scale)
                    )
    return pd.DataFrame(rows)


def summarize(res: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    """MASE et MAPE moyens par modèle et horizon (ou autre regroupement)."""
    by = by or ["modele", "h"]
    r = res.copy()
    r["erreur_abs"] = (r["reel"] - r["prevu"]).abs()
    r["mase"] = r["erreur_abs"] / r["echelle"]
    r["mape"] = r["erreur_abs"] / r["reel"] * 100
    return (
        r.groupby(by)
        .agg(MASE=("mase", "mean"), MAPE=("mape", "mean"), n=("mase", "size"))
        .round(3)
        .reset_index()
    )


def align_common(res: pd.DataFrame) -> pd.DataFrame:
    """Ne garde que les cas évalués par TOUS les modèles, pour une comparaison équitable."""
    keys = ["marche", "produit", "origine", "date", "h"]
    n_models = res["modele"].nunique()
    counts = res.groupby(keys)["modele"].transform("nunique")
    return res[counts == n_models].reset_index(drop=True)
