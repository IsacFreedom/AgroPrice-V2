"""Intervalles de prévision par calibration conforme sur les erreurs passées.

Pour chaque modèle et horizon, l'intervalle à l'origine t utilise les quantiles des
erreurs logarithmiques log(réel/prévu) déjà CONNUES à t (cible <= date d'origine).
Fonctionne avec n'importe quel modèle et n'utilise jamais le futur.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _log_err(df: pd.DataFrame) -> pd.Series:
    return np.log(df["reel"]) - np.log(df["prevu"].clip(lower=1.0))


def add_intervals(res: pd.DataFrame, alpha: float = 0.2, min_cal: int = 60) -> pd.DataFrame:
    """Ajoute les colonnes bas/haut (intervalle à 1-alpha) ; ignore les origines sans calibration."""
    r = res.copy()
    r["e"] = _log_err(r)
    out = []
    for (_, _), g in r.groupby(["modele", "h"]):
        for o in sorted(g["origine"].unique()):
            cal = g[g["date"] <= o]["e"].dropna()
            if len(cal) < min_cal:
                continue
            lo, hi = np.quantile(cal, [alpha / 2, 1 - alpha / 2])
            cur = g[g["origine"] == o].copy()
            p = cur["prevu"].clip(lower=1.0)
            cur["bas"], cur["haut"] = p * np.exp(lo), p * np.exp(hi)
            out.append(cur)
    return pd.concat(out, ignore_index=True).drop(columns="e")


def coverage_table(ri: pd.DataFrame) -> pd.DataFrame:
    """Couverture réelle et largeur relative moyenne par modèle et horizon."""
    d = ri.copy()
    d["dedans"] = (d["reel"] >= d["bas"]) & (d["reel"] <= d["haut"])
    d["largeur_rel"] = (d["haut"] - d["bas"]) / d["prevu"].clip(lower=1.0)
    return (
        d.groupby(["modele", "h"])
        .agg(couverture=("dedans", "mean"), largeur_rel=("largeur_rel", "mean"), n=("dedans", "size"))
        .round(3)
        .reset_index()
    )


def calibration_quantiles(res: pd.DataFrame, alpha: float = 0.2) -> dict[tuple[str, int], tuple[float, float]]:
    """Quantiles d'erreur calculés sur TOUT l'historique, pour les prévisions les plus récentes."""
    r = res.copy()
    r["e"] = _log_err(r)
    return {
        (m, int(h)): tuple(np.quantile(g["e"].dropna(), [alpha / 2, 1 - alpha / 2]))
        for (m, h), g in r.groupby(["modele", "h"])
    }
