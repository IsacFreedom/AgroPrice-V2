"""Prévisions à partir de la dernière donnée disponible, avec intervalles calibrés."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .global_model import predict_next
from .intervals import calibration_quantiles
from .models import ETS, Naive

MODELS = ("naif", "ets", "lightgbm_global")


def latest_forecasts(series: dict, res: pd.DataFrame, horizon: int = 6, alpha: float = 0.2) -> pd.DataFrame:
    """Prévisions h=1..horizon depuis la fin du calendrier commun, pour naïf, ETS, LightGBM."""
    end = max(s.index.max() for s in series.values())
    series = {k: s.reindex(pd.date_range(s.index.min(), end, freq="MS")) for k, s in series.items()}
    q = calibration_quantiles(res, alpha)
    wide_log = np.log(pd.DataFrame({f"{m}|{c}": s for (m, c), s in series.items()}).sort_index())
    glob = predict_next(wide_log, horizon)
    dates = pd.date_range(end, periods=horizon + 1, freq="MS")[1:]
    rows = []
    for (m, c), y in series.items():
        preds = {"naif": Naive().fit(y).predict(horizon), "ets": ETS().fit(y).predict(horizon)}
        if f"{m}|{c}" in glob:
            preds["lightgbm_global"] = glob[f"{m}|{c}"]
        last_obs = y.dropna().index[-1]
        for model, p in preds.items():
            for h, (d, v) in enumerate(zip(dates, p), start=1):
                lo, hi = q[(model, h)]
                v = max(float(v), 1.0)
                rows.append(dict(marche=m, produit=c, modele=model, h=h, date=d, prevu=v,
                                 bas=v * np.exp(lo), haut=v * np.exp(hi), derniere_obs=last_obs))
    df = pd.DataFrame(rows)
    df[["prevu", "bas", "haut"]] = df[["prevu", "bas", "haut"]].round(1)
    return df
