"""Détection de hausses anormales de prix (variation sur 3 mois inhabituelle)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def detect_spikes(series: dict, window: int = 3, lookback: int = 60, z_thresh: float = 2.0) -> pd.DataFrame:
    """Score z de la variation sur `window` mois, comparée aux `lookback` mois PRÉCÉDENTS.

    alerte = z > z_thresh (hausse inhabituelle). Statistiques calculées sur le passé seulement.
    """
    frames = []
    for (m, c), s in series.items():
        lg = np.log(s.interpolate(limit=3))
        r = lg.diff(window)
        mu = r.shift(1).rolling(lookback, min_periods=24).mean()
        sd = r.shift(1).rolling(lookback, min_periods=24).std()
        z = (r - mu) / sd
        frames.append(
            pd.DataFrame(
                {
                    "marche": m,
                    "produit": c,
                    "date": s.index,
                    "prix": s.to_numpy(),
                    "variation_3m_pct": ((np.exp(r) - 1) * 100).to_numpy(),
                    "z": z.to_numpy(),
                }
            )
        )
    out = pd.concat(frames, ignore_index=True).dropna(subset=["z"])
    out["alerte"] = out["z"] > z_thresh
    num = ["prix", "variation_3m_pct", "z"]
    out[num] = out[num].round(2)
    return out
