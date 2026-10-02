"""Les écarts entre modèles sont-ils réels ou dus au hasard ? Bootstrap apparié."""
from __future__ import annotations

import numpy as np
import pandas as pd

KEYS = ["marche", "produit", "origine", "date", "h"]


def paired_bootstrap(res: pd.DataFrame, baseline: str = "naif", n_boot: int = 5000, seed: int = 0) -> pd.DataFrame:
    """Écart de MASE entre chaque modèle et la référence, avec IC à 95 %.

    Écart = MASE(modèle) - MASE(référence) : négatif signifie que le modèle fait mieux.
    Rééchantillonnage par DATE D'ORIGINE (les marchés subissent les mêmes chocs au même
    moment : les traiter comme indépendants rendrait les IC trop optimistes).
    """
    r = res.copy()
    r["se"] = (r["reel"] - r["prevu"]).abs() / r["echelle"]
    wide = r.pivot_table(index=KEYS, columns="modele", values="se").dropna().reset_index()
    rng = np.random.default_rng(seed)
    rows = []
    for h, wh in wide.groupby("h"):
        origins = np.sort(wh["origine"].unique())
        k = len(origins)
        idx = rng.integers(0, k, size=(n_boot, k))
        for m in [c for c in wh.columns if c not in KEYS and c != baseline]:
            d = wh[m] - wh[baseline]
            g = d.groupby(wh["origine"])
            s = g.sum().reindex(origins).to_numpy()
            n = g.size().reindex(origins).to_numpy()
            boot = s[idx].sum(axis=1) / n[idx].sum(axis=1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            rows.append(
                dict(modele=m, h=int(h), ecart_mase=s.sum() / n.sum(), ic95_bas=lo, ic95_haut=hi,
                     significatif=bool(lo > 0 or hi < 0), n_origines=k)
            )
    return pd.DataFrame(rows).round(3)
