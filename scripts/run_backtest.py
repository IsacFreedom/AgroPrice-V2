"""Lance la validation glissante et écrit les résultats dans results/."""
import argparse
from pathlib import Path

from agroprice.data import audit, clean, load_raw, to_monthly_series
import pandas as pd

from agroprice.evaluate import align_common, backtest, summarize
from agroprice.global_model import NAME as GLOBAL, backtest_global
from agroprice.models import REGISTRY

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=list(REGISTRY) + [GLOBAL])
    p.add_argument("--horizon", type=int, default=6)
    p.add_argument("--min-train", type=int, default=96)
    p.add_argument("--step", type=int, default=6)
    a = p.parse_args()

    series = to_monthly_series(clean(load_raw()))
    out = ROOT / "results"
    out.mkdir(exist_ok=True)

    audit(series).to_csv(out / "audit_donnees.csv", index=False)
    print(f"{len(series)} séries retenues\n")

    local = [m for m in a.models if m in REGISTRY]
    parts = []
    if local:
        print("Modèles par série :", local, "(Prophet peut prendre plusieurs minutes)")
        parts.append(backtest(series, local, a.horizon, a.min_train, a.step))
    if GLOBAL in a.models:
        print("Modèle global :", GLOBAL)
        parts.append(backtest_global(series, a.horizon, a.min_train, a.step))
    res = align_common(pd.concat(parts, ignore_index=True))
    res.to_csv(out / "backtest_brut.csv", index=False)
    s = summarize(res)
    s.to_csv(out / "resume_par_modele_horizon.csv", index=False)
    summarize(res, ["produit", "modele"]).to_csv(out / "resume_par_produit.csv", index=False)
    print(s.pivot(index="modele", columns="h", values="MASE").to_string())
    print("\n(MASE < 1 : meilleur que le modèle naïf d'un pas en arrière)")


if __name__ == "__main__":
    main()
