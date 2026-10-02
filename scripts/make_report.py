"""Génère tout ce qui dérive du backtest : significativité, intervalles, alertes,
prévisions récentes et graphiques. À lancer après run_backtest.py."""
from pathlib import Path

import pandas as pd

from agroprice.alerts import detect_spikes
from agroprice.data import clean, load_raw, to_monthly_series
from agroprice.evaluate import summarize
from agroprice.forecast import latest_forecasts
from agroprice.intervals import add_intervals, coverage_table
from agroprice.plots import plot_coverage, plot_example, plot_mase, plot_prices, plot_significance
from agroprice.significance import paired_bootstrap

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
ALPHA = 0.2


def main():
    raw = RES / "backtest_brut.csv"
    if not raw.exists():
        raise SystemExit("results/backtest_brut.csv introuvable : lance d'abord scripts/run_backtest.py")
    FIG.mkdir(exist_ok=True)
    res = pd.read_csv(raw, parse_dates=["origine", "date"])
    series = to_monthly_series(clean(load_raw()))

    print("1/5 Significativité (bootstrap)...")
    sig = paired_bootstrap(res)
    sig.to_csv(RES / "significativite_vs_naif.csv", index=False)

    print("2/5 Intervalles de prévision...")
    ri = add_intervals(res, ALPHA)
    ri.to_csv(RES / "backtest_intervalles.csv", index=False)
    cov = coverage_table(ri)
    cov.to_csv(RES / "couverture_intervalles.csv", index=False)

    print("3/5 Alertes de hausse...")
    detect_spikes(series).to_csv(RES / "signaux_prix.csv", index=False)

    print("4/5 Prévisions les plus récentes...")
    latest_forecasts(series, res, alpha=ALPHA).to_csv(RES / "forecast_latest.csv", index=False)

    print("5/5 Graphiques...")
    summary = summarize(res)
    plot_prices(series, "Maize (white)", FIG / "01_prix_mais.png")
    plot_mase(summary, FIG / "02_mase_par_horizon.png")
    plot_significance(sig, FIG / "03_significativite.png")
    plot_coverage(cov, 1 - ALPHA, FIG / "04_couverture_intervalles.png")
    plot_example(series, ri, "Lomé", "Maize (white)", FIG / "05_exemple_lome_mais.png")

    print("\n=== MASE par horizon ===")
    print(summary.pivot(index="modele", columns="h", values="MASE").to_string())
    print("\n=== Écart vs naïf (négatif = mieux), * = significatif ===")
    t = sig.assign(v=sig.apply(lambda r: f"{r.ecart_mase:+.3f}{'*' if r.significatif else ''}", axis=1))
    print(t.pivot(index="modele", columns="h", values="v").to_string())
    print("\n=== Couverture des intervalles à 80 % ===")
    print(cov.pivot(index="modele", columns="h", values="couverture").to_string())
    print("\nFichiers écrits dans results/ et figures/")


if __name__ == "__main__":
    main()
