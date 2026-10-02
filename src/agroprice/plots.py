"""Graphiques du README et du dashboard."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LABELS = {
    "naif": "Naïf",
    "naif_saisonnier": "Naïf saisonnier",
    "ets": "ETS",
    "prophet": "Prophet",
    "lightgbm_global": "LightGBM global",
}
COLORS = {
    "naif": "#222222",
    "naif_saisonnier": "#9aa0a6",
    "ets": "#1f77b4",
    "prophet": "#d62728",
    "lightgbm_global": "#2ca02c",
}
PRODUITS_FR = {
    "Maize (white)": "Maïs blanc",
    "Cassava meal (gari)": "Gari",
    "Rice (imported)": "Riz importé",
    "Sorghum": "Sorgho",
}


def _style(ax, title, xlabel="", ylabel=""):
    ax.set_title(title, fontsize=12, loc="left", fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def plot_prices(series: dict, produit: str, path):
    fig, ax = plt.subplots(figsize=(9, 4))
    for (m, c), s in series.items():
        if c == produit:
            ax.plot(s.index, s.to_numpy(), lw=1.1, label=m)
    _style(ax, f"Prix de détail du produit « {PRODUITS_FR.get(produit, produit)} » par marché", "", "XOF / kg")
    ax.legend(ncol=3, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_mase(summary: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for m, g in summary.groupby("modele"):
        ax.plot(g["h"], g["MASE"], marker="o", label=LABELS.get(m, m), color=COLORS.get(m),
                lw=2.2 if m == "naif" else 1.6, ls="--" if m == "naif" else "-")
    _style(ax, "Erreur de prévision par horizon (MASE, plus bas = mieux)", "Horizon (mois)", "MASE")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_significance(sig: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(8, 4.4))
    models = list(sig["modele"].unique())
    off = np.linspace(-0.27, 0.27, len(models))
    for o, m in zip(off, models):
        g = sig[sig["modele"] == m]
        ax.errorbar(g["h"] + o, g["ecart_mase"],
                    yerr=[g["ecart_mase"] - g["ic95_bas"], g["ic95_haut"] - g["ecart_mase"]],
                    fmt="o", ms=4, capsize=2, label=LABELS.get(m, m), color=COLORS.get(m))
    ax.axhline(0, color="black", lw=1)
    _style(ax, "Écart de MASE vs naïf, IC 95 % (négatif = meilleur)",
           "Horizon (mois)", "Écart de MASE")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_coverage(cov: pd.DataFrame, target: float, path):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for m, g in cov.groupby("modele"):
        ax.plot(g["h"], g["couverture"] * 100, marker="o", label=LABELS.get(m, m), color=COLORS.get(m))
    ax.axhline(target * 100, color="black", ls="--", lw=1)
    ax.text(6.05, target * 100, f"cible {target:.0%}", va="center", fontsize=8)
    _style(ax, "Couverture réelle des intervalles de prévision", "Horizon (mois)", "% de prix réels dans l'intervalle")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_example(series, ri: pd.DataFrame, marche: str, produit: str, path, models=("naif", "lightgbm_global")):
    y = series[(marche, produit)]
    d = ri[(ri["marche"] == marche) & (ri["produit"] == produit)]
    origin = d["origine"].max()
    fig, ax = plt.subplots(figsize=(9, 4.2))
    hist = y[y.index >= origin - pd.DateOffset(years=3)]
    ax.plot(hist.index, hist.to_numpy(), color="#8a8a8a", lw=2.2, label="Prix observés")
    for m in models:
        g = d[(d["modele"] == m) & (d["origine"] == origin)].sort_values("h")
        ax.plot(g["date"], g["prevu"], marker="o", ms=3, color=COLORS[m], label=f"Prévision {LABELS[m]}")
        if m == models[-1]:
            ax.fill_between(g["date"], g["bas"], g["haut"], color=COLORS[m], alpha=0.18, label="Intervalle 80 %")
    ax.axvline(origin, color="grey", ls=":", lw=1)
    _style(ax, f"{marche}, {PRODUITS_FR.get(produit, produit).lower()} : prévision à 6 mois faite en {origin:%m/%Y}",
           "", "XOF / kg")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
