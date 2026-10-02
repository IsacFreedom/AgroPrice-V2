"""Chargement, nettoyage et audit des prix WFP du Togo."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "wfp_food_prices_tgo.csv"


def load_raw(path: str | Path = DEFAULT_PATH) -> pd.DataFrame:
    """Charge le CSV WFP (HDX). Ignore la ligne d'étiquettes HXL si présente."""
    df = pd.read_csv(path)
    if str(df.iloc[0, 0]).startswith("#"):  # ligne HXL: "#date", "#adm1+name", ...
        df = df.iloc[1:].reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Garde les prix de détail en XOF/kg, valides, sans doublon."""
    out = df.dropna(subset=["date", "price"]).copy()
    out = out[out["price"] > 0]
    for col, val in {"pricetype": "Retail", "currency": "XOF", "unit": "KG"}.items():
        if col in out.columns:
            out = out[out[col] == val]
    out = out.drop_duplicates(subset=["market", "commodity", "date"])
    return out.reset_index(drop=True)


def to_monthly_series(df: pd.DataFrame, min_obs: int = 120) -> dict[tuple[str, str], pd.Series]:
    """Une série mensuelle (index MS) par couple (marché, produit).

    Les mois manquants restent NaN : on ne les invente pas ici.
    """
    series: dict[tuple[str, str], pd.Series] = {}
    for (market, commodity), g in df.groupby(["market", "commodity"]):
        s = g.set_index(g["date"].dt.to_period("M").dt.to_timestamp())["price"].sort_index()
        s = s[~s.index.duplicated(keep="last")]
        s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="MS"))
        if s.notna().sum() >= min_obs:
            s.name = f"{market}|{commodity}"
            series[(market, commodity)] = s
    return series


def audit(series: dict[tuple[str, str], pd.Series]) -> pd.DataFrame:
    """Tableau d'audit : couverture, trous, plus longue séquence manquante."""
    rows = []
    for (market, commodity), s in series.items():
        miss = s.isna()
        run = (miss != miss.shift()).cumsum()
        longest = int(miss.groupby(run).sum().max()) if miss.any() else 0
        rows.append(
            {
                "marche": market,
                "produit": commodity,
                "debut": s.index.min().date(),
                "fin": s.index.max().date(),
                "mois": len(s),
                "manquants": int(miss.sum()),
                "plus_long_trou": longest,
                "prix_median": float(s.median()),
            }
        )
    return pd.DataFrame(rows).sort_values(["produit", "marche"]).reset_index(drop=True)
