"""Dashboard AgroPrice : prix, prévisions avec intervalles, alertes, performance des modèles."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from agroprice.data import clean, load_raw, to_monthly_series

RES, FIG = ROOT / "results", ROOT / "figures"
PRODUITS = {"Maize (white)": "Maïs blanc", "Cassava meal (gari)": "Gari", "Rice (imported)": "Riz importé", "Sorghum": "Sorgho"}
MODELES = {"lightgbm_global": "LightGBM global", "ets": "ETS", "naif": "Naïf (dernier prix)"}

st.set_page_config(page_title="AgroPrice Togo", page_icon="🌽", layout="wide")


@st.cache_data
def load_data():
    series = to_monthly_series(clean(load_raw()))
    fc = pd.read_csv(RES / "forecast_latest.csv", parse_dates=["date", "derniere_obs"])
    sig = pd.read_csv(RES / "signaux_prix.csv", parse_dates=["date"])
    summ = pd.read_csv(RES / "resume_par_modele_horizon.csv")
    cov = pd.read_csv(RES / "couverture_intervalles.csv")
    tst = pd.read_csv(RES / "significativite_vs_naif.csv")
    return series, fc, sig, summ, cov, tst


try:
    series, fc, sig, summ, cov, tst = load_data()
except FileNotFoundError:
    st.error("Résultats introuvables. Lance d'abord : `python scripts/run_all.py`")
    st.stop()

st.title("🌽 AgroPrice : prix des denrées sur les marchés du Togo")
last_data = max(s.index.max() for s in series.values())
st.warning(
    f"Les données WFP s'arrêtent en **{last_data:%m/%Y}**. Les prévisions ci-dessous partent de cette date : "
    "elles illustrent la méthode et ne sont pas des prévisions opérationnelles actuelles."
)

# ---- Sélection
with st.sidebar:
    st.header("Choix")
    produit = st.selectbox("Produit", list(PRODUITS), format_func=PRODUITS.get)
    marches = sorted(m for (m, c) in series if c == produit)
    marche = st.selectbox("Marché", marches, index=marches.index("Lomé") if "Lomé" in marches else 0)
    modele = st.selectbox("Modèle de prévision", list(MODELES), format_func=MODELES.get)
    annees = st.slider("Historique affiché (années)", 2, 20, 6)

y = series[(marche, produit)]
f = fc[(fc["marche"] == marche) & (fc["produit"] == produit) & (fc["modele"] == modele)].sort_values("h")
obs = y.dropna()

# ---- Indicateurs
c1, c2, c3 = st.columns(3)
c1.metric(f"Dernier prix observé ({obs.index[-1]:%m/%Y})", f"{obs.iloc[-1]:,.0f} XOF/kg".replace(",", " "))
if len(obs) > 3:
    var3 = (obs.iloc[-1] / obs.iloc[-4] - 1) * 100
    c2.metric("Variation sur 3 observations", f"{var3:+.1f} %")
if len(f) >= 3:
    r = f[f["h"] == 3].iloc[0]
    c3.metric(f"Prévision à 3 mois ({r['date']:%m/%Y})", f"{r['prevu']:,.0f} XOF/kg".replace(",", " "),
              help=f"Intervalle à 80 % : {r['bas']:,.0f} à {r['haut']:,.0f}".replace(",", " "))

tab1, tab2, tab3, tab4 = st.tabs(["Prévisions", "Alertes de hausse", "Performance des modèles", "Méthode"])

# ---- Prévisions
with tab1:
    fig, ax = plt.subplots(figsize=(10, 4))
    h = y[y.index >= y.index.max() - pd.DateOffset(years=annees)]
    ax.plot(h.index, h.to_numpy(), color="#555555", lw=1.8, label="Prix observés")
    if len(f):
        ax.plot(f["date"], f["prevu"], color="#2ca02c", marker="o", ms=4, label=f"Prévision ({MODELES[modele]})")
        ax.fill_between(f["date"], f["bas"], f["haut"], color="#2ca02c", alpha=0.2, label="Intervalle 80 %")
    ax.set_ylabel("XOF / kg")
    ax.grid(alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False)
    st.pyplot(fig)
    plt.close(fig)
    if len(f):
        show = f[["date", "prevu", "bas", "haut"]].rename(
            columns={"date": "Mois", "prevu": "Prévision", "bas": "Bas (10 %)", "haut": "Haut (90 %)"})
        show["Mois"] = show["Mois"].dt.strftime("%m/%Y")
        st.dataframe(show, hide_index=True, width="stretch")
    st.caption("Les intervalles sont calibrés sur les erreurs passées du modèle (voir l'onglet Performance). "
               "Un trou dans la courbe correspond à des mois sans relevé.")

# ---- Alertes
with tab2:
    s = sig[(sig["marche"] == marche) & (sig["produit"] == produit)]
    fig, ax = plt.subplots(figsize=(10, 3.6))
    hh = s[s["date"] >= s["date"].max() - pd.DateOffset(years=annees)]
    ax.plot(hh["date"], hh["prix"], color="#555555", lw=1.5)
    al = hh[hh["alerte"]]
    ax.scatter(al["date"], al["prix"], color="#d62728", zorder=3, label="Hausse anormale")
    ax.set_ylabel("XOF / kg")
    ax.grid(alpha=0.25)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False)
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Alerte : la hausse sur 3 mois dépasse de plus de 2 écarts-types les variations des 60 mois précédents.")
    t = s[s["alerte"]].sort_values("date", ascending=False).head(15)
    if len(t):
        t = t.assign(Mois=t["date"].dt.strftime("%m/%Y"))[["Mois", "prix", "variation_3m_pct", "z"]]
        t.columns = ["Mois", "Prix (XOF/kg)", "Variation 3 mois (%)", "Score z"]
        st.dataframe(t, hide_index=True, width="stretch")
    else:
        st.info("Aucune alerte sur cette série.")

# ---- Performance
with tab3:
    st.subheader("Erreur de prévision (MASE, plus bas = mieux)")
    st.dataframe(summ.pivot(index="modele", columns="h", values="MASE"), width="stretch")
    a, b = st.columns(2)
    a.image(str(FIG / "03_significativite.png"), caption="Écart vs naïf avec IC 95 %")
    b.image(str(FIG / "04_couverture_intervalles.png"), caption="Couverture des intervalles à 80 %")
    st.markdown(
        "**Lecture honnête.** Les prix sont très persistants : aucun modèle ne bat de façon "
        "répétée et significative le simple « dernier prix observé ». Prophet et le naïf saisonnier "
        "sont significativement moins bons. Voir le tableau d'écarts ci-dessous (colonne *significatif*)."
    )
    st.dataframe(tst, hide_index=True, width="stretch")

# ---- Méthode
with tab4:
    st.markdown(
        """
**Données** : prix de détail mensuels du Programme alimentaire mondial (WFP/HDX), Togo, 2001 à 2022, 6 marchés, 4 produits.

**Validation** : à origine glissante, entraînement uniquement sur le passé, prévision de 1 à 6 mois. Métrique : MASE.

**Modèles** : naïf, naïf saisonnier, ETS, Prophet, LightGBM global (un modèle sur toutes les séries).

**Intervalles** : quantiles des erreurs logarithmiques passées (calibration conforme), jamais le futur.

**Significativité** : bootstrap apparié par date d'origine, IC à 95 %.

**Limites** : données arrêtées en 2022, peu de marchés, pas de variables explicatives (pluie, carburant),
plusieurs comparaisons simultanées (un résultat « significatif » isolé peut être dû au hasard).
        """
    )
