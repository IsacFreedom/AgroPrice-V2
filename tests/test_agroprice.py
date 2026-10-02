import numpy as np
import pandas as pd

from agroprice.data import clean, to_monthly_series
from agroprice.evaluate import backtest, mase_scale, summarize
from agroprice.models import ETS, Naive, SeasonalNaive


def _series(n=150, seed=0):
    idx = pd.date_range("2005-01-01", periods=n, freq="MS")
    rng = np.random.default_rng(seed)
    y = 100 + 10 * np.sin(2 * np.pi * np.arange(n) / 12) + rng.normal(0, 1, n)
    return pd.Series(y, index=idx)


def test_naive_repeats_last_value():
    y = _series()
    assert (Naive().fit(y).predict(3) == y.iloc[-1]).all()


def test_seasonal_naive_matches_last_year():
    y = _series()
    p = SeasonalNaive().fit(y).predict(12)
    assert np.allclose(p, y.iloc[-12:].to_numpy())


def test_ets_returns_right_length_and_finite():
    p = ETS().fit(_series()).predict(6)
    assert len(p) == 6 and np.isfinite(p).all()


def test_no_leakage_train_ends_before_test():
    res = backtest({("A", "x"): _series()}, ["naif"], horizon=3, min_train=60, step=12)
    assert (res["date"] > res["origine"]).all()


def test_seasonal_naive_beats_naive_on_seasonal_data():
    res = backtest({("A", "x"): _series()}, ["naif", "naif_saisonnier"], horizon=6, min_train=60, step=6)
    s = summarize(res, ["modele"]).set_index("modele")["MASE"]
    assert s["naif_saisonnier"] < s["naif"]


def test_clean_removes_duplicates_and_invalid():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-15"] * 3),
        "market": ["A"] * 3, "commodity": ["x"] * 3,
        "price": [100, 100, -5], "pricetype": ["Retail"] * 3,
        "currency": ["XOF"] * 3, "unit": ["KG"] * 3,
    })
    assert len(clean(df)) == 1


def test_monthly_series_keeps_gaps_as_nan():
    dates = pd.to_datetime(["2020-01-15", "2020-02-15", "2020-04-15"])
    df = pd.DataFrame({"date": dates, "market": "A", "commodity": "x", "price": [1.0, 2.0, 4.0]})
    s = to_monthly_series(df, min_obs=1)[("A", "x")]
    assert len(s) == 4 and s.isna().sum() == 1


def test_mase_scale_positive():
    assert mase_scale(_series()) > 0


def _panel_series(n=150):
    return {(m, "x"): _series(n, seed=i) for i, m in enumerate(["A", "B", "C"])}


def test_global_model_no_future_leakage():
    """Modifier les données APRÈS l'origine ne doit pas changer la prévision à cette origine."""
    from agroprice.global_model import backtest_global

    s1 = _panel_series()
    s2 = {k: v.copy() for k, v in s1.items()}
    for v in s2.values():
        v.iloc[100:] = v.iloc[100:] * 3
    kw = dict(horizon=3, min_train=60, step=20)
    r1, r2 = backtest_global(s1, **kw), backtest_global(s2, **kw)
    cutoff = s1[("A", "x")].index[99]
    a = r1[r1["origine"] <= cutoff].sort_values(["marche", "origine", "h"])["prevu"].to_numpy()
    b = r2[r2["origine"] <= cutoff].sort_values(["marche", "origine", "h"])["prevu"].to_numpy()
    assert len(a) > 0 and np.allclose(a, b)


def test_global_model_output_schema():
    from agroprice.global_model import backtest_global

    r = backtest_global(_panel_series(), horizon=3, min_train=60, step=30)
    assert {"marche", "produit", "origine", "date", "modele", "h", "reel", "prevu", "echelle"} <= set(r.columns)
    assert np.isfinite(r["prevu"]).all() and (r["date"] > r["origine"]).all()


def test_prophet_forecast_length():
    import pytest

    pytest.importorskip("prophet")
    from agroprice.models import ProphetModel

    assert len(ProphetModel().fit(_series(100)).predict(4)) == 4


def test_bootstrap_detects_clearly_worse_model():
    from agroprice.significance import paired_bootstrap

    rows = []
    for k, o in enumerate(pd.date_range("2010-01-01", periods=30, freq="6MS")):
        for model, p in (("naif", 101.0), ("mauvais", 105.0)):
            rows.append(dict(marche="A", produit="x", origine=o, date=o + pd.offsets.MonthBegin(1),
                             modele=model, h=1, reel=100.0, prevu=p, echelle=1.0))
    s = paired_bootstrap(pd.DataFrame(rows), n_boot=500)
    assert s.loc[0, "ecart_mase"] > 0 and bool(s.loc[0, "significatif"])


def test_intervals_coverage_close_to_nominal_and_uses_only_past():
    from agroprice.intervals import add_intervals, coverage_table

    rng = np.random.default_rng(1)
    origins = pd.date_range("2000-01-01", periods=400, freq="MS")
    e = rng.normal(0, 0.1, len(origins))
    res = pd.DataFrame(dict(marche="A", produit="x", origine=origins,
                            date=origins + pd.offsets.MonthBegin(1), modele="m", h=1,
                            reel=100 * np.exp(e), prevu=100.0, echelle=1.0))
    ri = add_intervals(res, alpha=0.2, min_cal=60)
    assert 0.74 < coverage_table(ri)["couverture"].iloc[0] < 0.86
    assert ri["origine"].min() > origins[59]  # pas d'intervalle sans calibration passée


def test_spike_detection_flags_sudden_rise():
    from agroprice.alerts import detect_spikes

    y = _series(150)
    y.iloc[130:] = y.iloc[130:] * 2.5
    out = detect_spikes({("A", "x"): y})
    assert out[out["alerte"]]["date"].min() >= y.index[129]


def test_streamlit_app_runs():
    import pytest
    from pathlib import Path

    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    root = Path(__file__).resolve().parents[1]
    if not (root / "results" / "forecast_latest.csv").exists():
        pytest.skip("lance d'abord scripts/run_all.py")
    at = AppTest.from_file(str(root / "app" / "streamlit_app.py"), default_timeout=60).run()
    assert not at.exception
    at.selectbox[0].select("Rice (imported)").run()
    assert not at.exception
