"""Modèles de prévision avec une interface commune : fit(y) puis predict(h)."""
from __future__ import annotations

import warnings
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


def _fill(y: pd.Series, limit: int = 3) -> pd.Series:
    """Comble les petits trous (interpolation) ; les longs trous restants sont retirés."""
    y = y.interpolate(limit=limit, limit_direction="both")
    return y.dropna()


class Forecaster(ABC):
    name: str = "base"

    @abstractmethod
    def fit(self, y: pd.Series) -> "Forecaster": ...

    @abstractmethod
    def predict(self, h: int) -> np.ndarray: ...


class Naive(Forecaster):
    """Dernier prix observé. Référence difficile à battre sur les prix mensuels."""

    name = "naif"

    def fit(self, y):
        self.last_ = float(_fill(y).iloc[-1])
        return self

    def predict(self, h):
        return np.full(h, self.last_)


class SeasonalNaive(Forecaster):
    """Prix du même mois l'an dernier."""

    name = "naif_saisonnier"

    def __init__(self, m: int = 12):
        self.m = m

    def fit(self, y):
        self.hist_ = _fill(y).to_numpy()
        return self

    def predict(self, h):
        idx = [len(self.hist_) - self.m + (i % self.m) for i in range(h)]
        return self.hist_[idx]


class ETS(Forecaster):
    """Lissage exponentiel (tendance amortie, saisonnalité additive), via statsmodels."""

    name = "ets"

    def __init__(self, seasonal: bool = True, m: int = 12):
        self.seasonal, self.m = seasonal, m

    def fit(self, y):
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        y = _fill(y)
        self.fallback_ = float(y.iloc[-1])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.res_ = ExponentialSmoothing(
                    y.to_numpy(),
                    trend="add",
                    damped_trend=True,
                    seasonal="add" if self.seasonal else None,
                    seasonal_periods=self.m if self.seasonal else None,
                    initialization_method="estimated",
                ).fit()
        except Exception:
            self.res_ = None
        return self

    def predict(self, h):
        if self.res_ is None:
            return np.full(h, self.fallback_)
        return np.asarray(self.res_.forecast(h))


class ProphetModel(Forecaster):
    """Prophet (tendance + saisonnalité annuelle). Import différé : dépendance optionnelle."""

    name = "prophet"

    def fit(self, y):
        import logging

        from prophet import Prophet

        for lg in ("cmdstanpy", "prophet"):
            logging.getLogger(lg).setLevel(logging.ERROR)
        self.last_ = y.index[-1]
        z = _fill(y)
        df = pd.DataFrame({"ds": z.index, "y": z.to_numpy()})
        self.m_ = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            uncertainty_samples=0,  # pas d'intervalles pour l'instant (plus rapide)
        )
        self.m_.fit(df)
        return self

    def predict(self, h):
        future = pd.DataFrame({"ds": pd.date_range(self.last_, periods=h + 1, freq="MS")[1:]})
        return self.m_.predict(future)["yhat"].to_numpy()


# Modèles par série. Le modèle global LightGBM est dans global_model.py.
REGISTRY = {
    "naif": Naive,
    "naif_saisonnier": SeasonalNaive,
    "ets": ETS,
    "prophet": ProphetModel,
}
