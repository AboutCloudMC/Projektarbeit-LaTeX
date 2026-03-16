"""Modell-Training (Ridge/RF, logistische Klassifikatoren, ARIMA-Baselines)."""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import clone
try:
    from sklearn.frozen import FrozenEstimator
except ImportError:  # scikit-learn < 1.6
    FrozenEstimator = None
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import warnings

from castmark.features import HorizonDataset

LOGGER = logging.getLogger(__name__)


@dataclass
class ModelPrediction:
    """Hält ein trainiertes Modell inklusive Vorhersagen pro Datensplit fest."""

    name: str
    predictions: Dict[str, pd.Series]
    model: object
    extras: Dict[str, object] | None = None
    oof_predictions: Dict[str, pd.Series] | None = None


def _fit_ridge_regressor(train_split: pd.DataFrame, y: pd.Series, cfg: Dict[str, object]) -> Ridge:
    """Trainiert einen Ridge-Regressor mit den angegebenen Hyperparametern.

    Parameter
    ---------
    train_split : pd.DataFrame
        Trainingsfeatures (bereits skaliert, falls vorgesehen).
    y : pd.Series
        Zielreihe (volatilitätsskalierte Rendite).
    cfg : Dict[str, object]
        Abschnitt `models.return_regressors.ridge` aus der Konfiguration.

    Returns
    -------
    Ridge
        Trainiertes scikit-learn-Modell.
    """
    model = Ridge(alpha=float(cfg.get("alpha", 1.0)))
    model.fit(train_split, y)
    return model


def _build_random_forest(cfg: Dict[str, object]) -> RandomForestRegressor:
    """Erstellt einen RandomForestRegressor aus der Konfiguration (ohne Fit).

    Parameter
    ---------
    cfg : Dict[str, object]
        Hyperparameter der Random-Forest-Sektion.

    Returns
    -------
    RandomForestRegressor
        Noch nicht trainiertes Ensemble.
    """
    return RandomForestRegressor(
        n_estimators=int(cfg.get("n_estimators", 300)),
        max_depth=cfg.get("max_depth"),
        min_samples_split=int(cfg.get("min_samples_split", 2)),
        min_samples_leaf=int(cfg.get("min_samples_leaf", 1)),
        max_features=cfg.get("max_features", "sqrt"),
        n_jobs=int(cfg.get("n_jobs", -1)),
        random_state=int(cfg.get("random_state", 42)),
        bootstrap=bool(cfg.get("bootstrap", True)),
        max_samples=cfg.get("max_samples"),
    )


def _fit_random_forest(train_split: pd.DataFrame, y: pd.Series, cfg: Dict[str, object]) -> RandomForestRegressor:
    """Trainiert einen RandomForest-Regressor für horizon-spezifische Returns.

    Parameter
    ---------
    train_split : pd.DataFrame
        Trainingsfeatures.
    y : pd.Series
        Trainingszielwerte.
    cfg : Dict[str, object]
        Hyperparameter der Random-Forest-Sektion.

    Returns
    -------
    RandomForestRegressor
        Angepasstes Ensemble.
    """
    model = _build_random_forest(cfg)
    model.fit(train_split, y)
    return model


def _predict_across_splits(model, dataset: HorizonDataset) -> Dict[str, pd.Series]:
    """Berechnet Vorhersagen für Train/Validation/Test und gibt sie als Serien zurück.

    Parameter
    ---------
    model : Any
        Passendes scikit-learn-Objekt mit `predict`.
    dataset : HorizonDataset
        Enthält alle Splits samt Feature-Matrizen.

    Returns
    -------
    Dict[str, pd.Series]
        Prognosen je Split, index-aligniert zu den Input-Features.
    """
    predictions: Dict[str, pd.Series] = {}
    for split_name in ("train", "validation", "test"):
        split = getattr(dataset, split_name)
        preds = pd.Series(model.predict(split.X), index=split.X.index, name="prediction")
        predictions[split_name] = preds
    return predictions


def _time_series_oof_predictions(
    model_template,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int,
    min_train_size: int,
) -> pd.Series:
    """Erzeugt Out-of-Fold-Vorhersagen per TimeSeriesSplit.

    Pro Fold wird ein frischer Klon des Modells trainiert, damit Trainings-
    und Testdaten strikt getrennt bleiben. Indizes ohne Test-Fold erhalten NaN.

    Parameter
    ---------
    model_template : estimator
        Noch nicht gefittetes sklearn-Modell, das je Fold geklont wird.
    X : pd.DataFrame
        Feature-Matrix des Trainings-Splits.
    y : pd.Series
        Zielreihe des Trainings-Splits.
    n_splits : int
        Anzahl der zeitlich aufeinanderfolgenden Folds.
    min_train_size : int
        Mindestlänge des Trainingsfensters je Fold; kleinere Fenster werden übersprungen.

    Returns
    -------
    pd.Series
        OOF-Vorhersagen, indexiert wie X; NaN wo kein Test-Fold existiert.
    """
    splitter = TimeSeriesSplit(n_splits=n_splits)
    oof = pd.Series(np.nan, index=X.index, name="oof_prediction")
    for train_idx, test_idx in splitter.split(X):
        if len(train_idx) < min_train_size or len(test_idx) == 0:
            continue
        model = clone(model_template)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[test_idx])
        oof.iloc[test_idx] = preds
    return oof


def train_return_regressors(dataset: HorizonDataset, model_cfg: Dict[str, object]) -> Dict[str, ModelPrediction]:
    """Trainiert alle konfigurierten Regressionsmodelle für die Zielrenditen.

    Parameter
    ---------
    dataset : HorizonDataset
        Enthält train/val/test-Splits für einen Prognosehorizont.
    model_cfg : Dict[str, object]
        Abschnitt `models` aus der Konfiguration.

    Returns
    -------
    Dict[str, ModelPrediction]
        Mapping von Modellnamen zu Prognosen und Fit-Objekten.
    """
    regressors_cfg = model_cfg.get("return_regressors", {})
    oof_cfg = regressors_cfg.get("oof", {})
    oof_enabled = bool(oof_cfg.get("enabled", True))
    oof_splits = int(oof_cfg.get("n_splits", 5))
    oof_min_train = int(oof_cfg.get("min_train_size", 100))
    results: Dict[str, ModelPrediction] = {}

    if "ridge" in regressors_cfg:
        ridge_model = _fit_ridge_regressor(dataset.train.X, dataset.train.y, regressors_cfg["ridge"])
        oof_preds = None
        if oof_enabled:
            ridge_template = Ridge(alpha=float(regressors_cfg["ridge"].get("alpha", 1.0)))
            oof_preds = _time_series_oof_predictions(
                ridge_template,
                dataset.train.X,
                dataset.train.y,
                n_splits=oof_splits,
                min_train_size=oof_min_train,
            )
        results["ridge"] = ModelPrediction(
            name="ridge",
            model=ridge_model,
            predictions=_predict_across_splits(ridge_model, dataset),
            extras=None,
            oof_predictions={"train": oof_preds} if oof_preds is not None else None,
        )

    if "random_forest" in regressors_cfg:
        rf_model = _fit_random_forest(dataset.train.X, dataset.train.y, regressors_cfg["random_forest"])
        oof_preds = None
        if oof_enabled:
            rf_template = _build_random_forest(regressors_cfg["random_forest"])
            oof_preds = _time_series_oof_predictions(
                rf_template,
                dataset.train.X,
                dataset.train.y,
                n_splits=oof_splits,
                min_train_size=oof_min_train,
            )
        results["random_forest"] = ModelPrediction(
            name="random_forest",
            model=rf_model,
            predictions=_predict_across_splits(rf_model, dataset),
            extras=None,
            oof_predictions={"train": oof_preds} if oof_preds is not None else None,
        )

    return results


def _fit_directional_classifier(train_split: pd.DataFrame, direction: pd.Series, cfg: Dict[str, object]) -> LogisticRegression:
    """Passt eine logistische Regression auf die Richtungslabels an.

    Parameter
    ---------
    train_split : pd.DataFrame
        Trainingsfeatures (Base- oder Meta-Features).
    direction : pd.Series
        Binäre Labels (1 = steigender Kurs, 0 = fallender Kurs).
    cfg : Dict[str, object]
        Hyperparameter des Klassifikators.

    Returns
    -------
    LogisticRegression
        Trainiertes Klassifikationsmodell.
    """
    model_kwargs = {
        "C": float(cfg.get("C", 1.0)),
        "class_weight": cfg.get("class_weight", "balanced"),
        "max_iter": int(cfg.get("max_iter", 1000)),
        "solver": cfg.get("solver", "lbfgs"),
    }

    penalty = cfg.get("penalty")
    if penalty is not None:
        penalty_value = str(penalty).lower()
        if penalty_value == "none":
            # Neue sklearn-Versionen empfehlen C=np.inf statt penalty=None.
            model_kwargs["C"] = np.inf
        elif penalty_value != "l2":
            model_kwargs["penalty"] = penalty_value
            if penalty_value == "elasticnet" and cfg.get("l1_ratio") is not None:
                model_kwargs["l1_ratio"] = float(cfg.get("l1_ratio"))

    model = LogisticRegression(**model_kwargs)
    model.fit(train_split, direction)
    return model


def train_classifier_from_features(
    feature_splits: Dict[str, pd.DataFrame],
    direction_splits: Dict[str, pd.Series],
    classifier_cfg: Dict[str, object],
    calibration_cfg: Dict[str, object],
    name: str,
) -> ModelPrediction | None:
    """Trainiert (und kalibriert optional) einen Klassifikator auf vordefinierten Splits.

    Parameter
    ---------
    feature_splits : Dict[str, pd.DataFrame]
        Features pro Split (train/validation/test).
    direction_splits : Dict[str, pd.Series]
        Richtungslabels pro Split.
    classifier_cfg : Dict[str, object]
        Hyperparameter (Penalty, C, usw.).
    calibration_cfg : Dict[str, object]
        Einstellungen für die Kalibrierung (aktiv, Methode).
    name : str
        Anzeigename des Klassifikators.

    Returns
    -------
    ModelPrediction | None
        Enthält Wahrscheinlichkeiten pro Split oder `None`, falls Training nicht möglich war.
    """
    train_X = feature_splits.get("train")
    train_y = direction_splits.get("train")
    if train_X is None or train_y is None or train_y.empty:
        LOGGER.warning("Skipping classifier %s due to missing training data", name)
        return None

    train_frame = train_X.copy()
    train_frame["__y__"] = train_y
    train_frame = train_frame.dropna()
    if train_frame.empty:
        LOGGER.warning("Skipping classifier %s due to NaN-only training data", name)
        return None
    train_y = train_frame.pop("__y__")
    train_X = train_frame

    if train_y.nunique(dropna=True) < 2:
        LOGGER.warning("Skipping classifier %s due to single-class training labels", name)
        return None

    model = _fit_directional_classifier(train_X, train_y, classifier_cfg)
    predictor = model

    if calibration_cfg.get("enabled", True):
        val_X = feature_splits.get("validation")
        val_y = direction_splits.get("validation")
        if val_X is not None and val_y is not None and not val_y.empty:
            val_frame = val_X.copy()
            val_frame["__y__"] = val_y
            val_frame = val_frame.dropna()
            if val_frame.empty:
                LOGGER.warning("Calibration skipped for %s due to NaN-only validation data", name)
            else:
                val_y = val_frame.pop("__y__")
                val_X = val_frame
                if val_y.nunique(dropna=True) < 2:
                    LOGGER.warning("Calibration skipped for %s due to single-class validation data", name)
                else:
                    method = calibration_cfg.get("method", "isotonic")
                    # Verwende in neuen sklearn-Versionen einen eingefrorenen, bereits trainierten Estimator.
                    # Für ältere Versionen bleibt `cv="prefit"` als Fallback erhalten.
                    if FrozenEstimator is not None:
                        predictor = CalibratedClassifierCV(FrozenEstimator(model), method=method)
                    else:
                        predictor = CalibratedClassifierCV(model, method=method, cv="prefit")
                    predictor.fit(val_X, val_y)
        else:
            LOGGER.warning("Calibration skipped for %s due to missing validation data", name)

    predictions: Dict[str, pd.Series] = {}
    for split_name, X_split in feature_splits.items():
        if X_split is None or X_split.empty:
            predictions[split_name] = pd.Series(dtype=float, name="probability")
            continue
        valid_mask = X_split.notna().all(axis=1)
        probs = pd.Series(np.nan, index=X_split.index, name="probability")
        if valid_mask.any():
            prob = predictor.predict_proba(X_split.loc[valid_mask])[:, 1]
            probs.loc[valid_mask] = prob
        predictions[split_name] = probs

    return ModelPrediction(name=name, predictions=predictions, model=predictor, extras=None)


def _prepare_business_day_series(frame: pd.DataFrame, target_column: str) -> pd.Series:
    """Bereitet eine saubere Preisreihe für ARIMA vor, ohne künstliche Handelstage.

    Parameter
    ---------
    frame : pd.DataFrame
        Vollständiger Kursdatensatz.
    target_column : str
        Spaltenname des Preisfelds.

    Returns
    -------
    pd.Series
        Sortierte Preisreihe auf beobachteten Handelstagen.

    Raises
    ------
    ValueError
        Falls keine gültigen Preiswerte vorhanden sind.
    """
    series = frame[target_column].dropna().sort_index()
    if series.empty:
        raise ValueError("No target data available for ARIMA modelling")
    series = series[~series.index.duplicated(keep="last")]
    return series.astype(float)


def _grid_search_arima(train_series: pd.Series, cfg: Dict[str, object]) -> tuple[tuple[int, int, int], tuple[int, int, int, int]]:
    """Sucht per Grid-Search nach der besten (S)ARIMA-Konfiguration basierend auf dem AIC.

    Parameter
    ---------
    train_series : pd.Series
        Kontinuierliche Abschlusskurse (Business-Day-Frequenz).
    cfg : Dict[str, object]
        Abschnitt `models.arima` mit Grid-Definitionen.

    Returns
    -------
    tuple
        (`order`, `seasonal_order`) mit den jeweils besten Parametern.
    """
    p_values = cfg.get("p_values", [0, 1, 2])
    d_values = cfg.get("d_values", [0, 1])
    q_values = cfg.get("q_values", [0, 1, 2])

    seasonal = bool(cfg.get("seasonal", False))
    seasonal_period = int(cfg.get("seasonal_period", 5))
    if seasonal:
        P_values = cfg.get("seasonal_params", {}).get("P_values", [0, 1])
        D_values = cfg.get("seasonal_params", {}).get("D_values", [0, 1])
        Q_values = cfg.get("seasonal_params", {}).get("Q_values", [0, 1])
    else:
        P_values = D_values = Q_values = [0]

    model_series = pd.Series(train_series.to_numpy(dtype=float), index=pd.RangeIndex(len(train_series)))

    best_aic = np.inf
    best_order = (1, 1, 1)
    best_seasonal = (0, 0, 0, seasonal_period if seasonal else 0)

    for order in itertools.product(p_values, d_values, q_values):
        for seasonal_order in itertools.product(P_values, D_values, Q_values):
            seasonal_tuple = (*seasonal_order, seasonal_period)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=ConvergenceWarning)
                    model = SARIMAX(
                        model_series,
                        order=order,
                        seasonal_order=seasonal_tuple,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                    result = model.fit(disp=False)
                # Der Akaike-Informationskriterium (AIC) dient als Vergleichsmetrik.
                if result.aic < best_aic:
                    best_aic = result.aic
                    best_order = order
                    best_seasonal = seasonal_tuple
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("ARIMA order %s seasonal %s fehlgeschlagen: %s", order, seasonal_tuple, exc)
                continue

    return best_order, best_seasonal


def _fit_sarimax_full(series: pd.Series, order: tuple[int, int, int], seasonal_order: tuple[int, int, int, int]):
    """Trainiert ein SARIMAX-Modell auf der vollständigen Historie und liefert das Resultat.

    Parameter
    ---------
    series : pd.Series
        Zeitreihe mit regulären Abständen.
    order : tuple[int, int, int]
        Nicht-saisonale Parameter (p, d, q).
    seasonal_order : tuple[int, int, int, int]
        Saisonale Parameter (P, D, Q, s).

    Returns
    -------
    SARIMAXResults
        Gefittetes Statsmodels-Objekt.
    """
    # Statsmodels benötigt einen unterstützten Index mit Frequenz. Da Börsendaten
    # wegen Feiertagen oft keine fixe Frequenz tragen, modellieren wir intern auf
    # einer positionsbasierten Serie und behalten die Datumslogik außerhalb.
    model_series = pd.Series(series.to_numpy(dtype=float), index=pd.RangeIndex(len(series)))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        model = SARIMAX(
            model_series,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        result = model.fit(disp=False)
    return result


def _walk_forward_forecast(
    series: pd.Series,
    target_index: pd.Index,
    horizon: int,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    window_size: int,
    forecast_cache: Dict[tuple[object, tuple[int, int, int], tuple[int, int, int, int], int], np.ndarray] | None = None,
    max_forecast_steps: int | None = None,
) -> pd.Series:
    """Setzt ein Walk-Forward-Fenster auf und erstellt horizon-lange Vorhersagen.

    Parameter
    ---------
    series : pd.Series
        Vollständige Historie mit Business-Day-Frequenz.
    target_index : pd.Index
        Zeitpunkte, für die Vorhersagen benötigt werden (z. B. Test-Split-Indizes).
    horizon : int
        Wie viele Tage in die Zukunft der letzte Forecast-Schritt liegt.
    order : tuple[int, int, int]
        ARIMA-Parameter.
    seasonal_order : tuple[int, int, int, int]
        Saisonale Parameter.
    window_size : int
        Länge des rollierenden Trainingsfensters für das Walk-Forward-Szenario.

    Returns
    -------
    pd.Series
        Prognostizierte Preise, indexiert nach den angefragten Zeitpunkten.
    """
    predictions = {}
    forecast_steps = max(horizon, int(max_forecast_steps or horizon))
    for timestamp in target_index:
        try:
            pos = series.index.get_loc(timestamp)
        except KeyError:
            continue
        # Schneidet ein rollierendes Fenster der Historie aus, das vor dem Zielzeitpunkt endet.
        start = max(0, pos - window_size + 1)
        history = series.iloc[start : pos + 1]
        if len(history) < max(order[1] + seasonal_order[1], 5):
            continue
        cache_key = (timestamp, order, seasonal_order, window_size)
        cached_forecast = forecast_cache.get(cache_key) if forecast_cache is not None else None
        if cached_forecast is not None and len(cached_forecast) >= forecast_steps:
            predictions[timestamp] = float(cached_forecast[horizon - 1])
            continue
        try:
            fitted = _fit_sarimax_full(history, order, seasonal_order)
            forecast = fitted.get_forecast(steps=forecast_steps).predicted_mean.to_numpy(dtype=float)
            if forecast_cache is not None:
                forecast_cache[cache_key] = forecast
            predictions[timestamp] = float(forecast[horizon - 1])
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Walk-Forward-ARIMA bei %s fehlgeschlagen: %s", timestamp, exc)
            continue
    if not predictions:
        return pd.Series(dtype=float)
    return pd.Series(predictions, name="prediction")


def generate_arima_walk_forward_predictions(
    frame: pd.DataFrame,
    dataset: HorizonDataset,
    horizon: int,
    target_column: str,
    splits_cfg: Dict[str, float],
    model_cfg: Dict[str, object],
    selected_order: tuple[int, int, int] | None = None,
    selected_seasonal_order: tuple[int, int, int, int] | None = None,
    forecast_cache: Dict[tuple[object, tuple[int, int, int], tuple[int, int, int, int], int], np.ndarray] | None = None,
    max_forecast_steps: int | None = None,
) -> ModelPrediction | None:
    """Erzeugt volatilitätsskalierte ARIMA-Signale je Split, falls aktiviert.

    Parameter
    ---------
    frame : pd.DataFrame
        Rohdaten des aktuellen Tickers.
    dataset : HorizonDataset
        Enthält Splits für denselben Ticker/Horizont.
    horizon : int
        Prognosehorizont, bestimmt die Länge jedes Forecasts.
    target_column : str
        Kursfeld, das modelliert werden soll.
    splits_cfg : Dict[str, float]
        Pipeline-Split-Parameter plus Walk-Forward-Einstellungen.
    model_cfg : Dict[str, object]
        Gesamter Modell-Block mit ARIMA-Untersektion.

    Returns
    -------
    ModelPrediction | None
        Baseline mit ARIMA-Signalen oder `None`, wenn deaktiviert oder mangels Daten.
    """
    walk_cfg = splits_cfg.get("walk_forward", {})
    if not walk_cfg.get("enabled", False):
        return None

    series = _prepare_business_day_series(frame, target_column)
    train_ratio = float(splits_cfg.get("train_ratio", 0.7))
    train_end = max(10, int(len(series) * train_ratio))
    base_train_series = series.iloc[:train_end]
    if len(base_train_series) < 30:
        LOGGER.warning("Insufficient history for ARIMA walk-forward baseline")
        return None

    order = selected_order
    seasonal_order = selected_seasonal_order
    if order is None or seasonal_order is None:
        order, seasonal_order = _grid_search_arima(base_train_series, model_cfg.get("arima", {}))
        LOGGER.info("Best ARIMA order=%s seasonal=%s", order, seasonal_order)

    window_size = int(walk_cfg.get("window_size", min(len(series), 252)))

    predictions = {}
    for split_name in ("train", "validation", "test"):
        split_index = getattr(dataset, split_name).X.index
        # Jedes Split benötigt Vorhersagen genau für seine Zeitscheiben, daher Reindex über `split_index`.
        preds = _walk_forward_forecast(
            series,
            split_index,
            horizon,
            order,
            seasonal_order,
            window_size,
            forecast_cache=forecast_cache,
            max_forecast_steps=max_forecast_steps,
        )
        predictions[split_name] = preds

    return ModelPrediction(
        name=f"arima_h{horizon}",
        model=None,
        predictions=predictions,
        extras={"order": order, "seasonal_order": seasonal_order, "window_size": window_size},
    )
