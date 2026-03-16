"""Orchestriert den kompletten CastMark-Workflow von Daten bis zu Kennzahlen."""

from __future__ import annotations

import hashlib
import json
import logging
import random
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, f1_score

from castmark import artifacts as artifacts_module
from castmark import config as cfg
from castmark import data as data_module
from castmark import evaluation
from castmark import features
from castmark import models
from castmark.logging_utils import setup_logging

LOGGER = logging.getLogger(__name__)


def _save_metrics(metrics: List[evaluation.MetricResult], output_dir: Path, run_dir: Path) -> Path:
    """Persistiert alle Kennzahlen als CSV im Run- und Artefaktverzeichnis.

    Parameter
    ---------
    metrics : List[evaluation.MetricResult]
        Gesammelte Kennzahlen pro Modell/Ticker/Split.
    output_dir : Path
        Übergeordnetes Artefaktverzeichnis; hier wird eine ``metrics.csv``
        als "latest"-Snapshot abgelegt.
    run_dir : Path
        Lauf-spezifisches Verzeichnis, in dem die primäre ``metrics.csv`` gespeichert wird.

    Returns
    -------
    Path
        Vollständiger Pfad zur gespeicherten CSV-Datei im Run-Verzeichnis.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_frame = evaluation.metrics_to_frame(metrics)
    metrics_path = run_dir / "metrics.csv"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_frame.to_csv(metrics_path, index=False)
    LOGGER.info("Stored metrics at %s", metrics_path)
    # Zusätzlich eine "latest"-Kopie anlegen, damit alte Auswertungsskripte weiter funktionieren.
    latest_path = output_dir / "metrics.csv"
    metrics_frame.to_csv(latest_path, index=False)
    return metrics_path


def _git_commit_hash() -> str | None:
    """Liest den aktuellen Git-Commit-Hash aus, damit Läufe später reproduzierbar sind.

    Returns
    -------
    str | None
        Vollständiger SHA-1-Hash oder None, wenn kein Git-Repo vorhanden ist.
    """
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        return commit
    except Exception:
        return None


def _split_attribute(dataset: features.HorizonDataset, attr: str) -> Dict[str, pd.Series | pd.DataFrame | None]:
    """Extrahiert ein bestimmtes Attribut (X, y, direction, price) aus allen Splits.

    Parameter
    ---------
    dataset : features.HorizonDataset
        Struktur mit train/validation/test-Splits.
    attr : str
        Name des Split-Attributes (z. B. "X" oder "direction").

    Returns
    -------
    Dict[str, pd.Series | pd.DataFrame | None]
        Mapping Split -> Wert/None, abhängig davon, ob das Attribut existiert.
    """
    return {split: getattr(getattr(dataset, split), attr, None) for split in ("train", "validation", "test")}


def _build_meta_features(
    dataset: features.HorizonDataset,
    return_models: Dict[str, models.ModelPrediction],
    arima_signals: Dict[str, pd.Series],
    base_probs: Dict[str, pd.Series] | None = None,
) -> Dict[str, pd.DataFrame]:
    """Stapelt Return-Modelle, ARIMA-Signale und Kernfeatures zu Meta-Features.

    Parameter
    ---------
    dataset : features.HorizonDataset
        Enthält Eingangsfeatures pro Split.
    return_models : Dict[str, ModelPrediction]
        Prognosen verschiedener Regressionsmodelle.
    arima_signals : Dict[str, pd.Series]
        Optional verfügbare ARIMA-Baselines separat für jeden Split.
    base_probs : Dict[str, pd.Series], optional
        Wahrscheinlichkeitsausgaben des Basis-Klassifikators.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Meta-Feature-Matrizen für train/validation/test.
    """
    # Schlüsselfeatures dynamisch aus den vorhandenen Spalten ableiten,
    # damit Änderungen an rolling_windows automatisch berücksichtigt werden.
    available = set(dataset.train.X.columns)
    key_columns = sorted(c for c in available if c.startswith("momentum_") and c != "momentum_z")
    for extra in ("momentum_z", "realized_vol"):
        if extra in available:
            key_columns.append(extra)
    meta_frames: Dict[str, pd.DataFrame] = {}

    for split_name in ("train", "validation", "test"):
        split = getattr(dataset, split_name)
        # Leerer Frame mit dem Split-Index als Basis, damit alle Spalten später index-kompatibel sind.
        frame = pd.DataFrame(index=split.X.index)

        for name, prediction in return_models.items():
            preds = prediction.predictions.get(split_name)
            if split_name == "train" and prediction.oof_predictions:
                oof_train = prediction.oof_predictions.get("train")
                if oof_train is not None and not oof_train.empty:
                    preds = oof_train
            if preds is not None:
                frame[f"{name}_signal"] = preds.reindex(frame.index)

        if arima_signals.get(split_name) is not None:
            frame["arima_signal"] = arima_signals[split_name].reindex(frame.index)

        if base_probs and base_probs.get(split_name) is not None:
            frame["base_prob"] = base_probs[split_name].reindex(frame.index)

        for col in key_columns:
            if col in split.X.columns:
                frame[col] = split.X[col]

        meta_frames[split_name] = frame

    return meta_frames


def _compute_arima_signals(
    arima_result: models.ModelPrediction | None,
    dataset: features.HorizonDataset,
    horizon: int,
) -> Dict[str, pd.Series]:
    """Berechnet normierte ARIMA-Signale zu jedem Split, falls Baseline aktiv ist.

    Parameter
    ---------
    arima_result : ModelPrediction | None
        Ergebnis der ARIMA-Baseline (falls vorhanden).
    dataset : HorizonDataset
        Quelle für Preise und Volatilitäten.
    horizon : int
        Prognosedistanz, dient zur Volatilitätsskalierung.

    Returns
    -------
    Dict[str, pd.Series]
        Volatilitätsskalierte Signals pro Split. Fehlende Splits werden ausgelassen.
    """
    if not arima_result:
        return {}

    signals: Dict[str, pd.Series] = {}
    for split_name in ("train", "validation", "test"):
        preds = arima_result.predictions.get(split_name)
        if preds is None or preds.empty:
            continue
        split = getattr(dataset, split_name)
        prices = split.price
        if prices is None:
            continue
        # Nur Zeitpunkte verwenden, für die sowohl Preis als auch ARIMA-Signal existiert.
        aligned_pred = preds.reindex(prices.index).dropna()
        if aligned_pred.empty:
            continue
        aligned_price = prices.loc[aligned_pred.index]
        vol = split.X.loc[aligned_pred.index, "realized_vol"] if "realized_vol" in split.X.columns else None
        if vol is None:
            continue
        log_return = np.log(aligned_pred) - np.log(aligned_price)
        signal = log_return / (vol * np.sqrt(horizon) + 1e-8)
        signals[split_name] = signal
    return signals


def _search_threshold(
    y_true: pd.Series,
    y_prob: pd.Series,
    classification_cfg: Dict[str, object],
) -> float:
    """Durchsucht einen Schwellwert-Grid mit optional harter Coverage-Bedingung.

    Parameter
    ---------
    y_true : pd.Series
        Binäre Validierungslabels.
    y_prob : pd.Series
        Validierungswahrscheinlichkeiten.
    classification_cfg : Dict[str, object]
        Abschnitt `classification` der Konfiguration (Threshold-Grid, Coverage-Ziel, Abstain).

    Returns
    -------
    float
        Gewählter Schwellenwert im Bereich [0.5, max_threshold].
    """
    y_true, y_prob = y_true.align(y_prob, join="inner")
    valid_mask = y_true.notna() & y_prob.notna()
    y_true = y_true[valid_mask]
    y_prob = y_prob[valid_mask]
    if y_true.empty:
        return 0.5

    search_cfg = classification_cfg.get("threshold_search", {})
    grid_size = max(2, int(search_cfg.get("grid_size", 50)))
    metric = str(search_cfg.get("metric", "youden")).lower()
    max_threshold = min(0.999, float(search_cfg.get("max_threshold", 0.9)))
    max_threshold = max(0.5, max_threshold)
    hard_coverage_constraint = bool(search_cfg.get("hard_coverage_constraint", False))
    coverage_fallback = str(search_cfg.get("coverage_fallback", "nearest")).lower()
    thresholds = np.linspace(0.5, max_threshold, grid_size)
    min_probability = classification_cfg.get("abstain", {}).get("min_probability")
    coverage_target_raw = classification_cfg.get("coverage_target")
    coverage_target = None if coverage_target_raw is None else float(coverage_target_raw)
    if coverage_target is not None:
        coverage_target = min(max(coverage_target, 0.0), 1.0)
    coverage_tolerance = max(0.0, float(classification_cfg.get("coverage_tolerance", 0.05)))

    best_unconstrained_score = float("-inf")
    best_unconstrained_threshold = 0.5
    best_soft_score = float("-inf")
    best_soft_threshold = 0.5
    best_within_score = float("-inf")
    best_within_threshold: float | None = None
    best_nearest_delta = float("inf")
    best_nearest_score = float("-inf")
    best_nearest_threshold: float | None = None

    for threshold in thresholds:
        preds = evaluation.apply_threshold(y_prob, threshold, min_probability)
        mask = preds.notna()
        if not mask.any():
            continue
        y_eval = y_true[mask]
        y_pred = preds[mask]
        coverage = mask.mean()

        if metric == "f1":
            score = f1_score(y_eval, y_pred, zero_division=0)
        elif metric == "accuracy":
            score = accuracy_score(y_eval, y_pred)
        else:
            tp = ((y_pred == 1) & (y_eval == 1)).sum()
            fn = ((y_pred == 0) & (y_eval == 1)).sum()
            fp = ((y_pred == 1) & (y_eval == 0)).sum()
            tn = ((y_pred == 0) & (y_eval == 0)).sum()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            score = tpr - fpr

        if score > best_unconstrained_score:
            best_unconstrained_score = score
            best_unconstrained_threshold = threshold

        if coverage_target is None:
            continue

        coverage_delta = abs(coverage - coverage_target)
        if coverage_delta <= coverage_tolerance and score > best_within_score:
            best_within_score = score
            best_within_threshold = threshold

        if coverage_delta < best_nearest_delta or (
            np.isclose(coverage_delta, best_nearest_delta) and score > best_nearest_score
        ):
            best_nearest_delta = coverage_delta
            best_nearest_score = score
            best_nearest_threshold = threshold

        # Weiche Optimierung: gute Scores werden bevorzugt, Coverage-Abweichung wird penalisiert.
        soft_score = score - coverage_delta
        if soft_score > best_soft_score:
            best_soft_score = soft_score
            best_soft_threshold = threshold

    if coverage_target is None:
        return best_unconstrained_threshold
    if best_within_threshold is not None:
        return best_within_threshold
    if hard_coverage_constraint:
        if coverage_fallback == "score":
            return best_unconstrained_threshold
        if best_nearest_threshold is not None:
            return best_nearest_threshold
        return best_unconstrained_threshold
    return best_soft_threshold


def run(config_path: str = "config/defaults.yaml") -> pd.DataFrame:
    """Zentrale Pipeline: Daten laden, Modelle trainieren, Kennzahlen speichern.

    Parameter
    ---------
    config_path : str, default "config/defaults.yaml"
        Pfad zur YAML-Konfiguration.

    Returns
    -------
    pd.DataFrame
        Tabelle aller Kennzahlen für alle Ticker/Horizonte.
    """
    configuration = cfg.load_config(config_path)
    cfg.ensure_directories(configuration)

    seed = int(configuration.get("experiment", {}).get("seed", 42))
    np.random.seed(seed)
    random.seed(seed)

    now = datetime.now()
    run_ts = now.isoformat(timespec="seconds")
    run_id = now.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(configuration.get("experiment", {}).get("output_dir", "artifacts"))
    run_dir = output_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    config_payload = json.dumps(configuration, sort_keys=True, default=str).encode("utf-8")
    config_hash = hashlib.sha256(config_payload).hexdigest()

    with (run_dir / "config.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(configuration, fh, sort_keys=False)

    log_dir = configuration.get("logging", {}).get("log_dir", "artifacts/logs")
    log_level = configuration.get("logging", {}).get("level", "INFO")
    setup_logging(log_dir, log_level)

    LOGGER.info("Starting pipeline with config: %s", config_path)
    market_data = data_module.load_market_data(configuration)
    LOGGER.info("Loaded data for %s tickers", list(market_data))

    data_summary = {
        ticker: {
            "rows": int(len(frame)),
            "start": str(frame.index.min().date()),
            "end": str(frame.index.max().date()),
        }
        for ticker, frame in market_data.items()
    }
    run_meta = {
        "run_id": run_id,
        "run_ts": run_ts,
        "config_path": config_path,
        "config_hash": config_hash,
        "git_commit": _git_commit_hash(),
        "data_summary": data_summary,
    }
    with (run_dir / "run_meta.json").open("w", encoding="utf-8") as fh:
        json.dump(run_meta, fh, indent=2)

    metrics: List[evaluation.MetricResult] = []
    model_cfg = configuration.get("models", {})
    target_column = configuration.get("features", {}).get("target_column", "Close")
    splits_cfg = configuration.get("splits", {})
    horizons = configuration.get("features", {}).get("forecast_horizons", [5])
    classification_cfg = configuration.get("classification", {})
    classifiers_cfg = model_cfg.get("classifiers", {})
    calibration_cfg = classifiers_cfg.get("calibration", {})
    min_probability = classification_cfg.get("abstain", {}).get("min_probability")

    for ticker, frame in market_data.items():
        LOGGER.info("Processing ticker %s", ticker)
        horizon_datasets = features.prepare_datasets(frame, configuration)
        engineered = features.engineer_features(frame, configuration)
        arima_order_cache: tuple[int, int, int] | None = None
        arima_seasonal_cache: tuple[int, int, int, int] | None = None
        arima_forecast_cache: Dict[
            tuple[object, tuple[int, int, int], tuple[int, int, int, int], int], np.ndarray
        ] = {}
        max_arima_horizon = max(horizons) if horizons else None

        for horizon, dataset in horizon_datasets.items():
            LOGGER.info("Training return regressors | ticker=%s horizon=%s", ticker, horizon)
            # Schritt 1: Regressionsmodelle für skalierte Renditen.
            return_models = models.train_return_regressors(dataset, model_cfg)

            LOGGER.info("Generating ARIMA walk-forward baseline | ticker=%s horizon=%s", ticker, horizon)
            arima_result = models.generate_arima_walk_forward_predictions(
                frame=frame,
                dataset=dataset,
                horizon=horizon,
                target_column=target_column,
                splits_cfg=splits_cfg,
                model_cfg=model_cfg,
                selected_order=arima_order_cache,
                selected_seasonal_order=arima_seasonal_cache,
                forecast_cache=arima_forecast_cache,
                max_forecast_steps=max_arima_horizon,
            )
            if arima_result and arima_result.extras:
                order_extra = arima_result.extras.get("order")
                seasonal_extra = arima_result.extras.get("seasonal_order")
                if order_extra is not None:
                    arima_order_cache = tuple(order_extra)
                if seasonal_extra is not None:
                    arima_seasonal_cache = tuple(seasonal_extra)
            arima_signals = _compute_arima_signals(arima_result, dataset, horizon)

            for split_name in ("validation", "test"):
                y_true = getattr(dataset, split_name).y
                regression_preds: Dict[str, pd.Series] = {}

                for model_name, result in return_models.items():
                    y_pred = result.predictions.get(split_name)
                    if y_pred is None or y_pred.empty:
                        continue
                    regression_preds[model_name] = y_pred
                    metric = evaluation.evaluate_regression(
                        ticker=ticker,
                        model_name=model_name,
                        split_name=split_name,
                        horizon=horizon,
                        y_true=y_true,
                        y_pred=y_pred,
                        run_id=run_id,
                        run_ts=run_ts,
                        eval_scope="model_index",
                    )
                    metrics.append(metric)

                zero_pred = pd.Series(0.0, index=y_true.index, name="prediction")
                regression_preds["zero_return"] = zero_pred
                metrics.append(
                    evaluation.evaluate_regression(
                        ticker=ticker,
                        model_name="zero_return",
                        split_name=split_name,
                        horizon=horizon,
                        y_true=y_true,
                        y_pred=zero_pred,
                        run_id=run_id,
                        run_ts=run_ts,
                        eval_scope="model_index",
                    )
                )

                arima_signal = arima_signals.get(split_name)
                if arima_signal is not None and not arima_signal.empty:
                    regression_preds[f"arima_signal_h{horizon}"] = arima_signal
                    metrics.append(
                        evaluation.evaluate_regression(
                            ticker=ticker,
                            model_name=f"arima_signal_h{horizon}",
                            split_name=split_name,
                            horizon=horizon,
                            y_true=y_true,
                            y_pred=arima_signal,
                            run_id=run_id,
                            run_ts=run_ts,
                            eval_scope="model_index",
                        )
                    )

                common_index = y_true.index
                for series in regression_preds.values():
                    common_index = common_index.intersection(series.dropna().index)

                if len(common_index) > 0:
                    for model_name, series in regression_preds.items():
                        try:
                            metric = evaluation.evaluate_regression(
                                ticker=ticker,
                                model_name=model_name,
                                split_name=split_name,
                                horizon=horizon,
                                y_true=y_true.loc[common_index],
                                y_pred=series.reindex(common_index),
                                run_id=run_id,
                                run_ts=run_ts,
                                eval_scope="common_index",
                            )
                        except ValueError:
                            continue
                        metrics.append(metric)

            feature_splits = _split_attribute(dataset, "X")
            direction_splits = _split_attribute(dataset, "direction")

            LOGGER.info("Training base logistic classifier | ticker=%s horizon=%s", ticker, horizon)
            # Schritt 2: Richtungs-Klassifikatoren trainieren (erst Basis-, später Meta-Modell).
            base_classifier = models.train_classifier_from_features(
                feature_splits=feature_splits,
                direction_splits=direction_splits,
                classifier_cfg=classifiers_cfg.get("base_logistic", {}),
                calibration_cfg=calibration_cfg,
                name="base_logistic",
            )

            classifier_predictions: Dict[str, models.ModelPrediction] = {}
            if base_classifier:
                classifier_predictions["base_logistic"] = base_classifier

            train_direction = direction_splits.get("train")
            if train_direction is not None and not train_direction.dropna().empty:
                base_rate = float(train_direction.mean())
                baseline_preds = {
                    split: pd.Series(base_rate, index=split_data.index, name="probability")
                    for split, split_data in direction_splits.items()
                    if split_data is not None
                }
                classifier_predictions["baseline_rate"] = models.ModelPrediction(
                    name="baseline_rate",
                    predictions=baseline_preds,
                    model=None,
                    extras={"base_rate": base_rate},
                )

            # Meta-Features bündeln die stärksten Signale der Rückgabe-Modelle sowie ARIMA.
            meta_features = _build_meta_features(
                dataset=dataset,
                return_models=return_models,
                arima_signals=arima_signals,
                base_probs=base_classifier.predictions if base_classifier else None,
            )

            if (
                classifiers_cfg.get("meta_logistic", {}).get("enabled", True)
                and meta_features["train"].shape[1] > 0
            ):
                LOGGER.info("Training meta logistic classifier | ticker=%s horizon=%s", ticker, horizon)
                meta_classifier = models.train_classifier_from_features(
                    feature_splits=meta_features,
                    direction_splits=direction_splits,
                    classifier_cfg=classifiers_cfg.get("meta_logistic", {}),
                    calibration_cfg=calibration_cfg,
                    name="meta_logistic",
                )
                if meta_classifier:
                    classifier_predictions["meta_logistic"] = meta_classifier

            classifier_thresholds: Dict[str, float] = {}
            for name, clf_prediction in classifier_predictions.items():
                val_direction = direction_splits.get("validation")
                val_probs = clf_prediction.predictions.get("validation")
                if val_direction is None or val_direction.empty or val_probs is None or val_probs.empty:
                    threshold = 0.5
                else:
                    threshold = _search_threshold(val_direction, val_probs, classification_cfg)
                classifier_thresholds[name] = threshold

                for split_name in ("validation", "test"):
                    y_true = direction_splits.get(split_name)
                    y_prob = clf_prediction.predictions.get(split_name)
                    if y_true is None or y_prob is None or y_prob.empty:
                        continue
                    try:
                        metric = evaluation.evaluate_directional_classification(
                            ticker=ticker,
                            model_name=name,
                            split_name=split_name,
                            horizon=horizon,
                            y_true=y_true,
                            y_prob=y_prob,
                            threshold=threshold,
                            min_probability=min_probability,
                            run_id=run_id,
                            run_ts=run_ts,
                            eval_scope="model_index",
                        )
                    except ValueError:
                        continue
                    metrics.append(metric)

            # Artefakte für diesen Ticker/Horizont persistieren.
            run_artifacts = {
                "return_predictions": {
                    name: dict(pred.predictions) for name, pred in return_models.items()
                },
                "arima_predictions": dict(arima_result.predictions) if arima_result else None,
                "arima_extras": arima_result.extras if arima_result else None,
                "classifier_probabilities": {
                    name: dict(pred.predictions) for name, pred in classifier_predictions.items()
                },
                "classifier_thresholds": classifier_thresholds,
                "direction": direction_splits,
                "y_true": {split: getattr(dataset, split).y for split in ("train", "validation", "test")},
                "realized_vol": engineered["realized_vol"],
                "target_prices": engineered[target_column],
                "full_prices": frame[target_column].dropna().sort_index(),
                "min_probability": min_probability,
            }
            artifacts_module.save_run_artifacts(run_dir, ticker, horizon, run_artifacts)

    metrics_df = evaluation.metrics_to_frame(metrics)
    _save_metrics(metrics, output_dir, run_dir)
    return metrics_df
