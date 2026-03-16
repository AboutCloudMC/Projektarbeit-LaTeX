"""Metriken für Regressions- und Klassifikationsmodelle innerhalb der Pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    roc_auc_score,
    r2_score,
    recall_score,
)


@dataclass
class MetricResult:
    """Strukturiert alle Kennzahlen, die pro Modell/Ticker/Split gemessen werden."""

    ticker: str
    model_name: str
    split: str
    horizon: int
    task: str
    run_id: str | None = None
    run_ts: str | None = None
    eval_scope: str | None = None
    n_obs: int | None = None
    n_total: int | None = None
    n_pred: int | None = None
    rmse: float | None = None
    mae: float | None = None
    smape: float | None = None
    r2: float | None = None
    directional_accuracy: float | None = None
    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    roc_auc: float | None = None
    pr_auc: float | None = None
    logloss: float | None = None
    brier: float | None = None
    coverage: float | None = None
    base_rate: float | None = None
    uplift: float | None = None
    threshold: float | None = None


def _directional_accuracy(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Berechnet den Anteil korrekter Vorzeichen zwischen Ist- und Prognosewert.

    Nullwerte (Vorzeichen = 0) werden bei beiden Seiten ausgeschlossen, da ein
    Nullsignal keine echte Richtungsentscheidung darstellt.

    Parameter
    ---------
    y_true : pd.Series
        Tatsächliche Zielwerte (z. B. skalierte Renditen), index-synchron zur Prognose.
    y_pred : pd.Series
        Modellprognosen mit demselben Indexbereich.

    Returns
    -------
    float
        Anteil der Zeitpunkte, bei denen die Vorzeichen von Ist und Prognose übereinstimmen.
        Gibt ``NaN`` zurück, wenn keine validen Beobachtungen vorliegen.
    """
    actual = np.sign(y_true.to_numpy())
    predicted = np.sign(y_pred.to_numpy())
    valid_mask = ~np.isnan(actual) & ~np.isnan(predicted) & (actual != 0) & (predicted != 0)
    if not valid_mask.any():
        return float("nan")
    return float((actual[valid_mask] == predicted[valid_mask]).mean())


def _smape(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Symmetrische mittlere prozentuale Abweichung (sMAPE).

    Im Gegensatz zur klassischen MAPE ist sMAPE auch bei Zielwerten nahe Null
    numerisch stabil, da der Nenner den Durchschnitt der Absolutwerte von
    Ist- und Prognosewert verwendet.

    Parameter
    ---------
    y_true : pd.Series
        Tatsächliche Zielwerte.
    y_pred : pd.Series
        Prognostizierte Werte.

    Returns
    -------
    float
        Symmetrische prozentuale Abweichung im Bereich [0, 200].
        Gibt ``NaN`` zurück, wenn Ist- und Prognose gleichzeitig Null sind.
    """
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2 + 1e-8
    return float(np.mean(numerator / denominator) * 100)


def evaluate_regression(
    ticker: str,
    model_name: str,
    split_name: str,
    horizon: int,
    y_true: pd.Series,
    y_pred: pd.Series,
    run_id: str | None = None,
    run_ts: str | None = None,
    eval_scope: str | None = None,
) -> MetricResult:
    """Erstellt ein `MetricResult` mit klassischen Regressionsmetriken.

    Parameter
    ---------
    ticker : str
        Kürzel des Wertpapiers.
    model_name : str
        Name des Modells, um spätere Vergleiche zu erleichtern.
    split_name : str
        Datensplit (train/validation/test).
    horizon : int
        Prognosehorizont in Handelstagen.
    y_true : pd.Series
        Tatsächliche volatilitätsskalierte Zielwerte.
    y_pred : pd.Series
        Modellprognosen.

    Returns
    -------
    MetricResult
        Enthält RMSE, MAE, MAPE, R² und Richtungsgenauigkeit für diesen Lauf.

    Raises
    ------
    ValueError
        Wenn es keine Überlappung zwischen y_true und y_pred gibt.
    """
    y_true, y_pred = y_true.align(y_pred, join="inner")
    mask = y_pred.notna() & y_true.notna()
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if y_true.empty:
        raise ValueError("No overlapping observations between truth and prediction")

    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)
    smape = _smape(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    direction = _directional_accuracy(y_true, y_pred)

    return MetricResult(
        ticker=ticker,
        model_name=model_name,
        split=split_name,
        horizon=horizon,
        task="regression",
        run_id=run_id,
        run_ts=run_ts,
        eval_scope=eval_scope,
        n_obs=int(len(y_true)),
        rmse=rmse,
        mae=mae,
        smape=smape,
        r2=r2,
        directional_accuracy=direction,
    )


def apply_threshold(
    y_prob: pd.Series,
    threshold: float,
    min_probability: float | None = None,
) -> pd.Series:
    """Übersetzt Wahrscheinlichkeiten in diskrete Richtungen oder `NaN`.

    Parameter
    ---------
    y_prob : pd.Series
        Ausgegebenen Wahrscheinlichkeiten eines Klassifikators für "Up".
    threshold : float
        Ab welcher Wahrscheinlichkeit eine Long-Position angenommen wird.
    min_probability : float | None, optional
        Legt einen Mindestabstand zu 0.5 fest, andernfalls wird `NaN` (Abstention) gesetzt.

    Returns
    -------
    pd.Series
        Reihe aus {1, 0, NaN}, indexkompatibel zu `y_prob`.
    """
    preds = pd.Series(np.nan, index=y_prob.index, dtype=float)
    high_mask = y_prob >= threshold
    low_mask = y_prob <= 1 - threshold

    if min_probability is not None:
        strong_mask = (y_prob >= min_probability) | (y_prob <= 1 - min_probability)
        high_mask &= strong_mask
        low_mask &= strong_mask

    preds[high_mask] = 1
    preds[low_mask] = 0
    return preds


def evaluate_directional_classification(
    ticker: str,
    model_name: str,
    split_name: str,
    horizon: int,
    y_true: pd.Series,
    y_prob: pd.Series,
    threshold: float,
    min_probability: float | None = None,
    run_id: str | None = None,
    run_ts: str | None = None,
    eval_scope: str | None = None,
) -> MetricResult:
    """Bewertet eine Wahrscheinlichkeitsreihe anhand fester oder dynamischer Schwellen.

    Parameter
    ---------
    ticker : str
        Kürzel des Wertpapiers.
    model_name : str
        Interner Modellname.
    split_name : str
        Datensplit (train/validation/test).
    horizon : int
        Prognosehorizont in Handelstagen.
    y_true : pd.Series
        Binäre Richtungslabels.
    y_prob : pd.Series
        Modellwahrscheinlichkeiten für Klasse 1.
    threshold : float
        Schwelle, oberhalb derer Klasse 1 (Long) gewählt wird.
    min_probability : float | None, optional
        Mindestabstand zur Mitte, andernfalls Abstention.

    Returns
    -------
    MetricResult
        Objekt mit Accuracy-, F1-, LogLoss- und Coverage-Werten.
    """
    y_true, y_prob = y_true.align(y_prob, join="inner")
    mask = y_true.notna() & y_prob.notna()
    y_true = y_true[mask]
    y_prob = y_prob[mask]

    if y_true.empty:
        raise ValueError("No overlapping observations between truth and prediction")

    n_total = int(len(y_true))
    clipped_prob = np.clip(y_prob, 1e-6, 1 - 1e-6)
    roc_auc = pr_auc = float("nan")
    try:
        if len(np.unique(y_true)) > 1:
            roc_auc = roc_auc_score(y_true, clipped_prob)
            pr_auc = average_precision_score(y_true, clipped_prob)
    except ValueError:
        roc_auc = float("nan")
        pr_auc = float("nan")

    try:
        logloss_value = log_loss(y_true, clipped_prob)
    except ValueError:
        logloss_value = float("nan")
    brier = brier_score_loss(y_true, clipped_prob)

    preds = apply_threshold(y_prob, threshold, min_probability)
    valid_mask = preds.notna()
    n_pred = int(valid_mask.sum())
    coverage = float(valid_mask.mean())

    accuracy = precision = recall = f1 = directional_accuracy = float("nan")
    if valid_mask.any():
        y_true_eval = y_true[valid_mask]
        y_pred_eval = preds[valid_mask]
        accuracy = accuracy_score(y_true_eval, y_pred_eval)
        precision = precision_score(y_true_eval, y_pred_eval, zero_division=0)
        recall = recall_score(y_true_eval, y_pred_eval, zero_division=0)
        f1 = f1_score(y_true_eval, y_pred_eval, zero_division=0)
        directional_accuracy = accuracy

    base_rate_up = float(y_true.mean())
    base_rate = max(base_rate_up, 1 - base_rate_up)
    uplift = float("nan") if np.isnan(accuracy) else accuracy - base_rate

    return MetricResult(
        ticker=ticker,
        model_name=model_name,
        split=split_name,
        horizon=horizon,
        task="classification",
        run_id=run_id,
        run_ts=run_ts,
        eval_scope=eval_scope,
        n_obs=n_total,
        n_total=n_total,
        n_pred=n_pred,
        directional_accuracy=directional_accuracy,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        logloss=logloss_value,
        brier=brier,
        coverage=coverage,
        base_rate=base_rate,
        uplift=uplift,
        threshold=threshold,
    )


def metrics_to_frame(metrics: List[MetricResult]) -> pd.DataFrame:
    """Wandelt eine Liste strukturierter Kennzahlen in eine tabellarische Form.

    Parameter
    ---------
    metrics : List[MetricResult]
        Alle während eines Pipeline-Laufs gesammelten Kennzahlenobjekte.

    Returns
    -------
    pd.DataFrame
        Tabelle, die jede Kennzahlzeile als Record darstellt.
    """
    records = [m.__dict__ for m in metrics]
    return pd.DataFrame.from_records(records)
