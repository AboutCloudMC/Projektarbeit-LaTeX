"""Feature-Engineering und Datensatzaufbereitung für unterschiedliche Prognosehorizonte."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

LOGGER = logging.getLogger(__name__)


@dataclass
class SplitData:
    """Bündelt Features, Zielwerte und optionale Zusatzreihen für einen Datensplit."""

    X: pd.DataFrame
    y: pd.Series
    direction: pd.Series | None = None
    price: pd.Series | None = None


@dataclass
class HorizonDataset:
    """Enthält alle Splits und Metadaten für einen konkreten Prognosehorizont."""

    train: SplitData
    validation: SplitData
    test: SplitData
    scaler: StandardScaler | None
    feature_names: List[str]
    target_name: str
    direction_name: str


def _calendar_features(index: pd.Index) -> pd.DataFrame:
    """Leitet zyklische Kalendermerkmale (Wochen-/Monatsinfos) aus dem Index ab.

    Parameter
    ---------
    index : pd.Index
        Datumsindex der Kurszeitreihe.

    Returns
    -------
    pd.DataFrame
        Frame mit Tag der Woche, Kalenderwoche und Monat für jedes Datum.
    """
    return pd.DataFrame(
        {
            "day_of_week": index.dayofweek,
            "week_of_year": index.isocalendar().week.astype(int),
            "month": index.month,
        },
        index=index,
    )


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Berechnet den Relative-Strength-Index (RSI) über ein angegebenes Fenster.

    Parameter
    ---------
    series : pd.Series
        Preisserie, typischerweise der Schlusskurs.
    window : int, default 14
        Anzahl der Perioden für die geglätteten Auf- bzw. Abwärtsbewegungen.

    Returns
    -------
    pd.Series
        RSI-Werte im Bereich 0–100.
    """
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1 / window, min_periods=window).mean()
    roll_down = down.ewm(alpha=1 / window, min_periods=window).mean()
    rs = roll_up / roll_down
    return 100 - (100 / (1 + rs))


def _atr(frame: pd.DataFrame, window: int = 14) -> pd.Series:
    """Berechnet den Average True Range als einfache Volatilitätsnäherung.

    Parameter
    ---------
    frame : pd.DataFrame
        DataFrame mit mindestens den Spalten High, Low, Close.
    window : int, default 14
        Länge des Rollfensters für den gleitenden Durchschnitt.

    Returns
    -------
    pd.Series
        Durchschnittlicher True Range, beziehungsweise leere Serie, falls Spalten fehlen.
    """
    high = frame.get("High")
    low = frame.get("Low")
    close = frame.get("Close")
    if high is None or low is None or close is None:
        return pd.Series(index=frame.index, dtype=float)

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window=window).mean()


def _trend_strength(series: pd.Series, window: int) -> pd.Series:
    """Approximiert die Steigung einer linearen Regression innerhalb eines Rollfensters.

    Verwendet eine vektorisierte OLS-Formel über kumulative Summen anstelle von
    ``rolling.apply``, was bei großen Datenreihen ca. 100x schneller ist.

    Parameter
    ---------
    series : pd.Series
        Kurs- oder Feature-Serie.
    window : int
        Größe des Rollfensters, über das die Regressionssteigung geschätzt wird.

    Returns
    -------
    pd.Series
        Gleitende Schätzung der Steigung; ``NaN``, solange das Fenster nicht voll ist.
    """
    # OLS-Steigung = (n * sum(x*y) - sum(x) * sum(y)) / (n * sum(x^2) - sum(x)^2)
    # mit x = 0..window-1 sind sum(x) und sum(x^2) Konstanten.
    n = window
    sum_x = n * (n - 1) / 2
    sum_x2 = n * (n - 1) * (2 * n - 1) / 6
    denom = n * sum_x2 - sum_x * sum_x

    # Kumulative Summen für die rollierende Berechnung
    x_weights = np.arange(n, dtype=float)
    # sum(x_i * y_i) über ein Rollfenster = Korrelation mit linearem Trend
    sum_xy = series.rolling(window=n).apply(lambda y: np.dot(x_weights, y), raw=True)
    sum_y = series.rolling(window=n).sum()

    slope = (n * sum_xy - sum_x * sum_y) / denom
    return slope


def _realized_vol(log_returns: pd.Series, window: int) -> pd.Series:
    """Schätzt realisierte Volatilität als Standardabweichung der Log-Renditen.

    Parameter
    ---------
    log_returns : pd.Series
        Logarithmische Einperioden-Renditen.
    window : int
        Länge des Rollfensters für die Standardabweichung.

    Returns
    -------
    pd.Series
        Volatilitätsschätzung; `NaN`, solange nicht genügend Daten vorhanden sind.
    """
    return log_returns.rolling(window=window).std()


def engineer_features(
    frame: pd.DataFrame,
    config: Dict[str, object],
) -> pd.DataFrame:
    """Erstellt technische Indikatoren, Verzögerungen und Kalendermerkmale gemäß Konfiguration.

    Parameter
    ---------
    frame : pd.DataFrame
        Historische Marktdaten (Open, High, Low, Close, Volume, ...).
    config : Dict[str, object]
        Projektkonfiguration, deren Abschnitt `features` die gewünschten Indikatoren beschreibt.

    Returns
    -------
    pd.DataFrame
        Umfangreicher Feature-Frame ohne fehlende Werte.
    """
    features_cfg = config.get("features", {})
    windows = features_cfg.get("rolling_windows", [5, 10, 20])
    max_lag = int(features_cfg.get("max_lag", 20))
    include_rsi = bool(features_cfg.get("include_rsi", True))
    include_momentum = bool(features_cfg.get("include_momentum", True))
    include_volume_ratio = bool(features_cfg.get("include_volume_ratio", True))
    target_column = features_cfg.get("target_column", "Close")
    vol_window = int(features_cfg.get("vol_window", 20))

    dataset = frame.copy()
    dataset.index = pd.to_datetime(dataset.index)

    dataset["log_close"] = np.log(dataset[target_column])
    dataset["log_return_1"] = dataset["log_close"].diff()
    dataset["return_1"] = dataset[target_column].pct_change()
    # Historische Zielwerte dienen als zusätzliche Regressoren.
    for lag in range(1, max_lag + 1):
        dataset[f"{target_column.lower()}_lag_{lag}"] = dataset[target_column].shift(lag)

    for window in windows:
        # Rolling-Statistiken liefern Mittelwert, Volatilität sowie Extremwerte je Fenster.
        roll = dataset[target_column].rolling(window=window)
        dataset[f"rolling_mean_{window}"] = roll.mean()
        dataset[f"rolling_std_{window}"] = roll.std()
        dataset[f"rolling_min_{window}"] = roll.min()
        dataset[f"rolling_max_{window}"] = roll.max()

    if include_momentum:
        for window in windows:
            # Momentum misst die prozentuale Veränderung über größere Zeitfenster.
            dataset[f"momentum_{window}"] = dataset[target_column].pct_change(periods=window)

    if include_rsi:
        rsi_window = max(14, windows[0])
        dataset[f"rsi_{rsi_window}"] = _rsi(dataset[target_column], window=rsi_window)

    if include_volume_ratio and "Volume" in dataset:
        dataset["volume_ratio"] = dataset["Volume"] / dataset["Volume"].rolling(window=20).mean()

    dataset["realized_vol"] = _realized_vol(dataset["log_return_1"], vol_window)
    dataset["atr"] = _atr(dataset, window=vol_window)
    dataset["trend_strength"] = _trend_strength(dataset[target_column], window=max(10, vol_window))
    dataset["momentum_z"] = dataset["log_return_1"] / (dataset["realized_vol"] + 1e-8)
    dataset["abs_return"] = dataset["log_return_1"].abs()
    dataset["volatility_ratio"] = dataset["realized_vol"] / (dataset["realized_vol"].rolling(vol_window).mean())

    calendar = _calendar_features(dataset.index)
    dataset = dataset.join(calendar)

    # Feature-Matrix nur dort behalten, wo alle Indikatoren gefüllt werden konnten.
    dataset = dataset.dropna()
    return dataset


def _filter_low_variance_features(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Entfernt Features mit zu geringer Varianz, sofern ein Schwellwert gesetzt ist.

    Parameter
    ---------
    frame : pd.DataFrame
        Kandidaten-Features vor der Modellierung.
    threshold : float
        Minimale Varianz, die eine Spalte haben muss, um beibehalten zu werden.

    Returns
    -------
    pd.DataFrame
        Gefilterter Frame; unverändert, wenn kein Feature den Schwellwert erfüllt.
    """
    if threshold <= 0:
        return frame
    variances = frame.var()
    keep_cols = variances[variances > threshold].index
    if len(keep_cols) == 0:
        return frame
    dropped = sorted(set(frame.columns) - set(keep_cols))
    if dropped:
        LOGGER.debug("Dropping %s low-variance features", len(dropped))
    return frame[keep_cols]


def _chronological_split(
    X: pd.DataFrame,
    y: pd.Series,
    direction: pd.Series | None,
    price: pd.Series,
    splits_cfg: Dict[str, float],
) -> Dict[str, SplitData]:
    """Teilt die Daten seriell in Train/Val/Test auf und behält Reihenfolge bei.

    Parameter
    ---------
    X : pd.DataFrame
        Feature-Matrix.
    y : pd.Series
        Regressionstarget (z. B. volatilitätsskalierte Renditen).
    direction : pd.Series | None
        Optionales Klassifikationslabel.
    price : pd.Series
        Rohpreis, wird später zur Skalierung von ARIMA-Signalen genutzt.
    splits_cfg : Dict[str, float]
        Konfiguration mit `train_ratio`, `validation_ratio`, `test_ratio`.

    Returns
    -------
    Dict[str, SplitData]
        Zeitlich aufeinanderfolgende Splits; jeder Split enthält `X`, `y`, optional `direction` und `price`.

    Raises
    ------
    ValueError
        Wenn weniger als 50 Beobachtungen vorhanden sind (instabile Splits).
    """
    n_obs = len(X)
    if n_obs < 50:
        raise ValueError("Need at least 50 observations for stable splits")

    train_ratio = float(splits_cfg.get("train_ratio", 0.7))
    val_ratio = float(splits_cfg.get("validation_ratio", 0.15))
    test_ratio = float(splits_cfg.get("test_ratio", 0.15))
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        LOGGER.warning("Split ratios do not sum to 1.0; data will be normalized proportionally")
        # Falls sich Rundungsfehler eingeschlichen haben, werden die Gewichte normiert.
        total = train_ratio + val_ratio + test_ratio
        train_ratio /= total
        val_ratio /= total
        test_ratio /= total

    train_end = int(n_obs * train_ratio)
    val_end = train_end + int(n_obs * val_ratio)
    if val_end >= n_obs:
        val_end = n_obs - max(1, int(n_obs * test_ratio))
        LOGGER.warning(
            "Split-Verhältnisse ergeben keinen Platz für den Test-Split; "
            "Test-Set wird auf %d Beobachtungen reduziert",
            n_obs - val_end,
        )

    splits = {
        "train": SplitData(
            X=X.iloc[:train_end],
            y=y.iloc[:train_end],
            direction=None if direction is None else direction.iloc[:train_end],
            price=price.iloc[:train_end],
        ),
        "validation": SplitData(
            X=X.iloc[train_end:val_end],
            y=y.iloc[train_end:val_end],
            direction=None if direction is None else direction.iloc[train_end:val_end],
            price=price.iloc[train_end:val_end],
        ),
        "test": SplitData(
            X=X.iloc[val_end:],
            y=y.iloc[val_end:],
            direction=None if direction is None else direction.iloc[val_end:],
            price=price.iloc[val_end:],
        ),
    }

    return splits


def _scale_splits(splits: Dict[str, SplitData]) -> tuple[Dict[str, SplitData], StandardScaler]:
    """Standardisiert alle Features anhand der Statistik aus dem Trainingssplit.

    Parameter
    ---------
    splits : Dict[str, SplitData]
        Train/Val/Test-Splits mit unveränderten Features.

    Returns
    -------
    tuple
        - Dict[str, SplitData]: Splits mit skalierten `X`-Matrizen.
        - StandardScaler: trainierter Scaler, falls spätere Transformationen nötig sind.
    """
    scaler = StandardScaler()
    train_X = splits["train"].X
    scaler.fit(train_X)

    for key, split in splits.items():
        scaled = scaler.transform(split.X)
        splits[key] = SplitData(
            X=pd.DataFrame(scaled, columns=train_X.columns, index=split.X.index),
            y=split.y,
            direction=split.direction,
            price=split.price,
        )

    return splits, scaler


def _build_targets(
    frame: pd.DataFrame,
    horizon: int,
    target_column: str,
    vol_window: int,
    dead_zone_kappa: float,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Erzeugt alle Zielreihen für einen bestimmten Prognosehorizont.

    Parameter
    ---------
    frame : pd.DataFrame
        Feature-Frame mit realisierter Volatilität.
    horizon : int
        Prognosehorizont in Perioden.
    target_column : str
        Spaltenname des Preises (z. B. "Close").
    vol_window : int
        Fenster, das für die Volatilitätsskalierung genutzt wurde (für Konsistenz).
    dead_zone_kappa : float
        Schwellenwert, ab dem eine Richtung als Long/Short gewertet wird; dazwischen Abstention.

    Returns
    -------
    Tuple[pd.Series, pd.Series, pd.Series]
        (volatilitätsskalierte Rendite, Richtungslabel, rohe Log-Rendite).
    """
    future_close = frame[target_column].shift(-horizon)
    current_close = frame[target_column]
    future_log = np.log(future_close)
    current_log = np.log(current_close)
    log_return = future_log - current_log

    realized_vol = frame["realized_vol"]
    horizon_vol = realized_vol * np.sqrt(horizon)
    scaled_return = log_return / (horizon_vol + 1e-8)

    direction = pd.Series(np.nan, index=frame.index)
    direction.loc[scaled_return > dead_zone_kappa] = 1
    direction.loc[scaled_return < -dead_zone_kappa] = 0

    return scaled_return, direction, log_return


def prepare_datasets(frame: pd.DataFrame, config: Dict[str, object]) -> Dict[int, HorizonDataset]:
    """Bereitet pro Prognosehorizont skalierte Trainings-/Validierungs-/Testsätze vor.

    Parameter
    ---------
    frame : pd.DataFrame
        Vollständige Ticker-Historie.
    config : Dict[str, object]
        Projektkonfiguration.

    Returns
    -------
    Dict[int, HorizonDataset]
        Mapping von Prognosehorizont zu strukturiertem Datensatzpaket.
    """
    features_cfg = config.get("features", {})
    splits_cfg = config.get("splits", {})
    target_column = features_cfg.get("target_column", "Close")
    horizons = features_cfg.get("forecast_horizons", [5])
    scale_flag = bool(features_cfg.get("scale_features", True))
    variance_threshold = float(features_cfg.get("variance_threshold", 0.0))
    vol_window = int(features_cfg.get("vol_window", 20))
    dead_zone_kappa = float(features_cfg.get("dead_zone_kappa", 0.2))

    engineered = engineer_features(frame, config)
    prepared: Dict[int, HorizonDataset] = {}

    for horizon in horizons:
        scaled_name = f"vol_scaled_return_t+{horizon}"
        horizon_frame = engineered.copy()
        scaled_return, direction, raw_return = _build_targets(
            horizon_frame,
            horizon,
            target_column,
            vol_window,
            dead_zone_kappa,
        )
        direction_name = f"direction_t+{horizon}"
        horizon_frame[scaled_name] = scaled_return
        horizon_frame[direction_name] = direction
        horizon_frame[f"log_return_t+{horizon}"] = raw_return
        # Die Dead-Zone gilt nur für Klassifikation. Regressionsdaten dürfen dadurch
        # nicht verworfen werden.
        horizon_frame = horizon_frame.dropna(subset=[scaled_name])
        # Targets entfernen, sodass nur reine Input-Features übrig bleiben.
        feature_cols = [
            col for col in horizon_frame.columns if col not in {scaled_name, direction_name, f"log_return_t+{horizon}"}
        ]

        X = horizon_frame[feature_cols]
        X = _filter_low_variance_features(X, variance_threshold)
        y = horizon_frame[scaled_name]
        direction = horizon_frame[direction_name]
        price = horizon_frame[target_column]
        splits = _chronological_split(X, y, direction, price, splits_cfg)

        scaler = None
        if scale_flag:
            splits, scaler = _scale_splits(splits)

        prepared[horizon] = HorizonDataset(
            train=splits["train"],
            validation=splits["validation"],
            test=splits["test"],
            scaler=scaler,
            feature_names=list(X.columns),
            target_name=scaled_name,
            direction_name=direction_name,
        )

    return prepared
