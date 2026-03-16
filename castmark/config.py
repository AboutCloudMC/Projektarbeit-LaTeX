"""Hilfsfunktionen zum Laden und Validieren von Konfigurationsdateien."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import yaml

LOGGER = logging.getLogger(__name__)

# Erwartete Top-Level-Schlüssel mit ihren erwarteten Typen.
_REQUIRED_SECTIONS: Dict[str, type] = {
    "data": dict,
    "features": dict,
    "splits": dict,
    "models": dict,
}

_OPTIONAL_SECTIONS: Dict[str, type] = {
    "experiment": dict,
    "logging": dict,
    "classification": dict,
}


def _validate_config(config: Dict[str, Any]) -> List[str]:
    """Prüft die Konfiguration auf fehlende Pflichtabschnitte und offensichtliche Fehler.

    Parameter
    ---------
    config : Dict[str, Any]
        Geladener Konfigurationsbaum.

    Returns
    -------
    List[str]
        Liste menschenlesbarer Fehlermeldungen. Leer, wenn alles korrekt ist.
    """
    errors: List[str] = []

    for section, expected_type in _REQUIRED_SECTIONS.items():
        if section not in config:
            errors.append(f"Pflichtabschnitt '{section}' fehlt in der Konfiguration")
        elif not isinstance(config[section], expected_type):
            errors.append(
                f"Abschnitt '{section}' muss vom Typ {expected_type.__name__} sein, "
                f"ist aber {type(config[section]).__name__}"
            )

    for section, expected_type in _OPTIONAL_SECTIONS.items():
        if section in config and not isinstance(config[section], expected_type):
            errors.append(
                f"Optionaler Abschnitt '{section}' muss vom Typ {expected_type.__name__} sein, "
                f"ist aber {type(config[section]).__name__}"
            )

    # Detailprüfungen innerhalb der Pflichtabschnitte
    data_cfg = config.get("data", {})
    if isinstance(data_cfg, dict):
        if not data_cfg.get("tickers"):
            errors.append("'data.tickers' darf nicht leer sein")
        elif not isinstance(data_cfg["tickers"], list):
            errors.append("'data.tickers' muss eine Liste sein")

    features_cfg = config.get("features", {})
    if isinstance(features_cfg, dict):
        horizons = features_cfg.get("forecast_horizons")
        if horizons is not None:
            if not isinstance(horizons, list) or not horizons:
                errors.append("'features.forecast_horizons' muss eine nicht-leere Liste sein")
            elif any(not isinstance(h, (int, float)) or h <= 0 for h in horizons):
                errors.append("Alle Werte in 'features.forecast_horizons' müssen positiv sein")

        vol_window = features_cfg.get("vol_window")
        if vol_window is not None and (not isinstance(vol_window, (int, float)) or vol_window <= 0):
            errors.append("'features.vol_window' muss eine positive Zahl sein")

        dead_zone = features_cfg.get("dead_zone_kappa")
        if dead_zone is not None and (not isinstance(dead_zone, (int, float)) or dead_zone < 0):
            errors.append("'features.dead_zone_kappa' darf nicht negativ sein")

    splits_cfg = config.get("splits", {})
    if isinstance(splits_cfg, dict):
        for ratio_key in ("train_ratio", "validation_ratio", "test_ratio"):
            val = splits_cfg.get(ratio_key)
            if val is not None and (not isinstance(val, (int, float)) or not 0 < val < 1):
                errors.append(f"'{ratio_key}' muss zwischen 0 und 1 liegen (exklusiv)")

    return errors


def load_config(path: str | Path) -> Dict[str, Any]:
    """Liest eine YAML-Datei ein, validiert die Struktur und gibt das Mapping zurück.

    Parameter
    ---------
    path : str | Path
        Dateipfad zur YAML-Konfiguration (relativ oder absolut).

    Returns
    -------
    Dict[str, Any]
        Vollständiger, validierter Konfigurationsbaum.

    Raises
    ------
    FileNotFoundError
        Falls der Pfad nicht existiert.
    ValueError
        Falls die YAML-Datei kein Mapping an der Wurzel enthält oder
        Pflichtabschnitte fehlen bzw. ungültige Werte enthalten.
    """
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Konfigurationsdatei nicht gefunden: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    if not isinstance(data, dict):
        raise ValueError("Konfigurationsdatei muss ein Mapping an der Wurzelebene definieren")

    errors = _validate_config(data)
    if errors:
        msg = "Konfigurationsvalidierung fehlgeschlagen:\n  - " + "\n  - ".join(errors)
        raise ValueError(msg)

    return data


def ensure_directories(config: Dict[str, Any]) -> None:
    """Erzeugt alle in der Konfiguration referenzierten Basisverzeichnisse.

    Parameter
    ---------
    config : Dict[str, Any]
        Vollständige Projektkonfiguration, aus der die Zielpfade gelesen werden.

    Hinweise
    --------
    - Fehlt ein Block in der Konfiguration, werden sinnvolle Standardwerte verwendet.
    - `mkdir(..., exist_ok=True)` sorgt dafür, dass wiederholte Aufrufe idempotent bleiben.
    """
    cache_dir = Path(config.get("data", {}).get("cache_dir", "data/raw"))
    cache_dir.mkdir(parents=True, exist_ok=True)

    experiment_dir = Path(config.get("experiment", {}).get("output_dir", "artifacts"))
    experiment_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path(config.get("logging", {}).get("log_dir", experiment_dir / "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
