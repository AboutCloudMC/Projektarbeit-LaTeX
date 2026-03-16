"""Logging-Helfer zum gleichzeitigen Schreiben in Datei und Konsole."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path


def setup_logging(log_dir: str | Path, level: str = "INFO") -> Path:
    """Richtet Konsolen- und File-Logging ein und gibt den erzeugten Pfad zurück.

    Parameter
    ---------
    log_dir : str | Path
        Verzeichnis, in dem Log-Dateien pro Lauf abgelegt werden.
    level : str, default "INFO"
        Basis-Log-Level (z. B. "DEBUG", "WARNING").

    Returns
    -------
    Path
        Pfad der erzeugten Log-Datei (inklusive Zeitstempel).
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"run_{timestamp}.log"

    # Entfernt alte Handler, damit bei wiederholten Läufen keine Duplikate entstehen.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
        ],
    )

    logging.getLogger(__name__).info("Logging initialized at %s", log_file)
    return log_file
