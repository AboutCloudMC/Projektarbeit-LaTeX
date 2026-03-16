"""Persistierung und Laden von Run-Artefakten (Modelle, Vorhersagen, Schwellenwerte)."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib

LOGGER = logging.getLogger(__name__)


def _artifact_dir(run_dir: Path, ticker: str, horizon: int) -> Path:
    return run_dir / "models" / ticker.upper() / f"h{horizon}"


def save_run_artifacts(
    run_dir: Path,
    ticker: str,
    horizon: int,
    artifacts: dict,
) -> Path:
    """Speichert alle Artefakte eines Ticker/Horizont-Paares im Run-Verzeichnis."""
    out = _artifact_dir(run_dir, ticker, horizon)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "artifacts.joblib"
    joblib.dump(artifacts, path)
    LOGGER.info("Saved run artifacts to %s", path)
    return out


def load_run_artifacts(run_dir: Path, ticker: str, horizon: int) -> dict:
    """Lädt gespeicherte Artefakte für ein Ticker/Horizont-Paar."""
    path = _artifact_dir(run_dir, ticker, horizon) / "artifacts.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"No artifacts found at {path}. "
            f"Run the pipeline first to generate artifacts."
        )
    return joblib.load(path)


def available_horizons(run_dir: Path, ticker: str) -> list[int]:
    """Gibt die verfügbaren Horizonte für einen Ticker im Run-Verzeichnis zurück."""
    ticker_dir = run_dir / "models" / ticker.upper()
    if not ticker_dir.exists():
        return []
    result = []
    for d in sorted(ticker_dir.iterdir()):
        if d.is_dir() and d.name.startswith("h"):
            try:
                result.append(int(d.name[1:]))
            except ValueError:
                continue
    return result


def list_available(run_dir: Path) -> list[tuple[str, int]]:
    """Gibt alle verfügbaren (Ticker, Horizont)-Paare eines Run-Verzeichnisses zurück."""
    models_dir = run_dir / "models"
    if not models_dir.exists():
        return []
    result = []
    for ticker_dir in sorted(models_dir.iterdir()):
        if not ticker_dir.is_dir():
            continue
        for horizon_dir in sorted(ticker_dir.iterdir()):
            if horizon_dir.is_dir() and horizon_dir.name.startswith("h"):
                try:
                    result.append((ticker_dir.name, int(horizon_dir.name[1:])))
                except ValueError:
                    continue
    return result


def latest_run_dir(output_dir: Path = Path("artifacts")) -> Path:
    """Ermittelt das neueste Run-Verzeichnis anhand des Zeitstempels."""
    runs_dir = output_dir / "runs"
    if not runs_dir.exists():
        raise FileNotFoundError(f"No runs directory at {runs_dir}")
    runs = sorted(
        (d for d in runs_dir.iterdir() if d.is_dir()),
        reverse=True,
    )
    if not runs:
        raise FileNotFoundError(f"No runs found in {runs_dir}")
    return runs[0]
