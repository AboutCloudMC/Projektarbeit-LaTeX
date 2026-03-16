"""Einstiegspunkt für die Kommandozeile – nimmt einen Konfigurationspfad entgegen und startet die Pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from castmark import pipeline

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parst alle Kommandozeilenargumente für die CLI.

    Returns
    -------
    argparse.Namespace
        Enthält `config` (Pfad zur YAML-Datei) und `head` (Anzahl anzuzeigende Zeilen).
    """
    parser = argparse.ArgumentParser(description="CastMark forecasting CLI")
    parser.add_argument(
        "--config",
        default="config/defaults.yaml",
        help="Path to the YAML configuration file",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=5,
        help="Show the first N rows of the resulting metrics table",
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Interaktiver Eingabemodus: Parameter werden schrittweise abgefragt",
    )
    return parser.parse_args()


def _simple_mode(args: argparse.Namespace) -> None:
    """Füllt args interaktiv mit Nutzereingaben."""
    from castmark.interactive import ask_int, select_config
    print("\n=== CastMark – Interaktiver Modus ===")
    args.config = select_config()
    args.head = ask_int("Wie viele Ergebniszeilen anzeigen?", default=args.head, min_val=1)
    print()


def main() -> None:
    """Startet den kompletten Pipeline-Lauf und zeigt einen kurzen Ergebnis-Ausschnitt.

    Der Funktionsfluss ist:

    1. CLI-Argumente einsammeln (`parse_args`).
    2. Pipeline mit der gewünschten Konfiguration ausführen.
    3. Die ersten `head` Zeilen der Kennzahlen loggen, sodass eine schnelle Prüfung möglich ist.
    """
    args = parse_args()
    if args.simple:
        _simple_mode(args)
    metrics = pipeline.run(args.config)
    # Ein DataFrame-Slice ist für CLI-Ausgaben praktikabler als die komplette Tabelle.
    head = metrics.head(args.head)
    LOGGER.info("Metrics preview:\n%s", head.to_string(index=False))


if __name__ == "__main__":
    main()
