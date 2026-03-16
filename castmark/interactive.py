"""Interaktive Eingabehilfen für den --simple Modus."""

from __future__ import annotations

from pathlib import Path


def ask(prompt: str, default: str | None = None) -> str:
    """Fragt nach einer Texteingabe, optionaler Standardwert bei leerem Input."""
    suffix = f" [{default}]" if default is not None else ""
    while True:
        value = input(f"  {prompt}{suffix}: ").strip()
        if value:
            return value
        if default is not None:
            return default
        print("    Bitte einen Wert eingeben.")


def ask_choice(prompt: str, choices: list[str], labels: list[str] | None = None, default_index: int = 0) -> str:
    """Zeigt eine nummerierte Auswahlliste und gibt den gewählten Wert zurück."""
    display = labels if labels is not None else choices
    print(f"\n  {prompt}")
    for i, label in enumerate(display, 1):
        marker = "  *" if i - 1 == default_index else "   "
        print(f"  {marker} [{i}] {label}")
    while True:
        raw = input(f"  Auswahl [Standard: {default_index + 1}]: ").strip()
        if not raw:
            return choices[default_index]
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(choices):
                return choices[idx]
        if raw in choices:
            return raw
        print(f"    Ungültige Eingabe. Bitte eine Zahl zwischen 1 und {len(choices)} eingeben.")


def ask_int(prompt: str, default: int, min_val: int = 1) -> int:
    """Fragt nach einer ganzen Zahl mit Standardwert und optionalem Minimalwert."""
    while True:
        raw = input(f"  {prompt} [{default}]: ").strip()
        if not raw:
            return default
        if raw.isdigit() and int(raw) >= min_val:
            return int(raw)
        print(f"    Bitte eine ganze Zahl >= {min_val} eingeben.")


# ---------------------------------------------------------------------------
# Wiederverwendbare Auswahlblöcke
# ---------------------------------------------------------------------------

def select_run(output_dir: Path = Path("artifacts")) -> Path:
    """Lässt den Nutzer einen vorhandenen Run auswählen."""
    from castmark.artifacts import latest_run_dir
    runs_dir = output_dir / "runs"
    if not runs_dir.exists() or not any(runs_dir.iterdir()):
        raise FileNotFoundError(f"Keine Runs in {runs_dir} gefunden. Zuerst Pipeline ausführen.")
    runs = sorted((d for d in runs_dir.iterdir() if d.is_dir()), reverse=True)
    choice = ask_choice(
        "Run auswählen:",
        choices=[str(r) for r in runs],
        labels=[r.name for r in runs],
        default_index=0,
    )
    return Path(choice)


def select_ticker(run_dir: Path) -> str:
    """Lässt den Nutzer einen Ticker aus dem Run auswählen."""
    from castmark.artifacts import list_available
    pairs = list_available(run_dir)
    if not pairs:
        raise ValueError(f"Keine Artefakte in {run_dir} gefunden.")
    tickers = sorted(set(t for t, _ in pairs))
    return ask_choice("Ticker auswählen:", choices=tickers, default_index=0)


def select_horizon(run_dir: Path, ticker: str) -> int:
    """Lässt den Nutzer einen Horizont für den gewählten Ticker auswählen."""
    from castmark.artifacts import available_horizons
    horizons = available_horizons(run_dir, ticker)
    if not horizons:
        raise ValueError(f"Keine Horizonte für {ticker} in {run_dir} gefunden.")
    choice = ask_choice(
        "Prognosehorizont (Handelstage):",
        choices=[str(h) for h in horizons],
        default_index=0,
    )
    return int(choice)


def select_split() -> str:
    """Lässt den Nutzer einen Datensplit auswählen."""
    return ask_choice(
        "Datensplit:",
        choices=["test", "validation", "train"],
        default_index=0,
    )


def select_config() -> str:
    """Lässt den Nutzer eine Konfigurationsdatei auswählen."""
    configs = sorted(Path("config").glob("*.yaml"))
    if not configs:
        raise FileNotFoundError("Keine YAML-Konfigurationen in config/ gefunden.")
    defaults = [c for c in configs if c.stem == "defaults"]
    default_index = configs.index(defaults[0]) if defaults else 0
    choice = ask_choice(
        "Konfigurationsdatei auswählen:",
        choices=[str(c) for c in configs],
        labels=[c.name for c in configs],
        default_index=default_index,
    )
    return choice
