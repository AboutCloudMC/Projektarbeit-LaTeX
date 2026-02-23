"""Datenbeschaffung und -caching für Börsenticker über yfinance."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import pandas as pd
import yfinance as yf

LOGGER = logging.getLogger(__name__)


def _cache_file_name(ticker: str, interval: str, start: str | None, end: str | None) -> str:
    """Erzeugt den Dateinamen für den Cache eines Tickers.

    Parameter
    ---------
    ticker : str
        Kürzel des Wertpapiers (z. B. "AAPL").
    interval : str
        Zeitintervall der historischen Daten (z. B. "1d").
    start : str | None
        Startdatum der Abfrage; `None` steht für "yfinance-Default".
    end : str | None
        Enddatum der Abfrage; `None` steht für das aktuelle Datum.

    Returns
    -------
    str
        Dateiname im Format `<ticker>_<interval>_<start>_<end>.csv`, wobei problematische
        Zeichen ersetzt werden, damit alle Betriebssysteme damit umgehen können.
    """
    safe_start = (start or "start").replace(":", "-")
    safe_end = (end or "latest").replace(":", "-")
    return f"{ticker}_{interval}_{safe_start}_{safe_end}.csv"


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Standardisiert Spaltennamen, Index-Typen und Reihenfolge.

    Parameter
    ---------
    frame : pd.DataFrame
        Rohdaten aus dem yfinance-Download oder einem Cache.

    Returns
    -------
    pd.DataFrame
        Kopie mit einheitlichen Spalten und Datumindex.
    """
    frame = frame.copy()

    if isinstance(frame.columns, pd.MultiIndex):
        # yfinance liefert (Feld, Ticker)-Spaltenpaare, selbst wenn nur ein Symbol abgefragt wird.
        frame.columns = frame.columns.get_level_values(0)

    frame.columns = [str(col).title() for col in frame.columns]
    frame.index = pd.to_datetime(frame.index)
    frame.index.name = "Date"
    frame = frame.sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    return frame


def _load_cached_frame(cache_file: Path) -> pd.DataFrame:
    """Liest eine bereits vorhandene Cache-Datei und normalisiert sie.

    Parameter
    ---------
    cache_file : Path
        Vollständiger Pfad zur CSV-Datei im Cache-Verzeichnis.

    Returns
    -------
    pd.DataFrame
        Eindeutig benannter und sortierter Kursdataframe.
    """
    multi_level_header = False
    try:
        cached = pd.read_csv(cache_file, parse_dates=["Date"], index_col="Date")
    except ValueError as exc:
        # Rückwärtskompatibilität für ältere Caches mit MultiIndex-Headern.
        if "Missing column provided to 'parse_dates'" not in str(exc):
            raise
        cached = pd.read_csv(cache_file, header=[0, 1], index_col=0)
        cached.index.name = "Date"
        multi_level_header = True

    normalized = _normalize_columns(cached)
    if multi_level_header:
        # Speichert den konvertierten Cache für zukünftige Läufe direkt im Simple-Header-Format.
        normalized.to_csv(cache_file)
    return normalized


def fetch_ticker(
    ticker: str,
    start: str | None,
    end: str | None,
    interval: str,
    cache_dir: str | Path,
    adjust: bool = True,
) -> pd.DataFrame:
    """Lädt Kursdaten aus dem Cache oder lädt sie bei Bedarf herunter.

    Parameter
    ---------
    ticker : str
        Börsenticker, der abgefragt werden soll.
    start : str | None
        Optionales Startdatum im ISO-Format.
    end : str | None
        Optionales Enddatum; `None` bedeutet bis heute.
    interval : str
        Zeitauflösung (z. B. "1d", "1h").
    cache_dir : str | Path
        Verzeichnis, in dem CSV-Caches abgelegt werden sollen.
    adjust : bool, default True
        Ob Dividenden und Splits durch `yfinance` adjustiert werden sollen.

    Returns
    -------
    pd.DataFrame
        Normalisierte Kursdaten mit Datumindex.

    Raises
    ------
    ValueError
        Wenn keine Daten vom Remote-Endpunkt zurückgegeben werden.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    cache_file = cache_path / _cache_file_name(ticker, interval, start, end)
    if cache_file.exists():
        LOGGER.info("Loading %s data from cache %s", ticker, cache_file)
        return _load_cached_frame(cache_file)

    LOGGER.info(
        "Downloading %s | interval=%s | start=%s | end=%s",
        ticker,
        interval,
        start,
        end,
    )
    data = yf.download(
        tickers=ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=adjust,
        progress=False,
    )

    if data.empty:
        raise ValueError(f"No market data returned for {ticker}")

    data = _normalize_columns(data)
    data.to_csv(cache_file)
    return data


def load_market_data(config: Dict[str, object]) -> Dict[str, pd.DataFrame]:
    """Lädt alle in der Konfiguration definierten Ticker als DataFrames.

    Parameter
    ---------
    config : Dict[str, object]
        Gesamt-Konfiguration, die einen `data`-Abschnitt mit Tickerliste und
        Downloadparametern enthält.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Mapping von Tickerkürzeln zu ihren Kursreihen.

    Raises
    ------
    RuntimeError
        Wenn für keinen der Ticker Daten geladen werden konnten (z. B. wegen
        Netzwerkausfällen oder falscher Konfiguration).
    """
    data_cfg = config.get("data", {})
    tickers = data_cfg.get("tickers", [])
    start_date = data_cfg.get("start_date")
    end_date = data_cfg.get("end_date")
    interval = data_cfg.get("interval", "1d")
    cache_dir = data_cfg.get("cache_dir", "data/raw")
    adjust = bool(data_cfg.get("adjust", True))

    market_data: Dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        # Jeder Ticker wird isoliert verarbeitet, damit Fehler die restlichen Symbole nicht blockieren.
        try:
            market_data[ticker] = fetch_ticker(
                ticker=ticker,
                start=start_date,
                end=end_date,
                interval=interval,
                cache_dir=cache_dir,
                adjust=adjust,
            )
        except Exception as exc:  # pragma: no cover - logging + continue
            LOGGER.error("Failed to fetch %s: %s", ticker, exc)
            continue

    if not market_data:
        raise RuntimeError("Unable to load any market data; check configuration or network access")

    return market_data
