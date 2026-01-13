#!/usr/bin/env python3
"""Analyze a financial instrument by ISIN using Yahoo Finance endpoints."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import statistics
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Iterable, List, Tuple

YAHOO_SEARCH_URL = "https://query1.finance.yahoo.com/v1/finance/search"
YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
TRADING_DAYS = 252
DEFAULT_TIMEOUT = 20
DEFAULT_RETRIES = 5
DEFAULT_BACKOFF = 2.0
DEFAULT_THROTTLE = 1.5
DEFAULT_DEBUG = False
_LAST_REQUEST_AT = 0.0


@dataclass
class RequestConfig:
    retries: int = DEFAULT_RETRIES
    backoff: float = DEFAULT_BACKOFF
    timeout: float = DEFAULT_TIMEOUT
    throttle: float = DEFAULT_THROTTLE
    debug: bool = DEFAULT_DEBUG


@dataclass
class PriceSeries:
    dates: List[dt.date]
    closes: List[float]


@dataclass
class AnalysisResult:
    symbol: str
    name: str
    currency: str
    last_price: float
    last_date: dt.date
    period_start: dt.date
    period_end: dt.date
    total_return: float
    annualized_volatility: float
    max_drawdown: float
    rsi_14: float | None
    sma_50: float | None
    sma_200: float | None


def _throttle_requests(throttle: float) -> None:
    global _LAST_REQUEST_AT
    if throttle <= 0:
        return
    now = time.monotonic()
    elapsed = now - _LAST_REQUEST_AT
    if elapsed < throttle:
        time.sleep(throttle - elapsed)
    _LAST_REQUEST_AT = time.monotonic()


def _fetch_json(url: str, config: RequestConfig) -> dict:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    attempt = 0
    last_error: Exception | None = None
    while attempt <= config.retries:
        try:
            _throttle_requests(config.throttle)
            if config.debug:
                print(f"DEBUG request URL: {url}", file=sys.stderr)
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=config.timeout) as response:
                if config.debug:
                    print(
                        "DEBUG response: "
                        f"status={response.status} content-type={response.headers.get('Content-Type')}",
                        file=sys.stderr,
                    )
                payload = response.read().decode("utf-8")
            return json.loads(payload)
        except urllib.error.HTTPError as exc:
            last_error = exc
            if config.debug:
                print(
                    "DEBUG HTTPError: "
                    f"status={exc.code} retry-after={exc.headers.get('Retry-After')}",
                    file=sys.stderr,
                )
            if exc.code not in {429, 500, 502, 503, 504} or attempt == config.retries:
                raise
            retry_after = exc.headers.get("Retry-After")
            if retry_after:
                try:
                    time.sleep(float(retry_after))
                    attempt += 1
                    continue
                except ValueError:
                    pass
        except urllib.error.URLError as exc:
            last_error = exc
            if attempt == config.retries:
                raise
        sleep_for = config.backoff * (2**attempt)
        time.sleep(sleep_for)
        attempt += 1
    if last_error is not None:
        raise last_error
    raise RuntimeError("Unable to fetch data")


def search_symbol(isin: str, config: RequestConfig) -> Tuple[str, str, str]:
    query = urllib.parse.urlencode({"q": isin, "quotesCount": 10, "newsCount": 0})
    url = f"{YAHOO_SEARCH_URL}?{query}"
    data = _fetch_json(url, config)
    quotes = data.get("quotes", [])
    if not quotes:
        raise ValueError(f"No Yahoo Finance results found for ISIN {isin}")

    best = None
    for quote in quotes:
        if quote.get("quoteType") in {"ETF", "EQUITY", "MUTUALFUND"}:
            best = quote
            break
    if best is None:
        best = quotes[0]

    symbol = best.get("symbol")
    name = best.get("shortname") or best.get("longname") or symbol
    currency = best.get("currency") or ""

    if not symbol:
        raise ValueError(f"Unable to resolve symbol for ISIN {isin}")

    return symbol, name, currency


def fetch_prices(symbol: str, range_: str, interval: str, config: RequestConfig) -> PriceSeries:
    params = urllib.parse.urlencode({"range": range_, "interval": interval})
    url = f"{YAHOO_CHART_URL.format(symbol=symbol)}?{params}"
    data = _fetch_json(url, config)
    result = data.get("chart", {}).get("result", [])
    if not result:
        error = data.get("chart", {}).get("error")
        raise ValueError(f"Yahoo chart data unavailable: {error}")

    series = result[0]
    timestamps = series.get("timestamp") or []
    closes = series.get("indicators", {}).get("quote", [{}])[0].get("close") or []

    dates: List[dt.date] = []
    clean_closes: List[float] = []
    for ts, close in zip(timestamps, closes):
        if close is None:
            continue
        dates.append(dt.datetime.utcfromtimestamp(ts).date())
        clean_closes.append(float(close))

    if len(clean_closes) < 2:
        raise ValueError("Not enough price points to analyze")

    return PriceSeries(dates=dates, closes=clean_closes)


def load_series(path: str) -> PriceSeries:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    dates = [dt.date.fromisoformat(value) for value in payload["dates"]]
    closes = [float(value) for value in payload["closes"]]
    return PriceSeries(dates=dates, closes=closes)


def save_series(path: str, series: PriceSeries) -> None:
    payload = {
        "dates": [value.isoformat() for value in series.dates],
        "closes": series.closes,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _returns(prices: Iterable[float]) -> List[float]:
    prices_list = list(prices)
    return [
        (prices_list[i] / prices_list[i - 1]) - 1
        for i in range(1, len(prices_list))
        if prices_list[i - 1] != 0
    ]


def _max_drawdown(prices: Iterable[float]) -> float:
    peak = -math.inf
    max_dd = 0.0
    for price in prices:
        if price > peak:
            peak = price
        drawdown = (price / peak) - 1 if peak > 0 else 0
        if drawdown < max_dd:
            max_dd = drawdown
    return max_dd


def _simple_moving_average(prices: List[float], window: int) -> float | None:
    if len(prices) < window:
        return None
    slice_ = prices[-window:]
    return sum(slice_) / window


def _rsi(returns: List[float], window: int) -> float | None:
    if len(returns) < window:
        return None
    gains = [r for r in returns[-window:] if r > 0]
    losses = [-r for r in returns[-window:] if r < 0]
    avg_gain = sum(gains) / window if gains else 0.0
    avg_loss = sum(losses) / window if losses else 0.0
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def analyze(symbol: str, name: str, currency: str, series: PriceSeries) -> AnalysisResult:
    returns = _returns(series.closes)
    total_return = (series.closes[-1] / series.closes[0]) - 1
    volatility = statistics.pstdev(returns) * math.sqrt(TRADING_DAYS) if returns else 0.0
    max_drawdown = _max_drawdown(series.closes)
    rsi_14 = _rsi(returns, 14)
    sma_50 = _simple_moving_average(series.closes, 50)
    sma_200 = _simple_moving_average(series.closes, 200)

    return AnalysisResult(
        symbol=symbol,
        name=name,
        currency=currency,
        last_price=series.closes[-1],
        last_date=series.dates[-1],
        period_start=series.dates[0],
        period_end=series.dates[-1],
        total_return=total_return,
        annualized_volatility=volatility,
        max_drawdown=max_drawdown,
        rsi_14=rsi_14,
        sma_50=sma_50,
        sma_200=sma_200,
    )


def _format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def print_report(result: AnalysisResult) -> None:
    print("Analisi ISIN Yahoo Finance")
    print("=" * 32)
    print(f"Strumento: {result.name}")
    print(f"Simbolo:   {result.symbol}")
    if result.currency:
        print(f"Valuta:    {result.currency}")
    print(f"Ultimo prezzo: {result.last_price:.4f} ({result.last_date.isoformat()})")
    print(
        f"Periodo analizzato: {result.period_start.isoformat()} -> {result.period_end.isoformat()}"
    )
    print(f"Rendimento totale: {_format_pct(result.total_return)}")
    print(f"Volatilità annualizzata: {_format_pct(result.annualized_volatility)}")
    print(f"Max drawdown: {_format_pct(result.max_drawdown)}")
    if result.rsi_14 is not None:
        print(f"RSI 14gg: {result.rsi_14:.2f}")
    if result.sma_50 is not None:
        print(f"Media mobile 50gg: {result.sma_50:.4f}")
    if result.sma_200 is not None:
        print(f"Media mobile 200gg: {result.sma_200:.4f}")


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analisi base di un ISIN usando dati Yahoo Finance"
    )
    parser.add_argument("isin", help="Codice ISIN, es. IE00BF2GFH28")
    parser.add_argument(
        "--input-data",
        help="Percorso JSON con dati locali (salvati con --save-data)",
    )
    parser.add_argument(
        "--save-data",
        help="Percorso JSON per salvare i dati scaricati prima dell'analisi",
    )
    parser.add_argument(
        "--range",
        dest="range_",
        default="1y",
        help="Intervallo temporale Yahoo (es. 6mo, 1y, 5y, max)",
    )
    parser.add_argument(
        "--interval",
        default="1d",
        help="Intervallo Yahoo (es. 1d, 1wk, 1mo)",
    )
    parser.add_argument(
        "--symbol",
        help="Simbolo Yahoo da usare al posto della ricerca ISIN",
    )
    parser.add_argument(
        "--check-request",
        action="store_true",
        help="Esegue solo il download dati per verificare la richiesta (senza analisi)",
    )
    parser.add_argument(
        "--no-retry",
        action="store_true",
        help="Disabilita i retry automatici (utile per testare una singola richiesta)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_RETRIES,
        help="Numero massimo di retry per richieste Yahoo",
    )
    parser.add_argument(
        "--backoff",
        type=float,
        default=DEFAULT_BACKOFF,
        help="Fattore di backoff (secondi) tra i retry",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help="Timeout per ogni richiesta HTTP (secondi)",
    )
    parser.add_argument(
        "--throttle",
        type=float,
        default=DEFAULT_THROTTLE,
        help="Attesa minima tra richieste HTTP (secondi)",
    )
    parser.add_argument(
        "--debug-http",
        action="store_true",
        help="Stampa informazioni di debug sulle richieste HTTP",
    )
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    config = RequestConfig(
        retries=args.retries,
        backoff=args.backoff,
        timeout=args.timeout,
        throttle=args.throttle,
        debug=args.debug_http,
    )
    if args.no_retry:
        config.retries = 0
        config.backoff = 0.0

    try:
        if args.input_data:
            series = load_series(args.input_data)
            symbol = args.symbol or "LOCAL_DATA"
            name = symbol
            currency = ""
        else:
            if args.symbol:
                symbol = args.symbol
                name = args.symbol
                currency = ""
            else:
                symbol, name, currency = search_symbol(args.isin, config)
            series = fetch_prices(symbol, args.range_, args.interval, config)
            if args.save_data:
                save_series(args.save_data, series)
            if args.check_request:
                print("Richiesta OK: dati scaricati con successo.")
                print(f"Punti: {len(series.closes)}")
                print(
                    "Periodo: "
                    f"{series.dates[0].isoformat()} -> {series.dates[-1].isoformat()}"
                )
                return 0
        result = analyze(symbol, name, currency, series)
        print_report(result)
        return 0
    except (ValueError, urllib.error.URLError, urllib.error.HTTPError) as exc:
        print(f"Errore: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
