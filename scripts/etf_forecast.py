#!/usr/bin/env python3
"""Simple ETF direction forecast using a lightweight ML model."""

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
from typing import List, Sequence, Tuple

YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"


def _fetch_json(url: str, retries: int = 5, backoff: float = 2.0) -> dict:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    attempt = 0
    last_error: Exception | None = None
    while attempt <= retries:
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=15) as response:
                payload = response.read().decode("utf-8")
            return json.loads(payload)
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code not in {429, 500, 502, 503, 504} or attempt == retries:
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
            if attempt == retries:
                raise
        sleep_for = backoff * (2**attempt)
        time.sleep(sleep_for)
        attempt += 1
    if last_error is not None:
        raise last_error
    raise RuntimeError("Unable to fetch data")


def fetch_prices(symbol: str, range_: str, interval: str) -> Tuple[List[dt.date], List[float]]:
    params = urllib.parse.urlencode({"range": range_, "interval": interval})
    url = f"{YAHOO_CHART_URL.format(symbol=symbol)}?{params}"
    data = _fetch_json(url)
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

    if len(clean_closes) < 50:
        raise ValueError("Not enough data points to build features")

    return dates, clean_closes


def _returns(prices: Sequence[float]) -> List[float]:
    return [
        (prices[i] / prices[i - 1]) - 1
        for i in range(1, len(prices))
        if prices[i - 1] != 0
    ]


def _sma(prices: Sequence[float], window: int) -> float:
    return sum(prices[-window:]) / window


def _rsi(returns: Sequence[float], window: int) -> float:
    gains = [r for r in returns[-window:] if r > 0]
    losses = [-r for r in returns[-window:] if r < 0]
    avg_gain = sum(gains) / window if gains else 0.0
    avg_loss = sum(losses) / window if losses else 0.0
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def build_dataset(prices: Sequence[float]) -> Tuple[List[List[float]], List[int]]:
    returns = _returns(prices)
    features: List[List[float]] = []
    targets: List[int] = []

    min_index = 30
    for i in range(min_index, len(prices) - 1):
        window_prices = prices[: i + 1]
        window_returns = returns[:i]
        if len(window_returns) < 20:
            continue
        sma_10 = _sma(window_prices, 10)
        sma_20 = _sma(window_prices, 20)
        vol_10 = statistics.pstdev(window_returns[-10:]) if len(window_returns) >= 10 else 0.0
        rsi_14 = _rsi(window_returns, 14)
        momentum_5 = (window_prices[-1] / window_prices[-6]) - 1 if i >= 5 else 0.0
        features.append(
            [
                window_returns[-1],
                (window_prices[-1] / sma_10) - 1,
                (window_prices[-1] / sma_20) - 1,
                vol_10,
                rsi_14 / 100.0,
                momentum_5,
            ]
        )
        next_return = (prices[i + 1] / prices[i]) - 1
        targets.append(1 if next_return > 0 else 0)
    return features, targets


def _standardize(features: List[List[float]]) -> Tuple[List[List[float]], List[float], List[float]]:
    transposed = list(zip(*features))
    means = [statistics.mean(col) for col in transposed]
    stds = [statistics.pstdev(col) or 1.0 for col in transposed]

    scaled = []
    for row in features:
        scaled.append([(x - m) / s for x, m, s in zip(row, means, stds)])
    return scaled, means, stds


def _sigmoid(z: float) -> float:
    return 1 / (1 + math.exp(-z))


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def train_logistic_regression(
    features: List[List[float]],
    targets: List[int],
    epochs: int = 500,
    lr: float = 0.1,
) -> List[float]:
    weights = [0.0] * (len(features[0]) + 1)
    for _ in range(epochs):
        grad = [0.0] * len(weights)
        for row, target in zip(features, targets):
            row_with_bias = [1.0] + row
            pred = _sigmoid(_dot(weights, row_with_bias))
            error = pred - target
            for i, value in enumerate(row_with_bias):
                grad[i] += error * value
        for i in range(len(weights)):
            weights[i] -= lr * (grad[i] / len(features))
    return weights


def predict_probability(weights: Sequence[float], row: Sequence[float]) -> float:
    return _sigmoid(_dot(weights, [1.0] + list(row)))


def accuracy(weights: Sequence[float], features: List[List[float]], targets: List[int]) -> float:
    correct = 0
    for row, target in zip(features, targets):
        pred = 1 if predict_probability(weights, row) >= 0.5 else 0
        if pred == target:
            correct += 1
    return correct / len(targets) if targets else 0.0


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Previsione direzionale base per ETF con modello ML semplice"
    )
    parser.add_argument("symbol", help="Simbolo Yahoo Finance (es. VWCE.MI)")
    parser.add_argument("--range", dest="range_", default="5y")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.1)
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    dates, prices = fetch_prices(args.symbol, args.range_, args.interval)
    features, targets = build_dataset(prices)
    if not features:
        raise ValueError("Feature set empty; try a longer range")

    features_scaled, means, stds = _standardize(features)
    split_index = int(len(features_scaled) * args.train_split)
    train_x = features_scaled[:split_index]
    train_y = targets[:split_index]
    test_x = features_scaled[split_index:]
    test_y = targets[split_index:]

    weights = train_logistic_regression(train_x, train_y, epochs=args.epochs, lr=args.lr)
    test_accuracy = accuracy(weights, test_x, test_y)

    last_row = features_scaled[-1]
    prob_up = predict_probability(weights, last_row)
    last_date = dates[-1]

    print("Previsione ETF (modello ML semplice)")
    print("=" * 40)
    print(f"Simbolo: {args.symbol}")
    print(f"Periodo: {dates[0].isoformat()} -> {dates[-1].isoformat()}")
    print(f"Dati train: {len(train_x)} | test: {len(test_x)}")
    print(f"Accuratezza test: {test_accuracy * 100:.2f}%")
    print(f"Ultima data disponibile: {last_date.isoformat()}")
    print(f"Probabilità rialzo prossimo giorno: {prob_up * 100:.2f}%")
    print("Nota: modello didattico, non è consulenza finanziaria.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
