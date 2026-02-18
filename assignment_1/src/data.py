from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Iterable

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from .utils import clean_ticker, dedupe_preserve_order, is_plausible_ticker


@dataclass(frozen=True)
class MarketDataResult:
    tickers_requested: list[str]
    tickers_ok: list[str]
    tickers_failed: list[str]
    prices: pd.DataFrame  # adj close preferred, columns=tickers_ok
    returns: pd.DataFrame  # log returns, columns=tickers_ok
    weekly_anchors: list[pd.Timestamp]  # last trading day of each week


def _stooq_symbol(ticker: str) -> str | None:
    """
    Best-effort mapping to Stooq symbols for a small curated universe.
    Stooq commonly uses lower-case and suffixes like .us / .de / .fr / .nl.
    """
    t = clean_ticker(ticker)
    mapping = {
        # ETFs
        "SPY": "spy.us",
        "QQQ": "qqq.us",
        "IWM": "iwm.us",
        "IEF": "ief.us",
        "HYG": "hyg.us",
        "GLD": "gld.us",
        "VGK": "vgk.us",
        # Stocks
        "AAPL": "aapl.us",
        "MSFT": "msft.us",
        "TSLA": "tsla.us",
        "SAP.DE": "sap.de",
        "SIE.DE": "sie.de",
        # Some EU guesses (may vary on Stooq)
        "ASML.AS": "asml.nl",
        "AIR.PA": "air.fr",
    }
    if t in mapping:
        return mapping[t]
    # Heuristic fallback: if already has a market suffix, try lower-case as-is.
    return t.lower()


def _extract_close_from_yf_history(hist: pd.DataFrame) -> pd.DataFrame:
    """
    yf.Tickers(...).history(...) returns a DataFrame that is often MultiIndex on columns.
    Prefer 'Adj Close' when present, else 'Close'. Output columns are tickers.
    """
    if hist is None or hist.empty:
        return pd.DataFrame()

    if isinstance(hist.columns, pd.MultiIndex):
        lvl0 = set(map(str, hist.columns.get_level_values(0)))
        lvl1 = set(map(str, hist.columns.get_level_values(1)))
        field = None
        for f in ("Adj Close", "Close"):
            if f in lvl0 or f in lvl1:
                field = f
                break
        if field is None:
            return pd.DataFrame(index=hist.index)

        # Either (field, ticker) or (ticker, field)
        if field in lvl0:
            close = hist[field].copy()
        else:
            close = hist.xs(field, level=1, axis=1).copy()

        close.columns = [clean_ticker(str(c)) for c in close.columns]
        return close

    cols = set(map(str, hist.columns))
    field = "Adj Close" if "Adj Close" in cols else ("Close" if "Close" in cols else None)
    if field is None:
        return pd.DataFrame(index=hist.index)
    close = hist[[field]].copy()
    close.columns = ["SINGLE"]
    return close


@st.cache_data(show_spinner=False, ttl=6 * 60 * 60)
def download_prices_yahoo(
    tickers: tuple[str, ...],
    *,
    period: str,
    interval: str,
    refresh_nonce: int = 0,
) -> pd.DataFrame:
    """
    Market data via yfinance as an unofficial wrapper around Yahoo Finance:
    - create a multi-ticker object: yf.Tickers(...)
    - call .history(...) for OHLCV
    """
    """
    Minimal-request mode (helps with rate limits / flaky networks):
    - fetch tickers one-by-one
    - threads=False
    - progress=False
    """
    if not tickers:
        return pd.DataFrame()

    _ = refresh_nonce  # part of cache key; increment to force refresh
    frames = []
    for t in tickers:
        try:
            df = yf.download(
                t,
                period=period,
                interval=interval,
                threads=False,
                progress=False,
                auto_adjust=False,
                timeout=20,
                group_by="column",
            )
            if df is None or df.empty:
                continue
            # Depending on yfinance settings/version, this may already be MultiIndex (field, ticker).
            # Only wrap into MultiIndex if it's currently flat columns.
            if not isinstance(df.columns, pd.MultiIndex):
                df.columns = pd.MultiIndex.from_product([df.columns, [clean_ticker(t)]])
            frames.append(df)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1).sort_index()


@st.cache_data(show_spinner=False, ttl=60 * 60)
def download_prices_stooq(
    tickers: tuple[str, ...],
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    Live fallback provider (Stooq) when Yahoo is blocked.
    Uses daily close (not adjusted). Returned columns are original tickers.
    """
    import io
    import requests

    start_dt = pd.to_datetime(start).tz_localize(None)
    end_dt = pd.to_datetime(end).tz_localize(None)

    series: dict[str, pd.Series] = {}
    for t in tickers:
        sym = _stooq_symbol(t)
        if not sym:
            continue
        try:
            url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
            r = requests.get(url, timeout=20)
            if r.status_code != 200 or not r.text or "Date,Open,High,Low,Close,Volume" not in r.text[:80]:
                continue
            df = pd.read_csv(io.StringIO(r.text), parse_dates=["Date"])
            if df.empty or "Close" not in df.columns:
                continue
            df = df.rename(columns={"Date": "date"}).set_index("date").sort_index()
            df = df.loc[(df.index >= start_dt) & (df.index <= end_dt)]
            if df.empty:
                continue
            series[clean_ticker(t)] = df["Close"].astype(float)
        except Exception:
            continue
    if not series:
        return pd.DataFrame()
    out = pd.concat(series, axis=1)
    out.index = pd.to_datetime(out.index)
    return out


def generate_sample_market_data(
    tickers: Iterable[str],
    *,
    months: int = 12,
    seed: int = 42,
) -> MarketDataResult:
    """
    Offline fallback to keep the prototype usable when Yahoo Finance blocks requests.
    This generates *synthetic* prices with realistic-ish volatility and cross-correlation.
    The UI must clearly label this as sample data (not live).
    """
    tickers_clean = [clean_ticker(t) for t in tickers if t and is_plausible_ticker(t)]
    tickers_clean = dedupe_preserve_order(tickers_clean)

    end_dt = date.today()
    start_dt = end_dt - timedelta(days=int(months * 30.5))
    idx = pd.bdate_range(start_dt.isoformat(), (end_dt + timedelta(days=1)).isoformat(), inclusive="left")
    if len(idx) < 60 or not tickers_clean:
        return MarketDataResult(
            tickers_requested=tickers_clean,
            tickers_ok=[],
            tickers_failed=tickers_clean,
            prices=pd.DataFrame(index=idx),
            returns=pd.DataFrame(index=idx),
            weekly_anchors=[],
        )

    rng = np.random.default_rng(seed)
    n = len(tickers_clean)

    # 1 market factor + idiosyncratic noise creates correlated returns.
    market = rng.normal(0.0, 1.0, size=len(idx))
    betas = rng.uniform(0.3, 0.9, size=n)
    idio = rng.normal(0.0, 1.0, size=(len(idx), n))

    # Target annualized vols ~ 8% to 45%
    ann_vol = rng.uniform(0.08, 0.45, size=n)
    daily_vol = ann_vol / np.sqrt(252.0)

    # Build standardized correlated returns then scale by vol
    r = (market[:, None] * betas[None, :] + idio * (1.0 - betas[None, :]))
    r = (r - r.mean(axis=0, keepdims=True)) / (r.std(axis=0, keepdims=True) + 1e-12)
    r = r * daily_vol[None, :]

    # Convert to prices starting around 50-250
    start_prices = rng.uniform(50.0, 250.0, size=n)
    log_px = np.log(start_prices)[None, :] + np.cumsum(r, axis=0)
    px = np.exp(log_px)

    prices = pd.DataFrame(px, index=idx, columns=tickers_clean)
    returns = np.log(prices).diff()
    anchors = _build_weekly_anchors(prices.index)

    return MarketDataResult(
        tickers_requested=tickers_clean,
        tickers_ok=tickers_clean,
        tickers_failed=[],
        prices=prices,
        returns=returns,
        weekly_anchors=anchors,
    )


def _build_weekly_anchors(trading_index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    if len(trading_index) == 0:
        return []
    s = pd.Series(trading_index, index=trading_index)
    # Group by "week ending Friday" and take last available trading day in that week.
    anchors = s.groupby(s.index.to_period("W-FRI")).max().sort_values().tolist()
    return [pd.Timestamp(x) for x in anchors]


def _clean_and_align_prices(
    px: pd.DataFrame,
    *,
    max_missing_frac: float = 0.25,
    ffill_limit: int = 2,
) -> pd.DataFrame:
    if px.empty:
        return px

    px = px.sort_index()
    px = px.replace([np.inf, -np.inf], np.nan)

    # Minimal forward fill to handle occasional holidays / sparse gaps.
    px = px.ffill(limit=ffill_limit)

    # Drop days where too many assets are missing (bad alignment for correlations).
    missing_frac = px.isna().mean(axis=1)
    px = px.loc[missing_frac <= max_missing_frac]

    # For computations that require pairwise alignment, we keep remaining NaNs;
    # downstream code will drop per-window per-metric.
    return px


def load_market_data(
    tickers: Iterable[str],
    *,
    months: int = 12,
    provider: str = "stooq",
    yahoo_refresh_nonce: int = 0,
) -> MarketDataResult:
    tickers_clean = [clean_ticker(t) for t in tickers if t and is_plausible_ticker(t)]
    tickers_clean = dedupe_preserve_order(tickers_clean)

    end_dt = date.today()
    start_dt = end_dt - timedelta(days=int(months * 30.5))
    start = start_dt.isoformat()
    end = (end_dt + timedelta(days=1)).isoformat()

    provider = (provider or "stooq").strip().lower()
    close = pd.DataFrame()

    if provider == "stooq":
        close = download_prices_stooq(tuple(tickers_clean), start=start, end=end)
    elif provider == "yahoo":
        period = "2y" if int(months) >= 24 else "1y"
        raw = download_prices_yahoo(tuple(tickers_clean), period=period, interval="1d", refresh_nonce=int(yahoo_refresh_nonce))
        close = _extract_close_from_yf_history(raw)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    tickers_failed: list[str] = []
    tickers_ok: list[str] = []

    if close.empty:
        tickers_failed = tickers_clean[:]
        prices = pd.DataFrame()
    else:
        # Handle the single ticker case
        if close.columns.tolist() == ["SINGLE"] and len(tickers_clean) == 1:
            close.columns = [tickers_clean[0]]

        # Determine failed tickers: all-NaN series
        for t in tickers_clean:
            if t not in close.columns:
                tickers_failed.append(t)
            elif close[t].dropna().empty:
                tickers_failed.append(t)
            else:
                tickers_ok.append(t)

        prices = close[tickers_ok].copy() if tickers_ok else pd.DataFrame(index=close.index)

    prices = _clean_and_align_prices(prices)

    if prices.empty or not tickers_ok:
        returns = pd.DataFrame(index=prices.index)
        anchors = []
    else:
        # Log returns; keep NaNs where price missing.
        log_px = np.log(prices)
        returns = log_px.diff()
        anchors = _build_weekly_anchors(prices.index)

    return MarketDataResult(
        tickers_requested=tickers_clean,
        tickers_ok=tickers_ok,
        tickers_failed=tickers_failed,
        prices=prices,
        returns=returns,
        weekly_anchors=anchors,
    )

