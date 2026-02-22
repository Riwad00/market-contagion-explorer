"""
Contagion Explorer - AI-Powered Market Correlation Analysis
A polished, modern Streamlit dashboard with AI as the centerpiece.
"""

from __future__ import annotations

import io
import os
import json
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from typing import Any
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import requests

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Contagion Explorer | AI-Powered Market Analysis",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CUSTOM CSS - MODERN DARK THEME (all text forced light for readability)
# =============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    }

    /* ---- GLOBAL: force ALL text light on dark bg ---- */
    .stApp, .stApp p, .stApp span, .stApp li, .stApp div,
    .stApp label, .stApp td, .stApp th,
    .stMarkdown, .stMarkdown p, .stMarkdown span, .stMarkdown li, .stMarkdown div {
        color: #e2e8f0;
    }
    h1, h2, h3, h4, h5, h6 { color: #f8fafc !important; }
    .stCaption, .stCaption p { color: #94a3b8 !important; }

    /* Sidebar - Glassmorphism */
    section[data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.95) !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(148, 163, 184, 0.1);
    }
    section[data-testid="stSidebar"] .stMarkdown { color: #94a3b8; }

    /* Main content area */
    .main > div {
        padding: 2rem 3rem;
        max-width: 1400px;
        margin: 0 auto;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(59,130,246,0.12) 0%, rgba(139,92,246,0.12) 100%);
        border: 1px solid rgba(148,163,184,0.18);
        border-radius: 16px;
        padding: 1.25rem;
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(59,130,246,0.15);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #60a5fa, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #94a3b8 !important;
        margin-top: 0.35rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Headers */
    h1 { font-weight: 700 !important; font-size: 2.25rem !important; margin-bottom: 0.25rem !important; }
    h2 { font-weight: 600 !important; font-size: 1.6rem !important; margin-top: 1.5rem !important; margin-bottom: 0.75rem !important; }
    h3 { font-weight: 600 !important; font-size: 1.15rem !important; }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.3s !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(59,130,246,0.4) !important;
    }
    .stButton > button[kind="secondary"] {
        background: rgba(148,163,184,0.15) !important;
        color: #cbd5e1 !important;
    }

    /* Select boxes & inputs */
    div[data-testid="stSelectbox"] > div > div {
        background: rgba(30,41,59,0.8) !important;
        border: 1px solid rgba(148,163,184,0.2) !important;
        border-radius: 12px !important;
        color: #e2e8f0 !important;
    }
    .stTextInput > div > div > input {
        background: rgba(30,41,59,0.8) !important;
        border: 1px solid rgba(148,163,184,0.2) !important;
        border-radius: 12px !important;
        color: #e2e8f0 !important;
    }

    /* Sliders */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #3b82f6, #8b5cf6) !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(30,41,59,0.6);
        border-radius: 12px;
        padding: 0.5rem;
        gap: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: #94a3b8 !important;
        border-radius: 8px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 500 !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
        color: white !important;
    }

    /* Dataframes */
    .stDataFrame {
        background: rgba(30,41,59,0.6) !important;
        border-radius: 12px;
        overflow: hidden;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(30,41,59,0.6) !important;
        border-radius: 12px !important;
        color: #cbd5e1 !important;
        font-weight: 500 !important;
    }

    /* Success / Info / Warning / Error boxes */
    .stAlert > div { color: #e2e8f0 !important; }
    div[data-testid="stAlert"] p { color: #e2e8f0 !important; }

    /* Radio buttons */
    .stRadio > div {
        background: rgba(30,41,59,0.6);
        border-radius: 12px;
        padding: 0.5rem;
    }
    .stRadio label span { color: #e2e8f0 !important; }

    /* ---- FIX: prevent scrollable plotly charts ---- */
    [data-testid="stPlotlyChart"] { overflow: hidden !important; }
    [data-testid="stPlotlyChart"] > div { overflow: hidden !important; }
    iframe[title="streamlit_plotly_events"] { overflow: hidden !important; }

    /* Plotly chart background */
    .js-plotly-plot {
        background: rgba(30,41,59,0.4) !important;
        border-radius: 16px !important;
        padding: 0.5rem !important;
    }

    /* AI feature highlight card */
    .ai-card {
        background: linear-gradient(135deg, rgba(139,92,246,0.15) 0%, rgba(59,130,246,0.15) 100%);
        border: 1px solid rgba(139,92,246,0.3);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .finding-card {
        background: rgba(30,41,59,0.5);
        border-left: 3px solid #60a5fa;
        border-radius: 0 12px 12px 0;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
    }
    .caveat-card {
        background: rgba(245,158,11,0.08);
        border-left: 3px solid #fbbf24;
        border-radius: 0 12px 12px 0;
        padding: 0.85rem 1.25rem;
        margin-bottom: 0.5rem;
    }
    .implication-card {
        background: rgba(74,222,128,0.08);
        border-left: 3px solid #4ade80;
        border-radius: 0 12px 12px 0;
        padding: 0.85rem 1.25rem;
        margin-bottom: 0.5rem;
    }
    .news-card {
        background: rgba(30,41,59,0.45);
        border: 1px solid rgba(148,163,184,0.12);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.5rem;
    }

    /* Divider */
    hr { border-color: rgba(148,163,184,0.15) !important; }

    /* Positive / Negative / Neutral helpers */
    .clr-pos { color: #4ade80 !important; }
    .clr-neg { color: #f87171 !important; }
    .clr-warn { color: #fbbf24 !important; }
    .clr-blue { color: #60a5fa !important; }
    .clr-purple { color: #a78bfa !important; }
    .clr-muted { color: #94a3b8 !important; }
    .clr-white { color: #f8fafc !important; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CONSTANTS & CONFIG
# =============================================================================
ALL_AVAILABLE_TICKERS = {
    "NVDA":   "NVIDIA",
    "AAPL":   "Apple",
    "GOOGL":  "Alphabet (Google)",
    "MSFT":   "Microsoft",
    "AMZN":   "Amazon",
    "META":   "Meta",
    "AVGO":   "Broadcom",
    "TSLA":   "Tesla",
    "BRK-B":  "Berkshire Hathaway",
    "SPY":    "S&P 500 ETF",
    "QQQ":    "Nasdaq ETF",
    "IEF":    "7-10yr Treasury ETF",
    "GLD":    "Gold ETF",
    "HYG":    "High Yield Bond ETF",
    "IWM":    "Russell 2000 ETF",
}
DEFAULT_SELECTED = ["NVDA", "AAPL", "GOOGL", "MSFT", "AMZN", "META", "TSLA", "SPY", "QQQ", "GLD"]
DEFAULT_UNIVERSE = DEFAULT_SELECTED  # kept for backward compat

# =============================================================================
# SESSION STATE
# =============================================================================
def init_session_state():
    defaults = {
        "custom_tickers": [],
        "selected_tickers": DEFAULT_SELECTED[:],
        "global_lookback": 60,
        "global_threshold": 0.60,
        "llm_provider": "OpenAI",
        "llm_model": "gpt-4.1-mini",
        "llm_api_key": os.getenv("OPENAI_API_KEY", ""),
        "llm_custom_base_url": "",
        "llm_last_key_check": None,
        "use_sample_data": False,
        "live_provider": "yahoo",
        "yahoo_refresh_nonce": 0,
        "last_pair_evidence_json": None,
        "last_ai_brief": None,
        "last_news_analysis": None,
        "last_ai_run_meta": None,
        "ai_analysis_depth": "Fast",
        "news_api_key": os.getenv("NEWS_API_KEY", ""),
        "current_page": "home",
        "analysis_history": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# =============================================================================
# DATA MODULES
# =============================================================================
import yfinance as yf
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

@dataclass
class MarketDataResult:
    tickers_requested: list
    tickers_ok: list
    tickers_failed: list
    prices: pd.DataFrame
    returns: pd.DataFrame
    weekly_anchors: list

def clean_ticker(t: str) -> str:
    return t.strip().upper()

def is_plausible_ticker(t: str) -> bool:
    import re
    return bool(re.match(r"^[A-Z0-9][A-Z0-9\.\-]{0,14}$", clean_ticker(t)))

def dedupe_preserve_order(items):
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

def _build_weekly_anchors(trading_index):
    if len(trading_index) == 0:
        return []
    s = pd.Series(trading_index, index=trading_index)
    anchors = s.groupby(s.index.to_period("W-FRI")).max().sort_values().tolist()
    return [pd.Timestamp(x) for x in anchors]

# ---- Yahoo ----
@st.cache_data(show_spinner=False, ttl=300)
def download_prices_yahoo(tickers, period="1y", refresh_nonce=0):
    if not tickers:
        return pd.DataFrame()
    frames = []
    for t in tickers:
        try:
            df = yf.download(t, period=period, interval="1d", threads=False, progress=False, auto_adjust=True, timeout=20)
            if df is not None and not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df = df['Close']
                else:
                    df.columns = [clean_ticker(t)]
                frames.append(df)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1).sort_index()

# ---- Stooq ----
_STOOQ_MAP = {
    "SPY": "spy.us", "QQQ": "qqq.us", "IWM": "iwm.us", "IEF": "ief.us",
    "HYG": "hyg.us", "GLD": "gld.us", "VGK": "vgk.us",
    "AAPL": "aapl.us", "MSFT": "msft.us", "TSLA": "tsla.us",
    "SAP.DE": "sap.de", "SIE.DE": "sie.de", "ASML.AS": "asml.nl", "AIR.PA": "air.fr",
    "NVDA": "nvda.us", "AMD": "amd.us", "TSM": "tsm.us", "AMZN": "amzn.us",
    "GOOGL": "googl.us", "META": "meta.us",
}

def _stooq_symbol(ticker: str) -> str:
    t = clean_ticker(ticker)
    if t in _STOOQ_MAP:
        return _STOOQ_MAP[t]
    return t.lower()

@st.cache_data(show_spinner=False, ttl=3600)
def download_prices_stooq(tickers, start, end):
    """Fallback provider — Stooq daily closes (no API key needed)."""
    start_dt = pd.to_datetime(start).tz_localize(None)
    end_dt = pd.to_datetime(end).tz_localize(None)
    series: dict[str, pd.Series] = {}
    for t in tickers:
        sym = _stooq_symbol(t)
        try:
            url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
            r = requests.get(url, timeout=20)
            if r.status_code != 200 or not r.text or "Date" not in r.text[:60]:
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

# ---- Sample data ----
def generate_sample_market_data(tickers, months=12, seed=42):
    tickers_clean = [clean_ticker(t) for t in tickers if t and is_plausible_ticker(t)]
    tickers_clean = dedupe_preserve_order(tickers_clean)
    end_dt = date.today()
    start_dt = end_dt - timedelta(days=int(months * 30.5))
    idx = pd.bdate_range(start_dt.isoformat(), (end_dt + timedelta(days=1)).isoformat(), inclusive="left")
    if len(idx) < 60 or not tickers_clean:
        return MarketDataResult(tickers_clean, [], [], pd.DataFrame(), pd.DataFrame(), [])
    rng = np.random.default_rng(seed)
    n = len(tickers_clean)
    market = rng.normal(0.0, 1.0, size=len(idx))
    betas = rng.uniform(0.3, 0.9, size=n)
    idio = rng.normal(0.0, 1.0, size=(len(idx), n))
    ann_vol = rng.uniform(0.08, 0.45, size=n)
    daily_vol = ann_vol / np.sqrt(252.0)
    r = (market[:, None] * betas[None, :] + idio * (1.0 - betas[None, :]))
    r = (r - r.mean(axis=0, keepdims=True)) / (r.std(axis=0, keepdims=True) + 1e-12)
    r = r * daily_vol[None, :]
    start_prices = rng.uniform(50.0, 250.0, size=n)
    log_px = np.log(start_prices)[None, :] + np.cumsum(r, axis=0)
    px = np.exp(log_px)
    prices = pd.DataFrame(px, index=idx, columns=tickers_clean)
    returns = np.log(prices).diff()
    anchors = _build_weekly_anchors(prices.index)
    return MarketDataResult(tickers_clean, tickers_clean, [], prices, returns, anchors)

# ---- Unified loader ----
@st.cache_data(show_spinner=False, ttl=300)
def load_market_data(tickers, months=12, provider="yahoo", yahoo_refresh_nonce=0):
    tickers_clean = [clean_ticker(t) for t in tickers if t and is_plausible_ticker(t)]
    tickers_clean = dedupe_preserve_order(tickers_clean)
    period = "2y" if months >= 24 else "1y"
    end_dt = date.today()
    start_dt = end_dt - timedelta(days=int(months * 30.5))

    prices = pd.DataFrame()
    provider = (provider or "yahoo").strip().lower()

    if provider == "stooq":
        prices = download_prices_stooq(
            tuple(tickers_clean),
            start=start_dt.isoformat(),
            end=(end_dt + timedelta(days=1)).isoformat(),
        )
    else:
        # yahoo (default)
        try:
            prices = download_prices_yahoo(tuple(tickers_clean), period=period, refresh_nonce=yahoo_refresh_nonce)
        except Exception:
            prices = pd.DataFrame()

        # Auto-fallback to stooq if yahoo returned nothing
        if prices.empty:
            prices = download_prices_stooq(
                tuple(tickers_clean),
                start=start_dt.isoformat(),
                end=(end_dt + timedelta(days=1)).isoformat(),
            )

    tickers_failed = []
    tickers_ok = []
    if prices.empty:
        tickers_failed = tickers_clean[:]
    else:
        for t in tickers_clean:
            if t not in prices.columns or prices[t].dropna().empty:
                tickers_failed.append(t)
            else:
                tickers_ok.append(t)
        prices = prices[tickers_ok] if tickers_ok else pd.DataFrame()

    if prices.empty or not tickers_ok:
        returns = pd.DataFrame()
        anchors = []
    else:
        log_px = np.log(prices)
        returns = log_px.diff()
        anchors = _build_weekly_anchors(prices.index)

    return MarketDataResult(tickers_clean, tickers_ok, tickers_failed, prices, returns, anchors)

# =============================================================================
# METRICS FUNCTIONS
# =============================================================================
def compute_window_stats(returns, end_date, lookback):
    w = returns.loc[:end_date].tail(int(lookback))
    corr = w.corr(min_periods=max(10, int(0.5 * lookback)))
    vol = w.std(skipna=True) * np.sqrt(252.0)
    return type('obj', (object,), {'end_date': end_date, 'lookback': lookback, 'returns_window': w, 'corr': corr, 'vol_annualized': vol})()

def pair_corr_at_anchor(returns, asset_a, asset_b, end_date, lookback):
    clean = returns[[asset_a, asset_b]].loc[:end_date].dropna()
    w = clean.tail(int(lookback))
    if len(w) < max(10, int(0.5 * lookback)):
        return None
    return float(w[asset_a].corr(w[asset_b]))

def lagged_correlations(returns_window, asset_a, asset_b, max_lag=5):
    df = returns_window[[asset_a, asset_b]].dropna().copy()
    out = {"a_leads_b": [], "b_leads_a": []}
    if len(df) < 15:
        return out
    a, b = df[asset_a], df[asset_b]
    for k in range(1, max_lag + 1):
        corr_ab = a.corr(b.shift(-k))
        corr_ba = b.corr(a.shift(-k))
        out["a_leads_b"].append({"lag_days": k, "corr": None if pd.isna(corr_ab) else float(corr_ab)})
        out["b_leads_a"].append({"lag_days": k, "corr": None if pd.isna(corr_ba) else float(corr_ba)})
    return out

def extreme_move_overlap(returns_window, asset_a, asset_b, top_n=10):
    df = returns_window[[asset_a, asset_b]].dropna().copy()
    if df.empty:
        return {"top_n": top_n, "same_direction": 0, "opposite_direction": 0, "overlap_days": 0}
    a, b = df[asset_a], df[asset_b]
    a_top = a.abs().nlargest(min(top_n, len(a))).index
    b_top = b.abs().nlargest(min(top_n, len(b))).index
    overlap = a_top.intersection(b_top)
    same = sum(1 for d in overlap if np.sign(a.loc[d]) == np.sign(b.loc[d]) and np.sign(a.loc[d]) != 0)
    opp = len(overlap) - same
    return {"top_n": top_n, "overlap_days": int(len(overlap)), "same_direction": same, "opposite_direction": opp}

def bootstrap_corr_ci(returns_window, asset_a, asset_b, n_boot=400, seed=7):
    df = returns_window[[asset_a, asset_b]].dropna().copy()
    n = len(df)
    if n < 30:
        return {"n": int(n), "n_boot": int(n_boot), "ci_95": None, "note": "sample_too_small"}
    rng = np.random.default_rng(seed)
    a, b = df[asset_a].to_numpy(), df[asset_b].to_numpy()
    cors = np.array([np.corrcoef(a[rng.integers(0, n, size=n)], b[rng.integers(0, n, size=n)])[0, 1] for _ in range(n_boot)])
    lo, hi = np.nanpercentile(cors, [2.5, 97.5])
    return {"n": int(n), "n_boot": int(n_boot), "ci_95": [float(lo), float(hi)], "note": "ok"}

def window_sensitivity(returns, asset_a, asset_b, end_date, windows, threshold):
    rows = []
    for w in windows:
        c = pair_corr_at_anchor(returns, asset_a, asset_b, end_date, w)
        connected = None if c is None else (abs(c) >= threshold)
        rows.append({"lookback": int(w), "corr": c, "edge_status": None if connected is None else ("connected" if connected else "not_connected")})
    return rows

def corr_cluster_order(corr):
    if corr is None or corr.empty or len(corr) <= 2:
        return list(corr.index) if corr is not None else []
    c = corr.copy().fillna(0.0).clip(-1.0, 1.0)
    dist = 1.0 - c
    np.fill_diagonal(dist.values, 0.0)
    condensed = squareform(dist.values, checks=False)
    Z = linkage(condensed, method="average")
    order_idx = leaves_list(Z)
    return c.index.to_numpy()[order_idx].tolist()

# =============================================================================
# AI / LLM FUNCTIONS
# =============================================================================
PROVIDER_REGISTRY: dict[str, dict[str, str]] = {
    "OpenAI": {
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4.1-mini",
    },
    "Google Gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "default_model": "gemini-2.0-flash",
    },
    "Kimi (Moonshot)": {
        "base_url": "https://api.moonshot.cn/v1",
        "default_model": "moonshot-v1-8k",
    },
    "NVIDIA NIM": {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "default_model": "moonshotai/kimi-k2.5",
    },
    "DeepSeek": {
        "base_url": "https://api.deepseek.com",
        "default_model": "deepseek-chat",
    },
    "Groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "llama-3.3-70b-versatile",
    },
    "Together AI": {
        "base_url": "https://api.together.xyz/v1",
        "default_model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    },
    "Custom (OpenAI-compatible)": {
        "base_url": "",
        "default_model": "",
    },
}
PROVIDER_NAMES: list[str] = list(PROVIDER_REGISTRY.keys())

# Shorter, narrative-focused prompt for faster + more useful output
SYSTEM_PROMPT = """You are a senior quantitative analyst inside "Contagion Explorer", producing institutional-quality research notes.

Analyze the quantitative evidence about two assets and produce a structured research note. Think step by step:
1. What does the current correlation level tell us about co-movement?
2. How has the relationship changed recently (4-week delta)? Is it strengthening or weakening?
3. What do the lead-lag cross-correlations reveal about predictability?
4. Is the relationship stable (narrow bootstrap CI) or unreliable (wide CI, small sample)?
5. Do the window sensitivity tests show the correlation is robust across timeframes?
6. What do the extreme move overlaps tell us about tail-risk co-movement?

Rules:
1. Base your analysis ONLY on the provided evidence. Be specific — cite actual numbers.
2. Discuss "co-movement" / "association" / "lead-lag predictability" — never claim causation.
3. If the bootstrap CI is wide or the sample is small, flag the relationship as unreliable.
4. If lead-lag correlations are all near zero, clearly state no predictive lead-lag exists.
5. Compare 20d/60d/120d window sensitivity to assess if the relationship is timeframe-dependent.
6. Be concise, precise, and honest about uncertainty. No hype, no trading advice.
7. Return ONLY valid JSON matching the schema below — no markdown, no extra text.

JSON schema:
{
  "pair": {"asset_a": string, "asset_b": string},
  "time_context": {"asof_week": string, "lookback_days": number},
  "regime_summary": {"label": "calm"|"mixed"|"stress"|"unclear", "one_sentence": string},
  "story": string,
  "what_changed": [{"finding": string, "evidence": string}],
  "lead_lag": {"summary": string, "likely_leader": "A"|"B"|"none"|"unclear", "supporting_evidence": [string]},
  "relationship_now": {"corr_now": number, "corr_change_vs_4w": number, "edge_status": "connected"|"not_connected", "interpretation": string},
  "reliability": {"overall_confidence": "high"|"medium"|"low", "notes": [string]},
  "practical_implications": [string],
  "caveats": [string]
}

"story" should be 3-5 sentences. Start with the current relationship state, then describe what changed, then interpret what it means. Reference specific numbers (e.g., "correlation rose from 0.45 to 0.72 over 4 weeks").
"what_changed" must have 2-4 items, each citing specific evidence (numbers, CI ranges, lag correlations).
"practical_implications" must have exactly 3 items — actionable observations for portfolio managers.
"caveats" must have exactly 3 items — specific limitations of this analysis."""


def _get_llm_client(api_key: str, provider: str, custom_base_url: str = ""):
    """Create an OpenAI-compatible client for the selected provider."""
    from openai import OpenAI
    reg = PROVIDER_REGISTRY.get(provider, PROVIDER_REGISTRY["OpenAI"])
    base_url = reg["base_url"]
    if provider == "Custom (OpenAI-compatible)":
        base_url = (custom_base_url or "").strip()
        if not base_url:
            raise Exception("Custom provider requires a base URL (set in Settings).")
    return OpenAI(api_key=api_key, base_url=base_url or None, timeout=120)


def _provider_chat_kwargs(provider: str) -> dict[str, Any]:
    if provider == "NVIDIA NIM":
        return {"extra_body": {"chat_template_kwargs": {"thinking": False}}}
    return {}


def _chat_completion_create(*, client, provider, model, messages, temperature, max_tokens, stream=False):
    kwargs = _provider_chat_kwargs(provider)
    return client.chat.completions.create(
        model=model, messages=messages, temperature=temperature,
        max_tokens=max_tokens, stream=stream, **kwargs,
    )


def _api_key_fingerprint(api_key: str) -> str:
    if not api_key:
        return ""
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:12]


def verify_llm_api_key(provider: str, model: str, api_key: str, custom_base_url: str = "") -> dict[str, Any]:
    if not api_key:
        return {"ok": False, "message": "API key is empty."}
    try:
        client = _get_llm_client(api_key, provider=provider, custom_base_url=custom_base_url)
        started = time.perf_counter()
        resp = _chat_completion_create(
            client=client, provider=provider, model=model,
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
            temperature=0, max_tokens=8, stream=False,
        )
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        content = (resp.choices[0].message.content or "").strip()
        usage = getattr(resp, "usage", None)
        total_tokens = getattr(usage, "total_tokens", None) if usage else None
        return {
            "ok": True, "message": content or "Request succeeded.",
            "latency_ms": elapsed_ms, "provider": provider, "model": model,
            "token_usage": total_tokens,
            "checked_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "key_fp": _api_key_fingerprint(api_key),
        }
    except Exception as e:
        return {
            "ok": False, "message": str(e),
            "provider": provider, "model": model,
            "checked_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "key_fp": _api_key_fingerprint(api_key),
        }


def _extract_text_from_chat_response(resp: Any) -> str:
    choices = getattr(resp, "choices", None) or []
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            text_piece = None
            if isinstance(item, dict):
                text_piece = item.get("text") or item.get("content")
            else:
                text_piece = getattr(item, "text", None) or getattr(item, "content", None)
            if isinstance(text_piece, str) and text_piece.strip():
                chunks.append(text_piece)
        if chunks:
            return "\n".join(chunks)
    for attr in ("output_text", "text"):
        alt = getattr(resp, attr, None)
        if isinstance(alt, str) and alt.strip():
            return alt
    return ""


def _extract_json_object_text(text: str) -> str | None:
    if not text:
        return None
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _parse_json_from_llm_response(resp: Any, context: str) -> dict[str, Any]:
    text = _extract_text_from_chat_response(resp).strip()
    if not text:
        raise Exception(f"{context}: model returned empty content. Try a different model or run Test API Key.")
    json_text = _extract_json_object_text(text)
    if not json_text:
        preview = text[:240].replace("\n", " ")
        raise Exception(f"{context}: no JSON object found. Preview: {preview}")
    try:
        return json.loads(json_text)
    except json.JSONDecodeError as e:
        preview = json_text[:240].replace("\n", " ")
        raise Exception(f"{context}: invalid JSON ({e}). Preview: {preview}")


def _slim_evidence(evidence_json: dict, fast: bool = True) -> dict:
    """Strip down the evidence payload to reduce tokens for faster inference."""
    if not fast:
        return evidence_json
    return {
        "pair": evidence_json.get("pair"),
        "time_context": evidence_json.get("time_context"),
        "relationship": evidence_json.get("relationship"),
        "lead_lag": {
            "a_leads_b": evidence_json.get("lead_lag", {}).get("a_leads_b", [])[:3],
            "b_leads_a": evidence_json.get("lead_lag", {}).get("b_leads_a", [])[:3],
        },
        "extreme_move_overlap": evidence_json.get("extreme_move_overlap"),
        "stability": {
            "window_sensitivity": evidence_json.get("stability", {}).get("window_sensitivity"),
            "bootstrap_ci_95": evidence_json.get("stability", {}).get("bootstrap_ci_95"),
            "effective_sample_size": evidence_json.get("stability", {}).get("effective_sample_size"),
        },
    }


def generate_ai_brief(evidence_json, provider, model, api_key, depth="Fast", custom_base_url=""):
    if not api_key:
        raise Exception("API key required")
    from openai import OpenAI

    fast = depth == "Fast"
    slim = _slim_evidence(evidence_json, fast=fast)
    user_prompt = f"Analyze this evidence and return valid JSON:\n\n{json.dumps(slim, indent=1)}"
    max_tokens = 1400 if fast else 2500

    client = _get_llm_client(api_key, provider=provider, custom_base_url=custom_base_url)
    resp = _chat_completion_create(
        client=client, provider=provider, model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2, max_tokens=max_tokens, stream=False,
    )

    try:
        return _parse_json_from_llm_response(resp, context="AI brief")
    except Exception:
        # Retry with stricter instruction
        retry_msgs = [
            {"role": "system", "content": SYSTEM_PROMPT + "\nReturn ONLY valid JSON. No markdown fences."},
            {"role": "user", "content": user_prompt + "\n\nOutput a single JSON object."},
        ]
        retry_resp = _chat_completion_create(
            client=client, provider=provider, model=model,
            messages=retry_msgs, temperature=0.1, max_tokens=max_tokens, stream=False,
        )
        return _parse_json_from_llm_response(retry_resp, context="AI brief retry")


def fetch_news(ticker, api_key=None):
    """Fetch real news from NewsAPI. Returns empty list if no key — never fakes headlines."""
    if not api_key:
        return []
    try:
        url = f"https://newsapi.org/v2/everything?q={ticker}&sortBy=publishedAt&pageSize=5&apiKey={api_key}"
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return r.json().get("articles", [])
    except Exception:
        pass
    return []


def analyze_news_with_ai(news_items, ticker, evidence, api_key, provider, model, depth="Deep", custom_base_url=""):
    if not api_key:
        return None
    try:
        max_news = 2 if depth == "Fast" else 4
        news_text = "\n".join([f"- {n.get('title','')}: {n.get('description','')[:120]}" for n in news_items[:max_news]])

        rel = evidence.get('relationship', {})
        corr_now = rel.get('corr_now')
        corr_change = rel.get('corr_change_vs_4w')
        corr_now_str = f"{corr_now:.3f}" if corr_now is not None else "N/A"
        corr_change_str = f"{corr_change:+.3f}" if corr_change is not None else "N/A"
        edge_status = rel.get('edge_status', 'unknown')

        prompt = f"""Analyze the relationship between recent news and market data for {ticker}.

NEWS:
{news_text}

MARKET DATA:
- Correlation: {corr_now_str}
- 4w change: {corr_change_str}
- Edge: {edge_status}

Return JSON:
{{"sentiment":"positive"|"negative"|"neutral","sentiment_score":-100 to 100,"news_price_alignment":"aligned"|"divergent"|"unclear","key_themes":["..."],"correlation_explanation":"...","risk_factors":["..."],"outlook":"bullish"|"bearish"|"neutral","confidence":"high"|"medium"|"low"}}
Output only valid JSON."""

        client = _get_llm_client(api_key, provider=provider, custom_base_url=custom_base_url)
        resp = _chat_completion_create(
            client=client, provider=provider, model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2, max_tokens=350 if depth == "Fast" else 700, stream=False,
        )
        return _parse_json_from_llm_response(resp, context=f"News analysis for {ticker}")
    except Exception as e:
        return {"error": str(e)}


def _is_openai_provider(provider: str) -> bool:
    return provider == "OpenAI"


def generate_pair_explanation(asset_a: str, asset_b: str, corr_now: float | None,
                               provider: str, model: str, api_key: str,
                               custom_base_url: str = "") -> dict:
    """Ask the LLM to explain in plain English what each company does and WHY they are correlated."""
    if not api_key:
        return {"error": "No API key"}
    corr_str = f"{corr_now:+.2f}" if corr_now is not None else "unknown"
    prompt = f"""You are a clear, engaging financial educator explaining stock market relationships to a curious non-expert.

Two assets show a correlation of {corr_str}: {asset_a} and {asset_b}.

Write a short, human explanation covering:
1. What is {asset_a}? (1-2 sentences — what the company/fund does, what sector, why it matters)
2. What is {asset_b}? (1-2 sentences — same)
3. Why are they correlated? Give the most likely real-world reasons — shared macro drivers, supply chain links, same sector, investor behavior, etc. Be specific and interesting. If the correlation is negative, explain why they move in opposite directions.

Rules:
- Write in plain English. No jargon without explanation.
- Be concrete and specific — name actual industries, products, or economic forces.
- Keep total length to 4-6 sentences.
- If you genuinely cannot explain a plausible relationship, say so honestly.
- Do NOT use bullet points. Write in flowing prose.

Return ONLY a JSON object:
{{"asset_a_description": "...", "asset_b_description": "...", "correlation_explanation": "...", "confidence": "high"|"medium"|"low"}}"""

    try:
        client = _get_llm_client(api_key, provider=provider, custom_base_url=custom_base_url)
        resp = _chat_completion_create(
            client=client, provider=provider, model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3, max_tokens=400, stream=False,
        )
        return _parse_json_from_llm_response(resp, context="Pair explanation")
    except Exception as e:
        return {"error": str(e)}


def web_search_deep_analysis(asset_a, asset_b, evidence, api_key, model="gpt-4o-mini"):
    """Use OpenAI Responses API with web_search tool for real-time internet-powered analysis."""
    from openai import OpenAI

    rel = evidence.get('relationship', {})
    corr_now = rel.get('corr_now')
    corr_change = rel.get('corr_change_vs_4w')
    edge_status = rel.get('edge_status', 'unknown')
    lookback = evidence.get('time_context', {}).get('lookback_window_trading_days', 60)

    corr_str = f"{corr_now:.3f}" if corr_now is not None else "N/A"
    change_str = f"{corr_change:+.3f}" if corr_change is not None else "N/A"

    lag_data = evidence.get('lead_lag', {})
    a_leads = lag_data.get('a_leads_b', [])[:3]
    b_leads = lag_data.get('b_leads_a', [])[:3]
    lag_summary = ""
    if a_leads:
        lag_vals = [f"lag-{x['lag_days']}d: {x['corr']:.3f}" for x in a_leads if x.get('corr') is not None]
        if lag_vals:
            lag_summary += f"\n- {asset_a} leading {asset_b}: {', '.join(lag_vals)}"
    if b_leads:
        lag_vals = [f"lag-{x['lag_days']}d: {x['corr']:.3f}" for x in b_leads if x.get('corr') is not None]
        if lag_vals:
            lag_summary += f"\n- {asset_b} leading {asset_a}: {', '.join(lag_vals)}"

    overlap = evidence.get('extreme_move_overlap', {})
    overlap_str = f"{overlap.get('overlap_days', 0)} overlapping extreme days ({overlap.get('same_direction', 0)} same direction, {overlap.get('opposite_direction', 0)} opposite)"

    stability = evidence.get('stability', {})
    ci = stability.get('bootstrap_ci_95', {})
    ci_str = f"[{ci['ci_95'][0]:.3f}, {ci['ci_95'][1]:.3f}]" if ci.get('ci_95') else "N/A"
    n_effective = stability.get('effective_sample_size', 'N/A')

    sens = stability.get('window_sensitivity', [])
    sens_str = ", ".join([f"{s['lookback']}d: {s['corr']:.3f}" if s.get('corr') is not None else f"{s['lookback']}d: N/A" for s in sens]) if sens else "N/A"

    prompt = f"""You are a senior financial analyst with access to the internet. Search the web for the latest news, market developments, and analysis about {asset_a} and {asset_b}.

QUANTITATIVE EVIDENCE FROM OUR ANALYSIS:
- Rolling {lookback}-day correlation: {corr_str}
- 4-week correlation change: {change_str}
- Edge status (|ρ| ≥ threshold): {edge_status}
- Bootstrap 95% CI for correlation: {ci_str} (n={n_effective})
- Window sensitivity: {sens_str}
- Lead-lag cross-correlations:{lag_summary if lag_summary else ' No significant lead-lag detected'}
- Extreme move overlap (top-10): {overlap_str}

YOUR TASKS:
1. Search the web for recent news, earnings reports, analyst commentary, and macro events affecting {asset_a}
2. Search the web for recent news, earnings reports, analyst commentary, and macro events affecting {asset_b}
3. Using what you find, explain WHY these two assets show a correlation of {corr_str} — what fundamental, sector, or macro factors drive it?
4. If the correlation changed by {change_str} over 4 weeks, explain what recent events likely caused that shift
5. Identify shared risk factors and potential divergence catalysts

Return your analysis as a JSON object with this exact structure:
{{
    "asset_a": {{
        "ticker": "{asset_a}",
        "recent_headlines": ["headline 1", "headline 2", "headline 3"],
        "sentiment": "positive" or "negative" or "neutral",
        "sentiment_score": -100 to 100,
        "key_drivers": ["driver 1", "driver 2"],
        "recent_context": "2-3 sentences on recent price action and catalysts"
    }},
    "asset_b": {{
        "ticker": "{asset_b}",
        "recent_headlines": ["headline 1", "headline 2", "headline 3"],
        "sentiment": "positive" or "negative" or "neutral",
        "sentiment_score": -100 to 100,
        "key_drivers": ["driver 1", "driver 2"],
        "recent_context": "2-3 sentences on recent price action and catalysts"
    }},
    "correlation_analysis": {{
        "explanation": "3-5 sentences explaining WHY these assets have their current correlation level, referencing specific real-world factors you found",
        "change_explanation": "2-3 sentences explaining what drove the recent 4-week correlation change of {change_str}",
        "shared_drivers": ["shared macro/sector factor 1", "factor 2", "factor 3"],
        "divergence_risks": ["what could cause correlation to break 1", "risk 2"],
        "correlation_direction": "strengthening" or "weakening" or "stable",
        "direction_rationale": "1-2 sentences"
    }},
    "market_context": {{
        "macro_environment": "2-3 sentences on the current macro backdrop and how it affects this pair",
        "risk_sentiment": "risk-on" or "risk-off" or "mixed",
        "key_upcoming_events": ["event that could affect this pair 1", "event 2"]
    }},
    "confidence": "high" or "medium" or "low"
}}
Output ONLY valid JSON. No markdown fences, no extra text."""

    client = OpenAI(api_key=api_key)
    try:
        response = client.responses.create(
            model=model,
            tools=[{"type": "web_search"}],
            input=prompt,
        )
        text = response.output_text
    except (AttributeError, TypeError):
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1500,
        )
        text = (response.choices[0].message.content or "")
    except Exception as e:
        return {"error": str(e)}

    json_text = _extract_json_object_text(text)
    if json_text:
        try:
            return json.loads(json_text)
        except json.JSONDecodeError:
            pass
    return {"raw_analysis": text}


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================
def create_correlation_chart(hist_df, threshold, asset_a, asset_b, lookback):
    fig = go.Figure()
    if not hist_df.empty:
        fig.add_trace(go.Scatter(
            x=hist_df["week"], y=hist_df["corr"], mode="lines",
            line=dict(color="#60a5fa", width=3),
            fill="tozeroy", fillcolor="rgba(96,165,250,0.12)",
            name="Correlation",
            hovertemplate="<b>%{x}</b><br>ρ = %{y:.3f}<extra></extra>"
        ))
        fig.add_trace(go.Scatter(
            x=[hist_df["week"].iloc[-1]], y=[hist_df["corr"].iloc[-1]],
            mode="markers",
            marker=dict(size=12, color="#a78bfa", symbol="circle", line=dict(width=2, color="white")),
            name="Current",
            hovertemplate="<b>Current</b><br>ρ = %{y:.3f}<extra></extra>"
        ))
    fig.add_hline(y=threshold, line_dash="dash", line_color="#4ade80", line_width=1.5,
                  annotation_text=f"+{threshold}", annotation_position="right", annotation_font_color="#4ade80")
    fig.add_hline(y=-threshold, line_dash="dash", line_color="#f87171", line_width=1.5,
                  annotation_text=f"-{threshold}", annotation_position="right", annotation_font_color="#f87171")
    fig.add_hline(y=0, line_color="rgba(148,163,184,0.3)", line_width=1)
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(30,41,59,0.4)",
        height=380, margin=dict(l=10, r=60, t=50, b=10),
        title=dict(text=f"Rolling Correlation: {asset_a} vs {asset_b}", font=dict(size=16, color="#f8fafc"), x=0.5),
        yaxis=dict(range=[-1, 1], gridcolor="rgba(148,163,184,0.1)", tickformat=".2f", title="ρ"),
        xaxis=dict(gridcolor="rgba(148,163,184,0.1)", showgrid=False),
        showlegend=False, hovermode="x unified", font=dict(color="#e2e8f0"),
    )
    return fig

def create_heatmap(corr, order, title="Correlation Matrix"):
    cm = corr.reindex(index=order, columns=order)
    colorscale = [
        [0.0, "#dc2626"],
        [0.15, "#ef4444"],
        [0.35, "#6b7280"],
        [0.5, "#334155"],
        [0.65, "#6b7280"],
        [0.85, "#22c55e"],
        [1.0, "#16a34a"],
    ]
    text_vals = np.around(cm.values, 2).astype(str)
    fig = go.Figure(data=go.Heatmap(
        z=cm.values, x=order, y=order, zmin=-1, zmax=1, colorscale=colorscale,
        text=text_vals, texttemplate="%{text}", textfont=dict(size=10, color="#e2e8f0"),
        hovertemplate="%{x} vs %{y}<br>ρ = %{z:.3f}<extra></extra>",
        colorbar=dict(
            title=dict(text="ρ", font=dict(color="#e2e8f0")),
            thickness=12, tickfont=dict(color="#94a3b8", size=10),
            tickvals=[-1, -0.5, 0, 0.5, 1],
        ),
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(30,41,59,0.4)",
        height=460, margin=dict(l=10, r=10, t=50, b=10),
        title=dict(text=title, font=dict(size=15, color="#f8fafc"), x=0.5),
        xaxis=dict(tickangle=45, tickfont=dict(color="#94a3b8", size=10), side="bottom"),
        yaxis=dict(tickfont=dict(color="#94a3b8", size=10), autorange="reversed"),
        font=dict(color="#e2e8f0"),
    )
    return fig

def create_network_viz(corr, tickers, threshold):
    import math
    n = len(tickers)
    angles = [2 * math.pi * i / n for i in range(n)]
    node_x = [0.5 + 0.35 * math.cos(a) for a in angles]
    node_y = [0.5 + 0.35 * math.sin(a) for a in angles]
    edge_traces = []
    for i, a in enumerate(tickers):
        for j, b in enumerate(tickers):
            if i >= j:
                continue
            val = corr.loc[a, b] if a in corr.index and b in corr.columns else np.nan
            if pd.isna(val) or abs(val) < threshold:
                continue
            color = "#60a5fa" if val > 0 else "#f87171"
            width = 1 + 4 * abs(val)
            opacity = 0.3 + 0.5 * abs(val)
            edge_traces.append(go.Scatter(
                x=[node_x[i], node_x[j]], y=[node_y[i], node_y[j]], mode="lines",
                line=dict(width=width, color=color), opacity=opacity, hoverinfo="skip", showlegend=False,
            ))
    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers+text", text=tickers, textposition="middle center",
        textfont=dict(size=10, color="#f8fafc", family="Inter"),
        marker=dict(size=35, color="#1e293b", line=dict(width=2, color="#60a5fa")),
        hovertemplate="%{text}<extra></extra>", showlegend=False,
    )
    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(30,41,59,0.4)",
        height=520, margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(visible=False, range=[0, 1]), yaxis=dict(visible=False, range=[0, 1], scaleanchor="x"),
        showlegend=False, font=dict(color="#e2e8f0"),
    )
    return fig

def create_price_overlay(prices, tickers, lookback_days=None):
    """Normalized price chart overlaying selected tickers (indexed to 100 at start)."""
    df = prices[tickers].dropna()
    if lookback_days:
        df = df.tail(lookback_days)
    if df.empty:
        return go.Figure()
    normalized = df / df.iloc[0] * 100
    palette = ["#60a5fa", "#a78bfa", "#4ade80", "#fbbf24", "#f87171", "#38bdf8", "#e879f9", "#fb923c"]
    fig = go.Figure()
    for i, ticker in enumerate(tickers):
        if ticker in normalized.columns:
            color = palette[i % len(palette)]
            fig.add_trace(go.Scatter(
                x=normalized.index, y=normalized[ticker],
                mode="lines", name=ticker,
                line=dict(color=color, width=2.5),
                hovertemplate=f"<b>{ticker}</b><br>%{{x}}<br>Index: %{{y:.1f}}<extra></extra>",
            ))
    fig.add_hline(y=100, line_color="rgba(148,163,184,0.25)", line_width=1, line_dash="dot")
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(30,41,59,0.4)",
        height=360, margin=dict(l=10, r=10, t=50, b=10),
        title=dict(text="Normalized Prices (indexed to 100)", font=dict(size=15, color="#f8fafc"), x=0.5),
        yaxis=dict(gridcolor="rgba(148,163,184,0.1)", title=""),
        xaxis=dict(gridcolor="rgba(148,163,184,0.1)", showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#e2e8f0", size=11)),
        hovermode="x unified", font=dict(color="#e2e8f0"),
    )
    return fig


def create_gauge(value, title, min_val=-1, max_val=1):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value, domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 13, 'color': '#e2e8f0'}},
        number={'font': {'size': 26, 'color': '#f8fafc'}, 'valueformat': '.3f'},
        gauge={
            'axis': {'range': [min_val, max_val], 'tickwidth': 1, 'tickcolor': '#94a3b8'},
            'bar': {'color': '#60a5fa', 'thickness': 0.6},
            'bgcolor': 'rgba(30,41,59,0.6)', 'borderwidth': 2, 'bordercolor': 'rgba(148,163,184,0.2)',
            'steps': [
                {'range': [min_val, -0.5], 'color': 'rgba(248,113,113,0.15)'},
                {'range': [-0.5, 0.5], 'color': 'rgba(148,163,184,0.08)'},
                {'range': [0.5, max_val], 'color': 'rgba(96,165,250,0.15)'},
            ],
            'threshold': {'line': {'color': '#a78bfa', 'width': 3}, 'thickness': 0.8, 'value': value}
        }
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=220,
        margin=dict(l=20, r=20, t=45, b=15), font=dict(color="#e2e8f0"),
    )
    return fig

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def format_week_label(ts):
    return ts.strftime("%Y-%m-%d")

def get_correlation_series(md, asset_a, asset_b, lookback):
    rows = []
    for anchor in md.weekly_anchors:
        c = pair_corr_at_anchor(md.returns, asset_a, asset_b, anchor, lookback)
        if c is not None:
            rows.append({"week": anchor, "corr": c})
    return pd.DataFrame(rows)

def build_evidence_json(md, asset_a, asset_b, lookback, threshold):
    asof_week = md.weekly_anchors[-1]
    stats = compute_window_stats(md.returns[[asset_a, asset_b]], asof_week, lookback)
    corr_now = stats.corr.loc[asset_a, asset_b] if asset_a in stats.corr.index and asset_b in stats.corr.columns else np.nan
    corr_now = None if pd.isna(corr_now) else float(corr_now)
    idx = md.weekly_anchors.index(asof_week)
    corr_4w = None
    if idx >= 4:
        corr_4w = pair_corr_at_anchor(md.returns, asset_a, asset_b, md.weekly_anchors[idx - 4], lookback)
    corr_change = None
    if corr_now is not None and corr_4w is not None:
        corr_change = float(corr_now - corr_4w)
    edge_status = None
    if corr_now is not None:
        edge_status = "connected" if abs(corr_now) >= threshold else "not_connected"
    lag = lagged_correlations(stats.returns_window, asset_a, asset_b, max_lag=5)
    overlap = extreme_move_overlap(stats.returns_window, asset_a, asset_b, top_n=10)
    sens = window_sensitivity(md.returns, asset_a, asset_b, asof_week, [20, 60, 120], threshold)
    ci = bootstrap_corr_ci(stats.returns_window, asset_a, asset_b, n_boot=400)
    missing = stats.returns_window[[asset_a, asset_b]].isna().mean().to_dict()
    n_effective = int(stats.returns_window[[asset_a, asset_b]].dropna().shape[0])
    return {
        "universe": list(md.tickers_ok),
        "pair": {"asset_a": asset_a, "asset_b": asset_b},
        "time_context": {"asof_week": format_week_label(asof_week), "lookback_window_trading_days": int(lookback), "history_months": 12},
        "relationship": {"corr_now": corr_now, "corr_4w_ago": corr_4w, "corr_change_vs_4w": corr_change, "edge_threshold": float(threshold), "edge_status": edge_status},
        "lead_lag": lag,
        "extreme_move_overlap": overlap,
        "stability": {"window_sensitivity": sens, "bootstrap_ci_95": ci, "missing_frac_in_window": missing, "effective_sample_size": n_effective},
    }

# =============================================================================
# COMPANY NAME → TICKER LOOKUP (no LLM needed)
# =============================================================================
def _normalize_company_name(name: str) -> str:
    import re
    text = name.lower().strip()
    text = text.replace("&", "and")
    text = re.sub(r"[’'`]", "", text)
    text = re.sub(r"[\.,/\\\-()]", " ", text)
    text = re.sub(
        r"\b(incorporated|inc|corp|corporation|co|company|ltd|limited|plc|sa|ag|nv|oyj|holdings|holding|group)\b",
        " ",
        text,
    )
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _load_ticker_lookup() -> dict:
    """Load name→ticker mapping from src/company_tickers.json."""
    import json, pathlib
    json_path = pathlib.Path(__file__).parent / "src" / "company_tickers.json"
    if not json_path.exists():
        return {}
    with open(json_path, "r") as f:
        raw = json.load(f)
    mapping: dict[str, str] = {}
    entries = raw.values() if isinstance(raw, dict) else raw
    for item in entries:
        if not isinstance(item, dict):
            continue
        name = item.get("title") or item.get("name")
        ticker = item.get("ticker")
        if not name or not ticker:
            continue
        base = name.lower().strip()
        normalized = _normalize_company_name(name)
        for key in {base, normalized}:
            if key and key not in mapping:
                mapping[key] = ticker
        # Small alias for common "Alphabet" naming
        if "alphabet" in normalized and "google" not in mapping:
            mapping["google"] = ticker
    return mapping

_NAME_TO_TICKER: dict = _load_ticker_lookup()


def _resolve_names_to_tickers(raw: str) -> list[str]:
    """Convert free-text to ticker symbols using company_tickers.json.
    Works on sentences like 'Is Apple correlated with Gold?' — no LLM needed.
    Always returns tickers, never duplicates, never adds something already resolved."""
    import re
    text_lower = raw.lower().strip()
    text_norm = _normalize_company_name(raw)
    results = []

    # Pass 1: scan for known names (longest first to avoid 'gold' matching inside 'goldman')
    sorted_names = sorted(_NAME_TO_TICKER.keys(), key=len, reverse=True)
    remaining = text_lower
    remaining_norm = text_norm
    def _word_match(haystack: str, needle: str) -> bool:
        return f" {needle} " in f" {haystack} "
    for name in sorted_names:
        if _word_match(remaining, name) or _word_match(remaining_norm, name):
            ticker = _NAME_TO_TICKER[name]
            if ticker not in results:
                results.append(ticker)
            if _word_match(remaining, name):
                remaining = remaining.replace(name, " " * len(name), 1)
            if _word_match(remaining_norm, name):
                remaining_norm = remaining_norm.replace(name, " " * len(name), 1)

    # Pass 2: catch explicit ticker symbols not covered by names (e.g. AMD, TSM)
    _skip = {"A", "I", "IS", "IN", "OR", "AND", "THE", "FOR", "ARE", "TO", "OF",
             "MY", "HOW", "WHY", "DO", "VS", "IT", "BE", "BY", "WITH", "AT", "IF",
             "ME", "WE", "US", "AN", "AS", "ON", "UP", "GO", "SO", "NO", "DO"}
    for match in re.finditer(r'\b([A-Z]{1,5}(?:[.\-][A-Z]{1,2})?)\b', raw.upper()):
        t = match.group(1)
        if t not in _skip and t not in results:
            results.append(t)

    return results


# =============================================================================
# SIDEBAR
# =============================================================================
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 1.5rem; padding: 0.75rem;">
            <div style="font-size: 2.5rem; margin-bottom: 0.25rem;">🌊</div>
            <h2 style="margin: 0; font-size: 1.35rem; font-weight: 700;">Contagion</h2>
            <p style="color: #8b5cf6; margin: 0; font-size: 0.8rem; font-weight: 600;">AI-Powered Explorer</p>
        </div>
        """, unsafe_allow_html=True)

        pages = {
            "home":         ("🌊 Contagion Map",     "Portfolio risk at a glance"),
            "ai_insights":  ("🤖 AI Insights",       "Ask a question · get a verdict"),
            "settings":     ("⚙️ Settings",           "API keys · data source"),
        }
        st.caption("NAVIGATION")
        for key, (label, sublabel) in pages.items():
            is_active = st.session_state.get("current_page", "home") == key
            if st.button(label, key=f"nav_{key}", use_container_width=True, type="primary" if is_active else "secondary"):
                st.session_state["current_page"] = key
                st.rerun()
            if not is_active:
                st.markdown(f'<div style="color:#475569; font-size:0.72rem; margin:-0.4rem 0 0.4rem 0.5rem;">{sublabel}</div>', unsafe_allow_html=True)

        st.divider()
        st.caption("UNIVERSE")
        all_options = list(ALL_AVAILABLE_TICKERS.keys()) + [
            t for t in st.session_state.get("custom_tickers", [])
            if t not in ALL_AVAILABLE_TICKERS
        ]
        option_labels = {t: f"{t} — {ALL_AVAILABLE_TICKERS.get(t, t)}" for t in all_options}
        current_sel = [t for t in st.session_state.get("selected_tickers", DEFAULT_SELECTED) if t in all_options]

        # Widget key is "_ms_ui" — completely separate from the data key "selected_tickers"
        # This means code can freely write to "selected_tickers" without Streamlit complaining
        chosen = st.multiselect(
            "Active assets",
            options=all_options,
            default=current_sel,
            format_func=lambda t: option_labels.get(t, t),
            key="_ms_ui",
            label_visibility="collapsed",
        )
        # Sync widget value back to data key whenever user changes selection
        if chosen != st.session_state.get("selected_tickers"):
            st.session_state["selected_tickers"] = chosen

        if len(chosen) < 2:
            st.caption("⚠️ Select at least 2 assets.")

        st.caption("Add more (name or ticker):")
        sb_input = st.text_input(
            "Add assets", placeholder="e.g. Apple, Tesla, NVDA",
            key="sidebar_add_tickers", label_visibility="collapsed"
        )
        if st.button("＋ Add", key="sb_add_btn", use_container_width=True):
            raw = st.session_state.get("sidebar_add_tickers", "").strip()
            if raw:
                resolved = _resolve_names_to_tickers(raw)
                newly_added = []
                for t in resolved:
                    if t not in ALL_AVAILABLE_TICKERS and t not in st.session_state.get("custom_tickers", []):
                        st.session_state["custom_tickers"].append(t)
                        newly_added.append(t)
                    # Also auto-select newly resolved tickers
                    current_sel = list(st.session_state.get("selected_tickers", DEFAULT_SELECTED))
                    if t not in current_sel:
                        current_sel.append(t)
                        st.session_state["selected_tickers"] = current_sel
                        if t not in newly_added:
                            newly_added.append(t)
                if newly_added:
                    st.session_state["sb_last_added"] = newly_added
                    st.rerun()
                elif resolved:
                    st.session_state["sb_last_added"] = "already_in"
                else:
                    st.session_state["sb_last_added"] = "not_found"

        last = st.session_state.get("sb_last_added")
        if last == "already_in":
            st.caption("✓ Already selected.")
        elif last == "not_found":
            st.caption("⚠️ Not recognised — try a ticker symbol.")
        elif isinstance(last, list):
            st.caption(f"✓ Added: {', '.join(last)}")

        st.divider()
        st.caption("SETTINGS")
        months = st.radio("History", [12, 24], index=0, horizontal=True, label_visibility="collapsed")
        use_sample = st.toggle("Demo Mode", value=st.session_state["use_sample_data"], help="Use synthetic data for testing")
        st.session_state["use_sample_data"] = use_sample

        st.divider()
        prov = st.session_state.get("live_provider", "yahoo").capitalize()
        status = "🟢 Live" if not use_sample else "🟡 Demo"
        st.caption(f"{status} · {prov}")

        return months

# =============================================================================
# PAGE: AI INSIGHTS (MAIN FEATURE)
# =============================================================================
def render_ai_insights(md, months):
    import re

    # ── HERO: what this tool is for ──────────────────────────────────────────
    st.markdown("""
<div style="background:linear-gradient(135deg, rgba(59,130,246,0.13), rgba(139,92,246,0.13));
     border:1px solid rgba(139,92,246,0.30); border-radius:18px; padding:1.5rem 1.75rem; margin-bottom:1.5rem;">
  <div style="font-size:1.35rem; font-weight:800; color:#f8fafc; margin-bottom:0.4rem;">
    🌊 Contagion Explorer
  </div>
  <div style="font-size:0.95rem; color:#cbd5e1; line-height:1.65; max-width:780px; margin-bottom:1rem;">
    <b>Your job:</b> you manage a portfolio and are considering adding a position — or you want to stress-test
    an existing one. The question is always the same: <em>how correlated is this asset with what I already hold,
    is that relationship stable, and who is driving whom?</em>
    <br><br>
    This tool answers that by computing rolling correlations, lead-lag dynamics, and bootstrap confidence
    intervals from live market data — then sending all the evidence to an AI model that writes you a
    structured research brief and a plain-English verdict.
  </div>
  <div style="display:flex; gap:0.6rem; flex-wrap:wrap;">
    <span style="background:rgba(96,165,250,0.15); border:1px solid rgba(96,165,250,0.35); color:#60a5fa;
          padding:0.3rem 0.9rem; border-radius:20px; font-size:0.8rem; font-weight:600;">① Ask</span>
    <span style="color:#64748b; padding:0.3rem 0; font-size:0.8rem;">→</span>
    <span style="background:rgba(167,139,250,0.15); border:1px solid rgba(167,139,250,0.35); color:#a78bfa;
          padding:0.3rem 0.9rem; border-radius:20px; font-size:0.8rem; font-weight:600;">② Generate</span>
    <span style="color:#64748b; padding:0.3rem 0; font-size:0.8rem;">→</span>
    <span style="background:rgba(74,222,128,0.15); border:1px solid rgba(74,222,128,0.35); color:#4ade80;
          padding:0.3rem 0.9rem; border-radius:20px; font-size:0.8rem; font-weight:600;">③ Decide</span>
  </div>
</div>""", unsafe_allow_html=True)

    api_key = st.session_state.get("llm_api_key", "")
    if not api_key:
        st.info("🔑 **API Key Required** — Go to ⚙️ Settings to enter your LLM API key (OpenAI, Gemini, DeepSeek, Groq & more).")
        if st.button("Go to Settings →", type="primary"):
            st.session_state["current_page"] = "settings"
            st.rerun()
        return

    # Restore pair that was queued before a ticker-add rerun
    if st.session_state.get("pending_ai_a") and st.session_state["pending_ai_a"] in md.tickers_ok:
        st.session_state["ai_a"] = st.session_state.pop("pending_ai_a")
    if st.session_state.get("pending_ai_b") and st.session_state["pending_ai_b"] in md.tickers_ok:
        st.session_state["ai_b"] = st.session_state.pop("pending_ai_b")

    # ── STEP 1: NATURAL LANGUAGE QUESTION ────────────────────────────────────
    st.markdown("#### ① What do you want to know?")
    st.caption("Type any question — company names, tickers, or plain English. Unknown assets are added automatically.")

    q_col, btn_col = st.columns([5, 1])
    with q_col:
        nl_question = st.text_input(
            "Your question",
            value=st.session_state.get("ai_user_question", ""),
            placeholder='e.g. "Is Apple correlated with Microsoft?" · "Does SPY lead QQQ?"',
            label_visibility="collapsed",
            key="nl_question_input",
        )
    with btn_col:
        ask_clicked = st.button("→ Parse", use_container_width=True, key="nl_parse_btn")

    if ask_clicked and nl_question.strip():
        import re
        st.session_state["ai_user_question"] = nl_question.strip()

        # Resolve names + tickers from the question
        all_found = _resolve_names_to_tickers(nl_question)
        # Also include any already-in-universe tickers mentioned
        q_upper = nl_question.upper()
        for t in md.tickers_ok:
            if re.search(r'\b' + re.escape(t) + r'\b', q_upper) and t not in all_found:
                all_found.insert(0, t)  # universe hits first

        # Auto-add resolved tickers not yet loaded
        new_tickers = [t for t in all_found if t not in md.tickers_ok]
        if new_tickers:
            for t in new_tickers:
                if t not in st.session_state.get("custom_tickers", []):
                    st.session_state["custom_tickers"].append(t)
                # Also auto-select them
                current_sel = list(st.session_state.get("selected_tickers", DEFAULT_SELECTED))
                if t not in current_sel:
                    current_sel.append(t)
                    st.session_state["selected_tickers"] = current_sel
            st.toast(f"Added {', '.join(new_tickers)} — loading data…", icon="📈")
            if len(all_found) >= 2:
                st.session_state["pending_ai_a"] = all_found[0]
                st.session_state["pending_ai_b"] = all_found[1]
            st.rerun()

        # Pre-fill dropdowns
        if len(all_found) >= 2:
            st.session_state["ai_a"] = all_found[0]
            st.session_state["ai_b"] = all_found[1]
            st.toast(f"✅ {all_found[0]} vs {all_found[1]}", icon="✅")
            st.rerun()
        elif len(all_found) == 1:
            st.session_state["ai_a"] = all_found[0]
            st.toast(f"Found {all_found[0]} — pick the second asset below.", icon="ℹ️")
            st.rerun()
        else:
            st.warning("Couldn't identify any assets. Try company names like 'Apple' or tickers like 'AAPL'.")

    user_q = st.session_state.get("ai_user_question", "")
    if user_q:
        st.markdown(f"""<div style="background:rgba(96,165,250,0.07); border-left:3px solid #60a5fa;
border-radius:0 10px 10px 0; padding:0.55rem 1rem; margin:0.5rem 0 1rem 0;
font-size:0.87rem; color:#e2e8f0;">🗣️ <b>Your question:</b> {user_q}</div>""", unsafe_allow_html=True)

    # ── STEP 2: CONFIGURE THE PAIR ────────────────────────────────────────────
    st.markdown("#### ② Configure the pair")

    # Guard stale session state values — if they point to tickers not loaded, reset
    if st.session_state.get("ai_a") not in md.tickers_ok:
        st.session_state["ai_a"] = md.tickers_ok[0] if md.tickers_ok else None
    if st.session_state.get("ai_b") not in md.tickers_ok:
        st.session_state["ai_b"] = md.tickers_ok[1] if len(md.tickers_ok) > 1 else md.tickers_ok[0]

    col1, col2, col3, col4 = st.columns([1.1, 1.1, 0.8, 1])
    with col1:
        asset_a = st.selectbox("Primary Asset", md.tickers_ok, key="ai_a")
    with col2:
        b_opts = [t for t in md.tickers_ok if t != asset_a]
        # If ai_b was set to same as ai_a (edge case), reset it
        if st.session_state.get("ai_b") == asset_a and b_opts:
            st.session_state["ai_b"] = b_opts[0]
        asset_b = st.selectbox("Compare With", b_opts, key="ai_b")
    with col3:
        lookback = st.select_slider("Lookback Window", [20, 60, 120], value=60)
    with col4:
        depth = st.radio("Analysis Depth", ["Fast", "Deep"], index=0, horizontal=True,
                         help="Fast ≈ 5-15s · Deep adds live web research (OpenAI) or news context (other providers).")
        st.session_state["ai_analysis_depth"] = depth

    # Scenario hint based on selected pair
    scenario_hints = {
        ("SPY", "QQQ"): "Classic scenario: US large-cap equities vs tech-heavy index. Useful for checking whether your tech allocation adds real diversification.",
        ("SPY", "IEF"): "Equity-bond relationship. A negative correlation here is the classic 60/40 hedge — but it breaks down in inflationary regimes.",
        ("SPY", "GLD"): "Risk-on vs safe-haven. Gold tends to decouple from equities during stress — check whether that's happening now.",
        ("HYG", "SPY"): "Credit-equity signal. HYG leading SPY downward is a classic early-warning for equity drawdowns.",
        ("AAPL", "MSFT"): "Two mega-cap tech names. High correlation reduces diversification benefit — useful to know before sizing.",
    }
    pair_key = (asset_a, asset_b) if (asset_a, asset_b) in scenario_hints else (asset_b, asset_a) if (asset_b, asset_a) in scenario_hints else None
    if pair_key:
        st.markdown(f"""<div style="background:rgba(251,191,36,0.07); border-left:3px solid #fbbf24;
border-radius:0 10px 10px 0; padding:0.5rem 1rem; margin-bottom:0.75rem; font-size:0.85rem; color:#e2e8f0;">
💡 <b>Why this pair matters:</b> {scenario_hints[pair_key]}</div>""", unsafe_allow_html=True)

    # ── AI CONTEXT CARD: what each company is and why they're correlated ─────
    expl_cache_key = f"pair_expl_{asset_a}_{asset_b}"
    if st.session_state.get(expl_cache_key + "_pair") != (asset_a, asset_b):
        st.session_state.pop(expl_cache_key, None)

    if expl_cache_key not in st.session_state:
        with st.spinner(f"🤖 Looking up {asset_a} and {asset_b}…"):
            expl = generate_pair_explanation(
                asset_a, asset_b,
                corr_now=build_evidence_json(md, asset_a, asset_b, lookback, st.session_state["global_threshold"])["relationship"].get("corr_now"),
                provider=st.session_state.get("llm_provider", "OpenAI"),
                model=st.session_state.get("llm_model", "gpt-4.1-mini"),
                api_key=api_key,
                custom_base_url=st.session_state.get("llm_custom_base_url", ""),
            )
            st.session_state[expl_cache_key] = expl
            st.session_state[expl_cache_key + "_pair"] = (asset_a, asset_b)

    expl = st.session_state.get(expl_cache_key, {})
    if expl and not expl.get("error"):
        conf = expl.get("confidence", "medium")
        conf_color = {"high": "#4ade80", "medium": "#fbbf24", "low": "#f87171"}.get(conf, "#94a3b8")
        corr_display = build_evidence_json(md, asset_a, asset_b, lookback, st.session_state["global_threshold"])["relationship"].get("corr_now")
        corr_str = f"{corr_display:+.2f}" if corr_display is not None else "?"
        st.markdown(f"""
<div style="background:rgba(139,92,246,0.07); border:1px solid rgba(139,92,246,0.22); border-radius:14px; padding:1rem 1.3rem; margin-bottom:1rem;">
  <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.7rem;">
    <span style="font-size:1rem;">🤖</span>
    <span style="font-weight:700; color:#f8fafc; font-size:0.92rem;">What are these assets, and why are they correlated (ρ = {corr_str})?</span>
    <span style="margin-left:auto; background:{conf_color}22; border:1px solid {conf_color}; color:{conf_color}; padding:0.12rem 0.55rem; border-radius:10px; font-size:0.7rem; font-weight:600;">{conf.upper()}</span>
  </div>
  <div style="display:flex; gap:0.75rem; margin-bottom:0.8rem; flex-wrap:wrap;">
    <div style="flex:1; min-width:180px; background:rgba(96,165,250,0.07); border-radius:9px; padding:0.6rem 0.85rem;">
      <div style="color:#60a5fa; font-weight:700; font-size:0.73rem; text-transform:uppercase; letter-spacing:0.04em; margin-bottom:0.2rem;">{asset_a}</div>
      <div style="color:#e2e8f0; font-size:0.85rem; line-height:1.5;">{expl.get("asset_a_description", "")}</div>
    </div>
    <div style="flex:1; min-width:180px; background:rgba(167,139,250,0.07); border-radius:9px; padding:0.6rem 0.85rem;">
      <div style="color:#a78bfa; font-weight:700; font-size:0.73rem; text-transform:uppercase; letter-spacing:0.04em; margin-bottom:0.2rem;">{asset_b}</div>
      <div style="color:#e2e8f0; font-size:0.85rem; line-height:1.5;">{expl.get("asset_b_description", "")}</div>
    </div>
  </div>
  <div style="border-top:1px solid rgba(148,163,184,0.1); padding-top:0.65rem; color:#e2e8f0; font-size:0.88rem; line-height:1.6;">
    {expl.get("correlation_explanation", "")}
  </div>
</div>""", unsafe_allow_html=True)

    # Build evidence
    threshold = st.session_state["global_threshold"]
    evidence = build_evidence_json(md, asset_a, asset_b, lookback, threshold)
    st.session_state["last_pair_evidence_json"] = evidence

    # Fetch news
    news_a = fetch_news(asset_a, st.session_state.get("news_api_key"))
    news_b = fetch_news(asset_b, st.session_state.get("news_api_key"))

    # ── STEP 3: GENERATE ─────────────────────────────────────────────────────
    st.markdown("#### ③ Generate your research brief & decision verdict")
    provider_check = st.session_state.get("llm_provider", "OpenAI")
    if depth == "Deep" and _is_openai_provider(provider_check):
        run_label = f"🌐 Analyse {asset_a} vs {asset_b} — Deep (with Web Search)"
    elif depth == "Deep":
        run_label = f"🚀 Analyse {asset_a} vs {asset_b} — Deep"
    else:
        run_label = f"🚀 Analyse {asset_a} vs {asset_b} — Fast"
    if st.button(run_label, type="primary", use_container_width=True):
        provider = st.session_state.get("llm_provider", "OpenAI")
        model = st.session_state.get("llm_model", "gpt-4.1-mini")
        custom_base_url = st.session_state.get("llm_custom_base_url", "")
        include_deep_news = depth == "Deep"
        use_web = include_deep_news and _is_openai_provider(provider)
        total_steps = 2 if use_web else (3 if include_deep_news else 1)
        run_started = time.perf_counter()
        try:
            with st.status(f"🚀 Running {depth} AI analysis…", expanded=True) as status_box:
                status_box.write(f"**Provider:** `{provider}` · **Model:** `{model}` · **Depth:** {depth}")

                # Step 1 — Brief
                status_box.write(f"Step 1/{total_steps}: Generating analysis brief…")
                t0 = time.perf_counter()
                heartbeat = st.empty()
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(generate_ai_brief, evidence, provider, model, api_key, depth, custom_base_url)
                    while not future.done():
                        elapsed = time.perf_counter() - t0
                        if elapsed < 15:
                            heartbeat.info(f"Step 1/{total_steps} in progress… {elapsed:.0f}s")
                        else:
                            heartbeat.warning(f"Step 1/{total_steps} still running… {elapsed:.0f}s — provider may be slow")
                        time.sleep(1.0)
                    brief = future.result()
                heartbeat.empty()
                st.session_state["last_ai_brief"] = brief
                status_box.write(f"✅ Step 1 done in {time.perf_counter()-t0:.1f}s")

                web_research = None
                analysis_a = None
                analysis_b = None
                if include_deep_news:
                    use_web_search = _is_openai_provider(provider)
                    if use_web_search:
                        status_box.write(f"Step 2/{total_steps}: 🌐 Web-powered deep research for `{asset_a}` & `{asset_b}`…")
                        t1 = time.perf_counter()
                        hb2 = st.empty()
                        with ThreadPoolExecutor(max_workers=1) as executor:
                            future = executor.submit(web_search_deep_analysis, asset_a, asset_b, evidence, api_key, model)
                            while not future.done():
                                elapsed = time.perf_counter() - t1
                                if elapsed < 20:
                                    hb2.info(f"Step 2/{total_steps}: Searching the web… {elapsed:.0f}s")
                                else:
                                    hb2.warning(f"Step 2/{total_steps}: Still searching… {elapsed:.0f}s (web search can take 15-30s)")
                                time.sleep(1.0)
                            web_research = future.result()
                        hb2.empty()
                        status_box.write(f"✅ Step 2 done in {time.perf_counter()-t1:.1f}s")
                        total_steps = 2
                    else:
                        # Non-OpenAI: fall back to per-asset chat-based analysis
                        status_box.write(f"Step 2/{total_steps}: News analysis for `{asset_a}`…")
                        t1 = time.perf_counter()
                        hb2 = st.empty()
                        with ThreadPoolExecutor(max_workers=1) as executor:
                            future = executor.submit(analyze_news_with_ai, news_a, asset_a, evidence, api_key, provider, model, depth, custom_base_url)
                            while not future.done():
                                hb2.info(f"Step 2/{total_steps}… {time.perf_counter()-t1:.0f}s")
                                time.sleep(1.0)
                            analysis_a = future.result()
                        hb2.empty()
                        status_box.write(f"✅ Step 2 done in {time.perf_counter()-t1:.1f}s")

                        status_box.write(f"Step 3/{total_steps}: News analysis for `{asset_b}`…")
                        t2 = time.perf_counter()
                        hb3 = st.empty()
                        with ThreadPoolExecutor(max_workers=1) as executor:
                            future = executor.submit(analyze_news_with_ai, news_b, asset_b, evidence, api_key, provider, model, depth, custom_base_url)
                            while not future.done():
                                hb3.info(f"Step 3/{total_steps}… {time.perf_counter()-t2:.0f}s")
                                time.sleep(1.0)
                            analysis_b = future.result()
                        hb3.empty()
                        status_box.write(f"✅ Step 3 done in {time.perf_counter()-t2:.1f}s")
                else:
                    status_box.write("ℹ️ News drill-down skipped (Fast mode).")

                st.session_state["last_news_analysis"] = {
                    "asset_a": asset_a, "asset_b": asset_b,
                    "web_research": web_research,
                    "analysis_a": analysis_a, "analysis_b": analysis_b,
                    "source": "web_search" if (include_deep_news and _is_openai_provider(provider)) else "chat",
                    "depth": depth,
                }
                total_s = time.perf_counter() - run_started
                status_box.update(label=f"✅ Analysis complete in {total_s:.1f}s", state="complete")
                st.session_state["last_ai_run_meta"] = {"provider": provider, "model": model, "depth": depth, "steps": total_steps, "duration_s": round(total_s, 2), "finished_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        except Exception as e:
            st.session_state["last_ai_run_meta"] = {"provider": st.session_state.get("llm_provider"), "model": st.session_state.get("llm_model"), "depth": depth, "finished_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "error": str(e)}
            st.error(f"Analysis failed: {e}")

    # Run metadata
    run_meta = st.session_state.get("last_ai_run_meta")
    if run_meta:
        if run_meta.get("error"):
            st.caption(f"Last run failed at {run_meta.get('finished_at')} ({run_meta.get('provider')} / {run_meta.get('model')}): {run_meta.get('error')}")
        else:
            st.caption(f"Last run: {run_meta.get('finished_at')} · {run_meta.get('provider')} / {run_meta.get('model')} · {run_meta.get('depth')} · {run_meta.get('duration_s')}s")

    # ---- Display results ----
    brief = st.session_state.get("last_ai_brief")
    if not brief:
        return

    st.divider()

    # --- 1. AI Summary (compact) ---
    regime = brief.get("regime_summary", {})
    regime_label = regime.get("label", "unclear")
    regime_sentence = regime.get("one_sentence", "")
    regime_colors = {"calm": "#4ade80", "mixed": "#fbbf24", "stress": "#f87171", "unclear": "#94a3b8"}
    regime_color = regime_colors.get(regime_label, "#94a3b8")
    interpretation = brief.get("relationship_now", {}).get("interpretation", "")

    st.markdown(f"""<div class="ai-card">
<div style="display:flex; align-items:center; gap:0.75rem; margin-bottom:0.5rem;">
    <span style="font-size:1.4rem;">🧠</span>
    <span style="font-size:1.1rem; font-weight:700; color:#f8fafc;">{asset_a} & {asset_b}</span>
    <span style="margin-left:auto; background:{regime_color}22; border:1px solid {regime_color}; color:{regime_color}; padding:0.2rem 0.75rem; border-radius:20px; font-size:0.78rem; font-weight:600;">{regime_label.upper()}</span>
</div>
<p style="color:#e2e8f0; font-size:0.95rem; line-height:1.55; margin:0 0 0.4rem 0;">{regime_sentence}</p>
{"<p style='color:#94a3b8; font-size:0.88rem; line-height:1.5; margin:0;'>" + interpretation + "</p>" if interpretation else ""}
</div>""", unsafe_allow_html=True)

    # --- 2. Key Metrics Row ---
    rel = brief.get("relationship_now", {})
    reliability = brief.get("reliability", {})
    lead_lag = brief.get("lead_lag", {})

    corr_val = rel.get("corr_now", 0)
    corr_chg = rel.get("corr_change_vs_4w", 0)
    confidence = reliability.get("overall_confidence", "medium")
    leader = lead_lag.get("likely_leader", "unclear")
    edge = rel.get("edge_status", "unknown").replace("_", " ").title()

    conf_colors = {"high": "#4ade80", "medium": "#fbbf24", "low": "#f87171"}
    conf_color = conf_colors.get(confidence, "#94a3b8")

    cols = st.columns(5)
    metric_items = [
        ("Correlation", f"{corr_val:+.3f}" if corr_val else "N/A", "#60a5fa"),
        ("4w Change", f"{corr_chg:+.3f}" if corr_chg else "N/A", "#4ade80" if (corr_chg and corr_chg > 0) else "#f87171"),
        ("Edge", edge, "#4ade80" if edge == "Connected" else "#f87171"),
        ("Confidence", confidence.upper(), conf_color),
        ("Leader", leader.upper(), "#a78bfa"),
    ]
    for col, (label, value, color) in zip(cols, metric_items):
        with col:
            st.markdown(f'<div class="metric-card"><div style="font-size:1.6rem;font-weight:700;color:{color};">{value}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

    # --- 3. Charts ---
    c1, c2 = st.columns([1.6, 1])
    with c1:
        hist = get_correlation_series(md, asset_a, asset_b, lookback)
        if not hist.empty:
            fig = create_correlation_chart(hist, threshold, asset_a, asset_b, lookback)
            st.plotly_chart(fig, use_container_width=True, key="ai_chart")
    with c2:
        if corr_val is not None:
            fig_g = create_gauge(corr_val, "Current ρ")
            st.plotly_chart(fig_g, use_container_width=True, key="ai_gauge")

    # --- 4. Detailed Sections (tabs) ---
    tabs = st.tabs(["🔄 Lead-Lag", "📰 News & Sentiment", "✅ Implications & Caveats"])

    with tabs[0]:
        ll = brief.get("lead_lag", {})
        st.markdown(f"**Summary:** {ll.get('summary','N/A')}")
        leader_lbl = ll.get("likely_leader", "unclear")
        leader_map = {"A": asset_a, "B": asset_b, "none": "Neither", "unclear": "Unclear"}
        st.markdown(f"**Likely leader:** {leader_map.get(leader_lbl, leader_lbl)}")
        for ev in ll.get("supporting_evidence", []):
            st.markdown(f"- {ev}")
        # Lead-lag table from evidence data
        lag = evidence.get("lead_lag", {})
        if lag.get("a_leads_b"):
            lag_df = pd.DataFrame({
                "Lag (days)": [x["lag_days"] for x in lag["a_leads_b"]],
                f"{asset_a}→{asset_b}": [f"{x['corr']:+.3f}" if x['corr'] else "—" for x in lag["a_leads_b"]],
                f"{asset_b}→{asset_a}": [f"{x['corr']:+.3f}" if x['corr'] else "—" for x in lag["b_leads_a"]],
            })
            st.dataframe(lag_df, use_container_width=True, hide_index=True)

    with tabs[1]:
        last_na = st.session_state.get("last_news_analysis")
        has_web_research = (
            last_na
            and last_na.get("source") == "web_search"
            and last_na.get("web_research")
            and not last_na.get("web_research", {}).get("error")
            and last_na.get("asset_a") == asset_a
            and last_na.get("asset_b") == asset_b
        )

        if has_web_research:
            wr = last_na["web_research"]

            if wr.get("raw_analysis") and not wr.get("asset_a"):
                st.markdown("**🌐 AI Web Research**")
                st.markdown(f"""<div class="ai-card"><p style="color:#e2e8f0; line-height:1.7;">{wr['raw_analysis'][:2000]}</p></div>""", unsafe_allow_html=True)
            else:
                # --- Per-asset web research ---
                c_a, c_b = st.columns(2)
                for col, key, ticker_label in [(c_a, "asset_a", asset_a), (c_b, "asset_b", asset_b)]:
                    with col:
                        asset_data = wr.get(key, {})
                        sentiment = asset_data.get("sentiment", "neutral")
                        score = asset_data.get("sentiment_score", 0)
                        s_color = "#4ade80" if sentiment == "positive" else "#f87171" if sentiment == "negative" else "#94a3b8"

                        st.markdown(f"""<div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.5rem;">
<span style="font-weight:700; color:#f8fafc; font-size:1.05rem;">{ticker_label}</span>
<span style="background:{s_color}22; border:1px solid {s_color}; color:{s_color}; padding:0.15rem 0.6rem; border-radius:16px; font-size:0.75rem; font-weight:600;">{sentiment.upper()} ({score:+d})</span>
</div>""", unsafe_allow_html=True)

                        for headline in asset_data.get("recent_headlines", [])[:4]:
                            st.markdown(f"""<div class="news-card"><span style="color:#f8fafc; font-weight:500; font-size:0.88rem;">📰 {headline}</span></div>""", unsafe_allow_html=True)

                        context = asset_data.get("recent_context", "")
                        if context:
                            st.markdown(f"""<div style="background:rgba(30,41,59,0.4); border-radius:10px; padding:0.75rem; margin-top:0.4rem;">
<span style="color:#94a3b8; font-size:0.78rem; text-transform:uppercase; letter-spacing:0.04em;">Analysis</span><br>
<span style="color:#e2e8f0; font-size:0.88rem; line-height:1.6;">{context}</span>
</div>""", unsafe_allow_html=True)

                        drivers = asset_data.get("key_drivers", [])
                        if drivers:
                            driver_html = " ".join([f'<span style="background:rgba(96,165,250,0.15); border:1px solid rgba(96,165,250,0.3); color:#60a5fa; padding:0.2rem 0.5rem; border-radius:8px; font-size:0.75rem; margin-right:0.3rem;">{d}</span>' for d in drivers])
                            st.markdown(f'<div style="margin-top:0.4rem;">{driver_html}</div>', unsafe_allow_html=True)

                # --- Correlation explanation ---
                corr_analysis = wr.get("correlation_analysis", {})
                if corr_analysis:
                    st.divider()
                    st.markdown("**🔗 Correlation Analysis (Web-Powered)**")
                    explanation = corr_analysis.get("explanation", "")
                    change_expl = corr_analysis.get("change_explanation", "")

                    if explanation:
                        st.markdown(f"""<div class="ai-card">
<span style="color:#a78bfa; font-weight:600; font-size:0.85rem; text-transform:uppercase; letter-spacing:0.04em;">Why these assets move together</span><br><br>
<span style="color:#e2e8f0; font-size:0.95rem; line-height:1.7;">{explanation}</span>
</div>""", unsafe_allow_html=True)

                    if change_expl:
                        st.markdown(f"""<div class="finding-card">
<span style="color:#60a5fa; font-weight:600;">Recent correlation shift</span><br>
<span style="color:#e2e8f0; font-size:0.9rem;">{change_expl}</span>
</div>""", unsafe_allow_html=True)

                    shared = corr_analysis.get("shared_drivers", [])
                    diverge = corr_analysis.get("divergence_risks", [])
                    if shared or diverge:
                        d_c1, d_c2 = st.columns(2)
                        with d_c1:
                            if shared:
                                st.markdown("**Shared Drivers**")
                                for s in shared:
                                    st.markdown(f"""<div class="implication-card"><span style="color:#e2e8f0;">🔗 {s}</span></div>""", unsafe_allow_html=True)
                        with d_c2:
                            if diverge:
                                st.markdown("**Divergence Risks**")
                                for d in diverge:
                                    st.markdown(f"""<div class="caveat-card"><span style="color:#e2e8f0;">⚡ {d}</span></div>""", unsafe_allow_html=True)

                    direction = corr_analysis.get("correlation_direction", "")
                    rationale = corr_analysis.get("direction_rationale", "")
                    if direction:
                        dir_colors = {"strengthening": "#4ade80", "weakening": "#f87171", "stable": "#60a5fa"}
                        dir_color = dir_colors.get(direction, "#94a3b8")
                        st.markdown(f"""<div style="margin-top:0.5rem; display:flex; align-items:center; gap:0.5rem;">
<span style="font-weight:600; color:#94a3b8;">Outlook:</span>
<span style="background:{dir_color}22; border:1px solid {dir_color}; color:{dir_color}; padding:0.2rem 0.7rem; border-radius:16px; font-size:0.8rem; font-weight:600;">{direction.upper()}</span>
<span style="color:#e2e8f0; font-size:0.88rem;">{rationale}</span>
</div>""", unsafe_allow_html=True)

                # --- Market context ---
                mkt = wr.get("market_context", {})
                if mkt:
                    st.divider()
                    st.markdown("**🌍 Market Context**")
                    macro = mkt.get("macro_environment", "")
                    risk = mkt.get("risk_sentiment", "mixed")
                    events = mkt.get("key_upcoming_events", [])
                    risk_colors = {"risk-on": "#4ade80", "risk-off": "#f87171", "mixed": "#fbbf24"}
                    r_color = risk_colors.get(risk, "#94a3b8")

                    if macro:
                        st.markdown(f"""<div class="finding-card">
<span style="color:#60a5fa; font-weight:600;">Macro Environment</span>
<span style="float:right; background:{r_color}22; border:1px solid {r_color}; color:{r_color}; padding:0.1rem 0.5rem; border-radius:12px; font-size:0.75rem; font-weight:600;">{risk.upper()}</span><br>
<span style="color:#e2e8f0; font-size:0.9rem;">{macro}</span>
</div>""", unsafe_allow_html=True)

                    if events:
                        st.markdown("**Upcoming Events to Watch**")
                        for ev in events:
                            st.markdown(f"""<div class="news-card"><span style="color:#fbbf24;">📅</span> <span style="color:#e2e8f0; font-size:0.88rem;">{ev}</span></div>""", unsafe_allow_html=True)
        else:
            # Fallback: show fetched news + per-asset AI analysis
            news_api_key = st.session_state.get("news_api_key", "")
            if not news_api_key:
                st.info("📰 **No NewsAPI key configured.** Add one in Settings to see real news headlines here. The Deep mode web search (OpenAI) does not require a NewsAPI key.")
            else:
                c_a, c_b = st.columns(2)
                with c_a:
                    st.markdown(f"**{asset_a}**")
                    if news_a:
                        for n in news_a[:3]:
                            st.markdown(f"""<div class="news-card">
<span style="color:#f8fafc; font-weight:500; font-size:0.9rem;">{n.get('title','')}</span><br>
<span style="color:#64748b; font-size:0.8rem;">{(n.get('description','') or '')[:120]}</span>
</div>""", unsafe_allow_html=True)
                    else:
                        st.caption("No recent articles found.")
                with c_b:
                    st.markdown(f"**{asset_b}**")
                    if news_b:
                        for n in news_b[:3]:
                            st.markdown(f"""<div class="news-card">
<span style="color:#f8fafc; font-weight:500; font-size:0.9rem;">{n.get('title','')}</span><br>
<span style="color:#64748b; font-size:0.8rem;">{(n.get('description','') or '')[:120]}</span>
</div>""", unsafe_allow_html=True)
                    else:
                        st.caption("No recent articles found.")

            if last_na and last_na.get("depth") == "Deep" and last_na.get("asset_a") == asset_a and last_na.get("asset_b") == asset_b:
                st.divider()
                st.markdown("**🤖 AI News Drill-Down (Deep mode)**")
                for label, analysis in [(asset_a, last_na.get("analysis_a")), (asset_b, last_na.get("analysis_b"))]:
                    if not analysis or analysis.get("error"):
                        st.warning(f"{label}: {analysis.get('error', 'No result') if analysis else 'No result'}")
                        continue
                    sentiment = analysis.get("sentiment", "neutral")
                    score = analysis.get("sentiment_score", 0)
                    s_color = "#4ade80" if sentiment == "positive" else "#f87171" if sentiment == "negative" else "#94a3b8"
                    themes = ", ".join(analysis.get("key_themes", []))
                    expl = analysis.get("correlation_explanation", "N/A")
                    st.markdown(f"""<div class="finding-card">
<span style="font-weight:600; color:#f8fafc;">{label}</span> — <span style="color:{s_color}; font-weight:600;">{sentiment.upper()} ({score:+d})</span><br>
<span style="color:#94a3b8;">Themes: {themes}</span><br>
<span style="color:#e2e8f0; font-size:0.9rem;">{expl}</span>
</div>""", unsafe_allow_html=True)
            elif depth == "Deep":
                st.info("Run Deep mode analysis to see AI research. OpenAI provider enables live web search.")

    with tabs[2]:
        impl_col, cav_col = st.columns(2)
        with impl_col:
            st.markdown("**Implications**")
            for imp in brief.get("practical_implications", []):
                st.markdown(f"""<div class="implication-card"><span style="color:#e2e8f0;">✅ {imp}</span></div>""", unsafe_allow_html=True)
        with cav_col:
            st.markdown("**Caveats**")
            for cav in brief.get("caveats", []):
                st.markdown(f"""<div class="caveat-card"><span style="color:#e2e8f0;">⚠️ {cav}</span></div>""", unsafe_allow_html=True)
        notes = brief.get("reliability", {}).get("notes", brief.get("reliability", {}).get("reliability_notes", []))
        if notes:
            with st.expander("Reliability details"):
                for n in notes:
                    st.markdown(f"- {n}")

    # ── DECISION VERDICT ─────────────────────────────────────────────────────
    st.divider()
    st.markdown("#### 🎯 Decision Summary")
    rel = brief.get("relationship_now", {})
    reliability = brief.get("reliability", {})
    confidence = reliability.get("overall_confidence", "medium")
    corr_val = rel.get("corr_now", 0) or 0
    edge = rel.get("edge_status", "not_connected")
    leader_raw = brief.get("lead_lag", {}).get("likely_leader", "unclear")
    leader_map = {"A": asset_a, "B": asset_b, "none": "Neither", "unclear": "Unclear"}
    leader_label = leader_map.get(leader_raw, leader_raw)

    # Determine color for overall verdict
    if confidence == "high" and edge == "connected":
        verdict_color = "#4ade80"
        verdict_icon = "✅"
        verdict_text = f"Strong, reliable co-movement detected. {asset_a} and {asset_b} move together ({corr_val:+.2f}). Useful for hedging or diversification decisions."
    elif confidence == "low" or abs(corr_val) < 0.3:
        verdict_color = "#f87171"
        verdict_icon = "⚠️"
        verdict_text = f"Weak or unreliable signal. Do not rely on this correlation for positioning decisions — the relationship may be spurious."
    else:
        verdict_color = "#fbbf24"
        verdict_icon = "🔶"
        verdict_text = f"Moderate co-movement with medium confidence. Monitor over the next few weeks before acting on this signal."

    st.markdown(f"""<div style="background:{verdict_color}12; border:1px solid {verdict_color}44; border-radius:14px; padding:1.1rem 1.4rem;">
<div style="font-size:1.05rem; font-weight:700; color:{verdict_color}; margin-bottom:0.4rem;">{verdict_icon} Bottom Line</div>
<div style="color:#e2e8f0; font-size:0.92rem; line-height:1.6;">{verdict_text}</div>
<div style="margin-top:0.6rem; display:flex; gap:1.5rem; font-size:0.82rem; color:#94a3b8;">
  <span>Leader: <b style="color:#a78bfa;">{leader_label}</b></span>
  <span>Confidence: <b style="color:{verdict_color};">{confidence.upper()}</b></span>
  <span>ρ: <b style="color:#60a5fa;">{corr_val:+.3f}</b></span>
</div>
</div>""", unsafe_allow_html=True)

    # Save to analysis history
    run_meta = st.session_state.get("last_ai_run_meta", {})
    if run_meta and not run_meta.get("error"):
        regime = brief.get("regime_summary", {}).get("label", "unclear")
        history_entry = {
            "Time": run_meta.get("finished_at", ""),
            "Asset A": asset_a,
            "Asset B": asset_b,
            "ρ": f"{corr_val:+.3f}",
            "Regime": regime.upper(),
            "Confidence": confidence.upper(),
            "Leader": leader_label,
            "Window": f"{lookback}d",
            "Notes": "",
        }
        # Avoid duplicates
        existing = st.session_state.get("analysis_history", [])
        last = existing[-1] if existing else {}
        if not (last.get("Asset A") == asset_a and last.get("Asset B") == asset_b and last.get("Time") == history_entry["Time"]):
            existing.append(history_entry)
            st.session_state["analysis_history"] = existing[-20:]  # keep last 20

    # ── ANALYSIS HISTORY (st.data_editor — novel widget) ─────────────────────
    history = st.session_state.get("analysis_history", [])
    if history:
        st.divider()
        st.markdown("#### 📋 Analysis History")
        st.caption("Your past analyses this session. Add notes directly in the table — click any cell in the Notes column.")
        history_df = pd.DataFrame(history)
        edited_df = st.data_editor(
            history_df,
            use_container_width=True,
            hide_index=True,
            disabled=["Time", "Asset A", "Asset B", "ρ", "Regime", "Confidence", "Leader", "Window"],
            column_config={
                "Notes": st.column_config.TextColumn("Notes", help="Add your own notes about this analysis", max_chars=200),
                "ρ": st.column_config.TextColumn("ρ", width="small"),
                "Regime": st.column_config.TextColumn("Regime", width="small"),
                "Confidence": st.column_config.TextColumn("Confidence", width="small"),
                "Leader": st.column_config.TextColumn("Leader", width="medium"),
                "Window": st.column_config.TextColumn("Window", width="small"),
            },
            key="history_editor",
        )
        # Persist edits back
        st.session_state["analysis_history"] = edited_df.to_dict("records")
        if st.button("🗑️ Clear History", key="clear_history"):
            st.session_state["analysis_history"] = []
            st.rerun()

# =============================================================================
# PAGE: HOME — Contagion Map (hero) + inline pair drill-down
# =============================================================================
def render_home(md, months):
    # ── HERO BANNER ───────────────────────────────────────────────────────────
    st.markdown("""
<div style="background:linear-gradient(135deg,rgba(59,130,246,0.13),rgba(139,92,246,0.13));
     border:1px solid rgba(139,92,246,0.30); border-radius:18px; padding:1.4rem 1.75rem; margin-bottom:1.25rem;">
  <div style="font-size:1.35rem; font-weight:800; color:#f8fafc; margin-bottom:0.3rem;">
    🌊 Contagion Map
  </div>
  <div style="color:#cbd5e1; font-size:0.92rem; line-height:1.6; max-width:780px;">
    When markets stress, correlations spike — and diversification disappears exactly when you need it most.
    This map shows <b>which assets in your universe are linked right now</b>, how dense those links are,
    and which node would spread the most damage if it fell.
    Spot a pair? Select it below to drill in — the AI will explain <em>why</em> they're correlated.
  </div>
</div>""", unsafe_allow_html=True)

    if not md.tickers_ok or not md.weekly_anchors:
        st.warning("No data available. Enable Demo Mode in the sidebar or check your tickers.")
        return

    # ── READ SETTINGS — stored in session state, sliders at bottom update them
    # Use dedicated state keys that don't conflict with slider widget keys
    if "map_lookback" not in st.session_state:
        st.session_state["map_lookback"] = 120
    if "map_threshold" not in st.session_state:
        st.session_state["map_threshold"] = 0.40
    lookback = st.session_state["map_lookback"]
    threshold = st.session_state["map_threshold"]
    st.session_state["global_lookback"] = lookback
    st.session_state["global_threshold"] = threshold

    # ── COMPUTE UNIVERSE STATS ────────────────────────────────────────────────
    stats_now = compute_window_stats(md.returns[md.tickers_ok], md.weekly_anchors[-1], lookback)
    corr = stats_now.corr

    n = len(md.tickers_ok)
    max_edges = n * (n - 1) // 2
    edge_count = sum(
        1 for i in range(n) for j in range(i + 1, n)
        if not pd.isna(corr.iloc[i, j]) and abs(corr.iloc[i, j]) >= threshold
    )
    contagion_pct = edge_count / max_edges if max_edges > 0 else 0
    avg_abs = np.nanmean([abs(corr.iloc[i, j]) for i in range(n) for j in range(i+1, n)]) if n > 1 else 0

    degree = {t: sum(1 for other in md.tickers_ok if other != t
                     and t in corr.index and other in corr.columns
                     and not pd.isna(corr.loc[t, other])
                     and abs(corr.loc[t, other]) >= threshold)
              for t in md.tickers_ok}
    top_hub = max(degree, key=degree.get) if degree else "—"
    top_hub_links = degree.get(top_hub, 0)

    if contagion_pct >= 0.60:
        rc, ri, rl, rm = "#f87171", "🔴", "HIGH CONTAGION", f"{edge_count} of {max_edges} pairs strongly linked. Diversification is largely ineffective — one macro shock can ripple across almost everything."
    elif contagion_pct >= 0.30:
        rc, ri, rl, rm = "#fbbf24", "🟡", "MODERATE CONTAGION", f"{edge_count} of {max_edges} pairs linked. Clusters are forming — check which hub would spread the most damage."
    else:
        rc, ri, rl, rm = "#4ade80", "🟢", "LOW CONTAGION", f"Only {edge_count} of {max_edges} pairs linked above threshold. Assets are behaving independently — diversification is working."

    # ── GRAPHS FIRST — Network + Heatmap as the visual hero ──────────────────
    net_col, heat_col = st.columns([1.3, 1])
    with net_col:
        st.markdown("**Contagion network — who is linked to whom?**")
        st.caption("Each line is a correlation link above your threshold. Thicker = stronger. Red = negative (natural hedge).")
        fig_net = create_network_viz(corr, md.tickers_ok, threshold)
        st.plotly_chart(fig_net, use_container_width=True, key="home_network")
        st.markdown("""<div style="display:flex; gap:1.5rem; color:#94a3b8; font-size:0.78rem; margin-top:-0.25rem;">
<span><span style="color:#60a5fa;">━</span> Positive (move together)</span>
<span><span style="color:#f87171;">━</span> Negative (natural hedge)</span>
<span>Line thickness = |ρ|</span></div>""", unsafe_allow_html=True)
        if top_hub_links > 0:
            st.markdown(f"""<div style="background:rgba(248,113,113,0.07); border-left:3px solid #f87171;
border-radius:0 10px 10px 0; padding:0.55rem 1rem; margin-top:0.75rem; font-size:0.85rem; color:#e2e8f0;">
⚠️ <b>{top_hub}</b> is the biggest hub with {top_hub_links} active links — if it shocks, those are contagion channels.
</div>""", unsafe_allow_html=True)
    with heat_col:
        st.markdown("**Correlation heatmap — full picture**")
        st.caption("Dark green clusters = assets that behave as one position. Red = hedges. Sorted by similarity.")
        order = corr_cluster_order(corr)
        fig_heat = create_heatmap(corr, order)
        st.plotly_chart(fig_heat, use_container_width=True, key="home_heatmap")

    # ── QUICK METRICS (compact, no redundant banner) ─────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    for col, (label, val, sub, color) in zip(
        [m1, m2, m3, m4],
        [
            (f"{ri} Contagion", f"{contagion_pct:.0%}", f"{edge_count}/{max_edges} pairs linked", rc),
            ("Avg |ρ| Universe", f"{avg_abs:.2f}", "across all pairs", "#a78bfa"),
            ("Biggest Hub", top_hub, f"{top_hub_links} connections", "#f87171" if top_hub_links >= 4 else "#fbbf24"),
            ("Lookback", f"{lookback}d", "rolling window", "#60a5fa"),
        ]
    ):
        with col:
            st.markdown(f'<div class="metric-card"><div style="font-size:1.5rem;font-weight:700;color:{color};">{val}</div><div class="metric-label">{label}</div><div style="color:#64748b;font-size:0.72rem;margin-top:0.15rem;">{sub}</div></div>', unsafe_allow_html=True)

    # ── MAP SETTINGS (collapsed by default — clean landing) ──────────────────
    with st.expander("⚙️ Map settings", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            new_lookback = st.select_slider("Lookback window (days)", [20, 60, 120],
                                            value=st.session_state["map_lookback"],
                                            key="map_lookback_slider")
        with c2:
            new_threshold = st.slider("Contagion threshold |ρ|", 0.20, 0.90,
                                      st.session_state["map_threshold"], 0.05,
                                      key="map_threshold_slider",
                                      help="Pairs above this are drawn as links in the network")
        if new_lookback != st.session_state["map_lookback"] or new_threshold != st.session_state["map_threshold"]:
            st.session_state["map_lookback"] = new_lookback
            st.session_state["map_threshold"] = new_threshold
            st.rerun()

    # ── INLINE PAIR DRILL-DOWN ────────────────────────────────────────────────
    st.divider()
    st.markdown("#### 🔍 Drill into a pair — what's driving this connection?")
    st.caption("Select any two assets. The AI will explain what each company does and why they're correlated.")

    p1, p2 = st.columns(2)
    with p1:
        pair_a = st.selectbox("Asset A", md.tickers_ok, key="home_pair_a")
    with p2:
        b_opts = [t for t in md.tickers_ok if t != pair_a]
        if st.session_state.get("home_pair_b") == pair_a and b_opts:
            st.session_state["home_pair_b"] = b_opts[0]
        pair_b = st.selectbox("Asset B", b_opts, key="home_pair_b")

    evidence = build_evidence_json(md, pair_a, pair_b, lookback, threshold)
    rel = evidence["relationship"]
    stab = evidence["stability"]
    corr_now = rel["corr_now"]
    corr_delta = rel["corr_change_vs_4w"]
    edge_status = rel["edge_status"]

    delta_color = "#4ade80" if (corr_delta or 0) > 0.05 else "#f87171" if (corr_delta or 0) < -0.05 else "#94a3b8"
    edge_color = "#4ade80" if edge_status == "connected" else "#f87171"

    pm1, pm2, pm3, pm4 = st.columns(4)
    for col, (label, val, sub, color) in zip(
        [pm1, pm2, pm3, pm4],
        [
            ("ρ Right Now", f"{corr_now:+.3f}" if corr_now is not None else "N/A", "current co-movement", "#60a5fa"),
            ("4-Week Shift", f"{corr_delta:+.3f}" if corr_delta is not None else "N/A", "strengthening or fading?", delta_color),
            ("Signal", (edge_status or "—").replace("_", " ").title(), f"|ρ| ≥ {threshold}", edge_color),
            ("Sample", f"{stab['effective_sample_size']}d", "effective observations", "#94a3b8"),
        ]
    ):
        with col:
            st.markdown(f'<div class="metric-card"><div style="font-size:1.4rem;font-weight:700;color:{color};">{val}</div><div class="metric-label">{label}</div><div style="color:#64748b;font-size:0.72rem;margin-top:0.15rem;">{sub}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── AI EXPLANATION of the pair (inline, not a separate tab) ──────────────
    api_key = st.session_state.get("llm_api_key", "")
    expl_cache_key = f"pair_expl_{pair_a}_{pair_b}"

    if api_key:
        # Auto-generate if pair changed or not yet generated
        if st.session_state.get(expl_cache_key + "_pair") != (pair_a, pair_b):
            st.session_state.pop(expl_cache_key, None)

        if expl_cache_key not in st.session_state:
            with st.spinner(f"🤖 AI is explaining why {pair_a} and {pair_b} are correlated…"):
                result = generate_pair_explanation(
                    pair_a, pair_b, corr_now,
                    provider=st.session_state.get("llm_provider", "OpenAI"),
                    model=st.session_state.get("llm_model", "gpt-4.1-mini"),
                    api_key=api_key,
                    custom_base_url=st.session_state.get("llm_custom_base_url", ""),
                )
                st.session_state[expl_cache_key] = result
                st.session_state[expl_cache_key + "_pair"] = (pair_a, pair_b)

        expl = st.session_state.get(expl_cache_key, {})

        if expl and not expl.get("error"):
            conf = expl.get("confidence", "medium")
            conf_color = {"high": "#4ade80", "medium": "#fbbf24", "low": "#f87171"}.get(conf, "#94a3b8")
            desc_a = expl.get("asset_a_description", "")
            desc_b = expl.get("asset_b_description", "")
            why = expl.get("correlation_explanation", "")

            st.markdown(f"""
<div style="background:rgba(139,92,246,0.08); border:1px solid rgba(139,92,246,0.25); border-radius:14px; padding:1.1rem 1.4rem; margin-bottom:0.75rem;">
  <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.75rem;">
    <span style="font-size:1.1rem;">🤖</span>
    <span style="font-weight:700; color:#f8fafc; font-size:0.95rem;">AI: Why are {pair_a} and {pair_b} correlated?</span>
    <span style="margin-left:auto; background:{conf_color}22; border:1px solid {conf_color}; color:{conf_color}; padding:0.15rem 0.6rem; border-radius:12px; font-size:0.72rem; font-weight:600;">{conf.upper()} CONFIDENCE</span>
  </div>
  <div style="display:flex; gap:1rem; margin-bottom:0.9rem; flex-wrap:wrap;">
    <div style="flex:1; min-width:200px; background:rgba(96,165,250,0.07); border-radius:10px; padding:0.65rem 0.9rem;">
      <div style="color:#60a5fa; font-weight:700; font-size:0.78rem; text-transform:uppercase; letter-spacing:0.04em; margin-bottom:0.25rem;">{pair_a}</div>
      <div style="color:#e2e8f0; font-size:0.87rem; line-height:1.55;">{desc_a}</div>
    </div>
    <div style="flex:1; min-width:200px; background:rgba(167,139,250,0.07); border-radius:10px; padding:0.65rem 0.9rem;">
      <div style="color:#a78bfa; font-weight:700; font-size:0.78rem; text-transform:uppercase; letter-spacing:0.04em; margin-bottom:0.25rem;">{pair_b}</div>
      <div style="color:#e2e8f0; font-size:0.87rem; line-height:1.55;">{desc_b}</div>
    </div>
  </div>
  <div style="border-top:1px solid rgba(148,163,184,0.12); padding-top:0.75rem;">
    <div style="color:#94a3b8; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.04em; margin-bottom:0.3rem;">Why they move together (ρ = {corr_now:+.2f})</div>
    <div style="color:#e2e8f0; font-size:0.9rem; line-height:1.65;">{why}</div>
  </div>
</div>""", unsafe_allow_html=True)
        elif expl.get("error"):
            st.caption(f"AI explanation unavailable: {expl['error']}")
    else:
        st.markdown("""<div style="background:rgba(30,41,59,0.5); border:1px solid rgba(148,163,184,0.1); border-radius:12px; padding:0.85rem 1.1rem; font-size:0.87rem; color:#64748b;">
🤖 Add an API key in ⚙️ Settings to get an AI explanation of why this pair is correlated.
</div>""", unsafe_allow_html=True)

    # ── PRICE OVERLAY + CORRELATION side by side ─────────────────────────────
    ch1, ch2 = st.columns(2)
    with ch1:
        st.markdown(f"**{pair_a} vs {pair_b} — normalized price**")
        st.caption("Both indexed to 100 at start of window. How have they moved relative to each other?")
        if not md.prices.empty and pair_a in md.prices.columns and pair_b in md.prices.columns:
            fig_px = create_price_overlay(md.prices, [pair_a, pair_b], lookback_days=lookback)
            st.plotly_chart(fig_px, use_container_width=True, key="home_price_overlay")
    with ch2:
        st.markdown(f"**{pair_a} vs {pair_b} — rolling correlation (ρ)**")
        st.caption("Spikes toward ±1.0 often coincide with macro stress events.")
        hist = get_correlation_series(md, pair_a, pair_b, lookback)
        if not hist.empty:
            fig_c = create_correlation_chart(hist, threshold, pair_a, pair_b, lookback)
            st.plotly_chart(fig_c, use_container_width=True, key="home_pair_chart")

    # ── CTA to AI Insights ────────────────────────────────────────────────────
    st.markdown(f"""<div style="background:rgba(59,130,246,0.07); border:1px solid rgba(59,130,246,0.2);
border-radius:12px; padding:0.85rem 1.2rem; font-size:0.88rem; color:#cbd5e1; margin-top:0.5rem;">
💡 Want the full research brief on <b>{pair_a} vs {pair_b}</b> — lead-lag analysis, bootstrap confidence, regime classification, and a decision verdict?
</div>""", unsafe_allow_html=True)
    if st.button(f"→ Full AI Analysis: {pair_a} vs {pair_b}", type="primary"):
        st.session_state["ai_a"] = pair_a
        st.session_state["ai_b"] = pair_b
        st.session_state["current_page"] = "ai_insights"
        st.rerun()


# =============================================================================
# PAGE: SETTINGS
# =============================================================================
def render_settings():
    st.markdown("## ⚙️ Settings")
    st.caption("Configure your analysis environment")

    # ---- 1. AI Configuration ----
    st.markdown("#### 🤖 AI Provider & API Key")
    current_provider = st.session_state.get("llm_provider", "OpenAI")
    provider_idx = PROVIDER_NAMES.index(current_provider) if current_provider in PROVIDER_NAMES else 0

    col_p, col_m = st.columns(2)
    with col_p:
        provider = st.selectbox("LLM Provider", options=PROVIDER_NAMES, index=provider_idx, key="settings_provider")
        st.session_state["llm_provider"] = provider
    with col_m:
        default_model = PROVIDER_REGISTRY.get(provider, {}).get("default_model", "gpt-4.1-mini")
        if st.session_state.get("_last_provider") != provider:
            st.session_state["llm_model"] = default_model
            st.session_state["_last_provider"] = provider
        st.session_state["llm_model"] = st.text_input(
            "Model", value=st.session_state.get("llm_model", default_model),
            help=f"Default: {default_model}" if default_model else None,
        )

    if provider == "Custom (OpenAI-compatible)":
        st.session_state["llm_custom_base_url"] = st.text_input(
            "Custom Base URL", value=st.session_state.get("llm_custom_base_url", ""),
            placeholder="https://your-api.example.com/v1",
        )

    st.session_state["llm_api_key"] = st.text_input(
        f"{provider} API Key", value=st.session_state["llm_api_key"], type="password",
        help="Required for AI insights. Stored only in session.",
    )

    # Endpoint info
    base_url = PROVIDER_REGISTRY.get(provider, {}).get("base_url", "")
    if base_url and provider != "Custom (OpenAI-compatible)":
        st.caption(f"Endpoint: `{base_url}`")

    # Test button
    custom_base_url = st.session_state.get("llm_custom_base_url", "")
    if st.button("🧪 Test API Key", use_container_width=False):
        with st.spinner(f"Testing {provider} / {st.session_state.get('llm_model', '')}…"):
            st.session_state["llm_last_key_check"] = verify_llm_api_key(
                provider=provider,
                model=st.session_state.get("llm_model", ""),
                api_key=st.session_state.get("llm_api_key", ""),
                custom_base_url=custom_base_url,
            )

    check = st.session_state.get("llm_last_key_check")
    if check:
        current_fp = _api_key_fingerprint(st.session_state.get("llm_api_key", ""))
        is_current = (
            check.get("provider") == provider
            and check.get("model") == st.session_state.get("llm_model", "")
            and check.get("key_fp") == current_fp
        )
        suffix = "" if is_current else "  *(different provider/model/key)*"
        if check.get("ok"):
            token_msg = f" · tokens: {check.get('token_usage')}" if check.get("token_usage") is not None else ""
            st.success(f"✅ Passed at {check.get('checked_at')} — {check.get('provider')} / {check.get('model')} ({check.get('latency_ms')}ms{token_msg}){suffix}")
        else:
            st.error(f"❌ Failed at {check.get('checked_at')} — {check.get('provider')} / {check.get('model')}{suffix}")
            st.caption(f"Error: `{check.get('message')}`")

    st.divider()

    # ---- 2. Data Source ----
    st.markdown("#### 📡 Data Source")
    col_ds, col_demo = st.columns(2)
    with col_ds:
        provider_options = ["Yahoo Finance", "Stooq (backup)"]
        current_live = st.session_state.get("live_provider", "yahoo")
        ds_idx = 0 if current_live == "yahoo" else 1
        ds_choice = st.selectbox("Market Data Provider", provider_options, index=ds_idx, help="Stooq is a free backup if Yahoo is blocked on your network.")
        st.session_state["live_provider"] = "yahoo" if ds_choice.startswith("Yahoo") else "stooq"
    with col_demo:
        st.session_state["use_sample_data"] = st.toggle("Demo Mode (synthetic data)", value=st.session_state["use_sample_data"])

    st.divider()

    # ---- 3. Universe Management ----
    st.markdown("#### 📈 Ticker Universe")
    st.info("💡 Manage your active tickers in the **sidebar** — use the multiselect to choose which assets to analyse, or add custom ones by name/ticker.", icon="ℹ️")

    st.divider()

    # ---- 4. News API ----
    st.markdown("#### 📰 News API")
    st.session_state["news_api_key"] = st.text_input(
        "NewsAPI Key (optional)", value=st.session_state["news_api_key"], type="password",
        help="For real news. Leave empty for demo headlines.",
    )

    st.divider()

    # ---- 5. About ----
    with st.expander("ℹ️ About Contagion Explorer"):
        st.markdown("""
**Methodology:**
- Weekly snapshots using last trading day of each week
- Rolling correlations on 20 / 60 / 120 day windows
- Hierarchical clustering for heatmap ordering
- Bootstrap confidence intervals for reliability

**⚠️ Correlations show association, not causation.**
""")

# =============================================================================
# MAIN
# =============================================================================
def main():
    months = render_sidebar()
    page = st.session_state.get("current_page", "home")

    selected = st.session_state.get("selected_tickers", DEFAULT_SELECTED)
    if len(selected) < 2:
        st.warning("Select at least 2 assets in the sidebar to continue.")
        return
    all_tickers = list(dict.fromkeys(selected))[:20]

    with st.spinner("Loading market data…"):
        if st.session_state["use_sample_data"]:
            md = generate_sample_market_data(all_tickers, months=months)
        else:
            md = load_market_data(
                all_tickers, months=months,
                provider=st.session_state.get("live_provider", "yahoo"),
                yahoo_refresh_nonce=st.session_state["yahoo_refresh_nonce"],
            )

    if st.session_state["use_sample_data"]:
        st.info("📊 Demo mode — using synthetic data")
    if md.tickers_failed:
        st.warning(f"⚠️ Failed to load: {', '.join(md.tickers_failed)}")
    if not md.tickers_ok:
        st.error("❌ No data available. Try demo mode or check tickers.")
        return

    if page == "home":
        render_home(md, months)
    elif page == "ai_insights":
        render_ai_insights(md, months)
    elif page == "settings":
        render_settings()

if __name__ == "__main__":
    main()