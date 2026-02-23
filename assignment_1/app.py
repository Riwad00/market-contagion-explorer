"""
Contagion Explorer
------------------
A prototype for portfolio managers evaluating whether a new asset
will add real diversification — or just more of the same risk.
"""

from __future__ import annotations

import io
import os
import re
import json
import hashlib
import math
import time
from types import SimpleNamespace
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

# =============================================================================
# CONFIG
# =============================================================================
st.set_page_config(
    page_title="Contagion Explorer",
    page_icon="🌊",
    layout="wide",
)

# =============================================================================
# CONSTANTS
# =============================================================================
DEFAULT_TICKERS = {
    "SPY":   "S&P 500 ETF",
    "QQQ":   "Nasdaq ETF",
    "IEF":   "7-10yr Treasury ETF",
    "GLD":   "Gold ETF",
    "HYG":   "High Yield Bond ETF",
    "AAPL":  "Apple",
    "MSFT":  "Microsoft",
    "NVDA":  "NVIDIA",
    "TSLA":  "Tesla",
    "AMZN":  "Amazon",
    "META":  "Meta",
    "GOOGL": "Alphabet",
    "IWM":   "Russell 2000 ETF",
    "BRK-B": "Berkshire Hathaway",
    "AVGO":  "Broadcom",
}

PROVIDER_REGISTRY = {
    "OpenAI":                     {"base_url": "https://api.openai.com/v1",                                "default_model": "gpt-4.1-mini"},
    "Google Gemini":              {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai/", "default_model": "gemini-2.0-flash"},
    "Kimi (Moonshot)":            {"base_url": "https://api.moonshot.cn/v1",                               "default_model": "moonshot-v1-8k"},
    "NVIDIA NIM":                 {"base_url": "https://integrate.api.nvidia.com/v1",                      "default_model": "moonshotai/kimi-k2.5"},
    "DeepSeek":                   {"base_url": "https://api.deepseek.com",                                 "default_model": "deepseek-chat"},
    "Groq":                       {"base_url": "https://api.groq.com/openai/v1",                           "default_model": "llama-3.3-70b-versatile"},
    "Together AI":                {"base_url": "https://api.together.xyz/v1",                              "default_model": "meta-llama/Llama-3.3-70B-Instruct-Turbo"},
    "Custom (OpenAI-compatible)": {"base_url": "",                                                         "default_model": ""},
}

OPENAI_MODELS = [
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-4o",
    "gpt-4o-mini",
]

SCENARIO_HINTS = {
    ("SPY", "QQQ"): "Classic: US large-cap equities vs tech-heavy index.",
    ("SPY", "IEF"): "Equity-bond — negative correlation is the 60/40 hedge, but breaks in inflationary regimes.",
    ("SPY", "GLD"): "Risk-on vs safe-haven. Gold tends to decouple during market stress.",
    ("HYG", "SPY"): "Credit-equity signal — HYG leading SPY down is an early-warning for equity drawdowns.",
    ("AAPL", "MSFT"): "Two mega-cap tech names — high correlation limits diversification benefit.",
}

# =============================================================================
# SESSION STATE
# =============================================================================
_DEFAULTS = {
    "llm_provider":       "OpenAI",
    "llm_model":          "gpt-4.1-mini",
    "llm_api_key":        os.getenv("OPENAI_API_KEY", ""),
    "llm_custom_base_url": "",
    "llm_last_key_check": None,
    "news_api_key":       os.getenv("NEWS_API_KEY", ""),
    "ai_analysis_depth":  "Deep",
    "use_demo":           False,
    "portfolio":          ["NVDA", "GOOGL", "MSFT", "AMZN", "META", "TSLA", "SPY", "QQQ"],
    "candidate":          "AAPL",
    "lookback":           60,
    "threshold":          0.35,
    "last_brief":         None,
    "last_news_analysis": None,
    "last_ai_run_meta":   None,
    "analysis_log":       [],
    "custom_tickers":     [],
    "last_added":         None,
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# =============================================================================
# DATA
# =============================================================================
@dataclass
class MarketData:
    tickers_ok:     list
    tickers_failed: list
    prices:         pd.DataFrame
    returns:        pd.DataFrame
    weekly_anchors: list


def _build_weekly_anchors(index):
    if len(index) == 0:
        return []
    s = pd.Series(index, index=index)
    return [pd.Timestamp(x) for x in s.groupby(s.index.to_period("W-FRI")).max().sort_values().tolist()]


@st.cache_data(show_spinner=False)
def _load_company_tickers():
    path = os.path.join(os.path.dirname(__file__), "src", "company_tickers.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}, {}

    name_to_ticker = {}
    ticker_to_name = {}
    for item in data.values():
        ticker = (item.get("ticker") or "").strip().upper()
        title  = (item.get("title") or "").strip()
        if not ticker:
            continue
        ticker_to_name[ticker] = title or ticker
        if title:
            norm = _normalize_company_name(title)
            if norm:
                name_to_ticker[norm] = ticker
    return name_to_ticker, ticker_to_name


def _normalize_company_name(text):
    if not text:
        return ""
    lowered = text.lower()
    lowered = re.sub(
        r"\b(incorporated|inc|corp|corporation|co|company|ltd|limited|plc|sa|ag|nv|oyj|holdings|holding|group)\b",
        " ",
        lowered,
    )
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def _resolve_one_token(token, name_to_ticker, ticker_to_name):
    cleaned = re.sub(r"[^A-Za-z0-9.\-]", "", token.upper())
    if cleaned in DEFAULT_TICKERS or cleaned in ticker_to_name:
        return cleaned
    norm = _normalize_company_name(token)
    return name_to_ticker.get(norm, "")


def _resolve_ticker_input(raw):
    if not raw:
        return []
    name_to_ticker, ticker_to_name = _load_company_tickers()
    raw = raw.strip()
    parts = [p.strip() for p in re.split(r"[;,]+", raw) if p.strip()]
    resolved = []
    if len(parts) == 1:
        full = parts[0]
        single = _resolve_one_token(full, name_to_ticker, ticker_to_name)
        if single:
            resolved.append(single)
        else:
            for token in full.split():
                r = _resolve_one_token(token, name_to_ticker, ticker_to_name)
                if r:
                    resolved.append(r)
    else:
        for part in parts:
            single = _resolve_one_token(part, name_to_ticker, ticker_to_name)
            if single:
                resolved.append(single)
            else:
                for token in part.split():
                    r = _resolve_one_token(token, name_to_ticker, ticker_to_name)
                    if r:
                        resolved.append(r)
    return list(dict.fromkeys(resolved))


def _ticker_label(ticker):
    _, ticker_to_name = _load_company_tickers()
    return DEFAULT_TICKERS.get(ticker, ticker_to_name.get(ticker, ticker))


@st.cache_data(show_spinner=False, ttl=300)
def _fetch_yahoo(tickers, period="1y"):
    frames = []
    for t in tickers:
        try:
            df = yf.download(t, period=period, interval="1d", threads=False, progress=False, auto_adjust=True, timeout=20)
            if df is None or df.empty:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                close = df["Close"]
                if isinstance(close, pd.Series):
                    close = close.to_frame(t)
            elif "Close" in df.columns:
                close = df[["Close"]].rename(columns={"Close": t})
            else:
                close = df.iloc[:, :1].rename(columns={df.columns[0]: t})
            frames.append(close)
        except Exception:
            continue
    return pd.concat(frames, axis=1).sort_index() if frames else pd.DataFrame()


_STOOQ_MAP = {
    "SPY": "spy.us", "QQQ": "qqq.us", "IWM": "iwm.us", "IEF": "ief.us",
    "HYG": "hyg.us", "GLD": "gld.us", "AAPL": "aapl.us", "MSFT": "msft.us",
    "TSLA": "tsla.us", "NVDA": "nvda.us", "AMZN": "amzn.us", "GOOGL": "googl.us",
    "META": "meta.us", "AVGO": "avgo.us", "BRK-B": "brk-b.us",
}


@st.cache_data(show_spinner=False, ttl=3600)
def _fetch_stooq(tickers, start, end):
    start_dt = pd.to_datetime(start).tz_localize(None)
    end_dt   = pd.to_datetime(end).tz_localize(None)
    series = {}
    for t in tickers:
        sym = _STOOQ_MAP.get(t, t.lower())
        try:
            r = requests.get(f"https://stooq.com/q/d/l/?s={sym}&i=d", timeout=20)
            if r.status_code != 200 or "Date" not in r.text[:60]:
                continue
            df = pd.read_csv(io.StringIO(r.text), parse_dates=["Date"])
            if df.empty or "Close" not in df.columns:
                continue
            df = df.rename(columns={"Date": "date"}).set_index("date").sort_index()
            df = df.loc[(df.index >= start_dt) & (df.index <= end_dt)]
            if not df.empty:
                series[t] = df["Close"].astype(float)
        except Exception:
            continue
    if not series:
        return pd.DataFrame()
    out = pd.concat(series, axis=1)
    out.index = pd.to_datetime(out.index)
    return out


def _generate_demo(tickers, months=12):
    idx = pd.bdate_range(date.today() - timedelta(days=int(months * 30.5)), date.today())
    rng = np.random.default_rng(42)
    n   = len(tickers)
    market = rng.normal(0, 1, len(idx))
    betas  = rng.uniform(0.3, 0.9, n)
    idio   = rng.normal(0, 1, (len(idx), n))
    vol    = rng.uniform(0.08, 0.45, n) / np.sqrt(252)
    r = (market[:, None] * betas + idio * (1 - betas))
    r = (r - r.mean(0)) / (r.std(0) + 1e-12) * vol
    px = np.exp(np.log(rng.uniform(50, 250, n)) + np.cumsum(r, 0))
    return pd.DataFrame(px, index=idx, columns=tickers)


def load_data(tickers, months=12, demo=False):
    if demo:
        prices = _generate_demo(tickers, months)
        ok, failed = tickers[:], []
    else:
        period  = "2y" if months >= 24 else "1y"
        prices  = _fetch_yahoo(tuple(tickers), period=period)
        if prices.empty:
            end_s   = (date.today() + timedelta(days=1)).isoformat()
            start_s = (date.today() - timedelta(days=int(months * 30.5))).isoformat()
            prices  = _fetch_stooq(tuple(tickers), start=start_s, end=end_s)
        ok     = [t for t in tickers if t in prices.columns and not prices[t].dropna().empty]
        failed = [t for t in tickers if t not in ok]
        prices = prices[ok] if ok else pd.DataFrame()

    if prices.empty:
        return MarketData([], tickers, pd.DataFrame(), pd.DataFrame(), [])

    returns = np.log(prices).diff()
    return MarketData(ok, failed, prices, returns, _build_weekly_anchors(prices.index))


# =============================================================================
# QUANT
# =============================================================================
def rolling_corr(returns, a, b, lookback):
    clean = returns[[a, b]].dropna().tail(int(lookback))
    if len(clean) < max(10, int(0.5 * lookback)):
        return None
    return float(clean[a].corr(clean[b]))


def corr_series(md, a, b, lookback):
    rows = []
    for anchor in md.weekly_anchors:
        w = md.returns[[a, b]].loc[:anchor].dropna().tail(int(lookback))
        if len(w) >= max(10, int(0.5 * lookback)):
            rows.append({"week": anchor, "corr": float(w[a].corr(w[b]))})
    return pd.DataFrame(rows)


def full_corr_matrix(returns, tickers, lookback):
    w = returns[tickers].tail(int(lookback))
    return w.corr(min_periods=max(10, int(0.5 * lookback)))


def bootstrap_ci(returns, a, b, lookback, n_boot=300):
    df = returns[[a, b]].dropna().tail(int(lookback))
    n  = len(df)
    if n < 30:
        return None
    rng  = np.random.default_rng(7)
    av, bv = df[a].to_numpy(), df[b].to_numpy()
    cors = [np.corrcoef(av[rng.integers(0, n, n)], bv[rng.integers(0, n, n)])[0, 1] for _ in range(n_boot)]
    return [float(np.nanpercentile(cors, 2.5)), float(np.nanpercentile(cors, 97.5))]


def lagged_correlations(window, a, b, max_lag=5):
    df  = window[[a, b]].dropna()
    out = {"a_leads_b": [], "b_leads_a": []}
    if len(df) < 15:
        return out
    av, bv = df[a], df[b]
    for k in range(1, max_lag + 1):
        c_ab = av.corr(bv.shift(-k))
        c_ba = bv.corr(av.shift(-k))
        out["a_leads_b"].append({"lag_days": k, "corr": None if pd.isna(c_ab) else float(c_ab)})
        out["b_leads_a"].append({"lag_days": k, "corr": None if pd.isna(c_ba) else float(c_ba)})
    return out


def extreme_move_overlap(window, a, b, top_n=10):
    df = window[[a, b]].dropna()
    if df.empty:
        return {"top_n": top_n, "overlap_days": 0, "same_direction": 0, "opposite_direction": 0}
    av, bv  = df[a], df[b]
    a_top   = av.abs().nlargest(min(top_n, len(av))).index
    b_top   = bv.abs().nlargest(min(top_n, len(bv))).index
    overlap = a_top.intersection(b_top)
    same    = sum(1 for d in overlap if np.sign(av.loc[d]) == np.sign(bv.loc[d]) and np.sign(av.loc[d]) != 0)
    return {"top_n": top_n, "overlap_days": int(len(overlap)), "same_direction": same, "opposite_direction": len(overlap) - same}


def window_sensitivity(returns, a, b, end_date, threshold):
    rows = []
    for w in [20, 60, 120]:
        sub = returns[[a, b]].loc[:end_date].dropna()
        win = sub.tail(w)
        if len(win) >= max(10, int(0.5 * w)):
            c = float(win[a].corr(win[b]))
        else:
            c = None
        rows.append({
            "lookback": w, "corr": c,
            "edge": None if c is None else ("connected" if abs(c) >= threshold else "not_connected"),
        })
    return rows


def build_evidence_json(md, a, b, lookback, threshold):
    asof   = md.weekly_anchors[-1]
    window = md.returns[[a, b]].loc[:asof].dropna().tail(int(lookback))

    raw_corr = float(window[a].corr(window[b])) if len(window) >= 10 else None
    corr_now = None if (raw_corr is None or pd.isna(raw_corr)) else raw_corr

    idx      = md.weekly_anchors.index(asof)
    corr_4w  = None
    if idx >= 4:
        w4      = md.returns[[a, b]].loc[:md.weekly_anchors[idx - 4]].dropna().tail(int(lookback))
        raw_4w  = float(w4[a].corr(w4[b])) if len(w4) >= 10 else None
        corr_4w = None if (raw_4w is None or pd.isna(raw_4w)) else raw_4w

    corr_change = float(corr_now - corr_4w) if (corr_now is not None and corr_4w is not None) else None
    edge_status = None if corr_now is None else ("connected" if abs(corr_now) >= threshold else "not_connected")

    lag     = lagged_correlations(window, a, b)
    overlap = extreme_move_overlap(window, a, b)
    sens    = window_sensitivity(md.returns, a, b, asof, threshold)
    ci_raw  = bootstrap_ci(md.returns, a, b, lookback)
    ci      = {"ci_95": ci_raw, "n": int(len(window)), "note": "ok"} if ci_raw else {"ci_95": None, "n": int(len(window)), "note": "sample_too_small"}

    return {
        "pair":          {"asset_a": a, "asset_b": b},
        "time_context":  {"asof_week": asof.strftime("%Y-%m-%d"), "lookback_days": int(lookback)},
        "relationship":  {
            "corr_now": corr_now, "corr_4w_ago": corr_4w,
            "corr_change_vs_4w": corr_change,
            "edge_threshold": float(threshold), "edge_status": edge_status,
        },
        "lead_lag":             lag,
        "extreme_move_overlap": overlap,
        "stability":            {"window_sensitivity": sens, "bootstrap_ci_95": ci, "effective_sample_size": int(len(window))},
    }


def cluster_order(corr):
    if corr is None or corr.empty or len(corr) <= 2:
        return list(corr.index) if corr is not None else []
    c = corr.fillna(0).clip(-1, 1)
    dist = 1 - c
    np.fill_diagonal(dist.values, 0)
    return c.index[leaves_list(linkage(squareform(dist.values, checks=False), method="average"))].tolist()


# =============================================================================
# AI
# =============================================================================
SYSTEM_PROMPT = """You are a senior quantitative analyst inside "Contagion Explorer", producing institutional-quality research notes.

Analyze the quantitative evidence and produce a structured research note. Think step by step:
1. What does the current correlation level tell us about co-movement?
2. How has the relationship changed recently (4-week delta)? Is it strengthening or weakening?
3. What do the lead-lag cross-correlations reveal about predictability?
4. Is the relationship stable (narrow bootstrap CI) or unreliable (wide CI, small sample)?
5. Do the window sensitivity tests show the correlation is robust across timeframes?
6. What do the extreme move overlaps tell us about tail-risk co-movement?
7. Based on the candidate's correlations to the full portfolio, does it diversify or overlap? Explicitly reference multiple holdings.

Rules:
1. Base your analysis ONLY on the provided evidence. Be specific — cite actual numbers.
2. Discuss "co-movement" / "association" — never claim causation.
3. If the bootstrap CI is wide or the sample is small, flag the relationship as unreliable.
4. Be concise, precise, and honest about uncertainty. No hype, no trading advice.
5. Return ONLY valid JSON matching the schema below — no markdown, no extra text.

JSON schema:
{
  "pair": {"asset_a": string, "asset_b": string},
  "time_context": {"asof_week": string, "lookback_days": number},
  "regime_summary": {"label": "calm"|"mixed"|"stress"|"unclear", "one_sentence": string},
  "portfolio_fit": {"summary": string, "overlaps": [string], "diversifiers": [string]},
  "story": string,
  "what_changed": [{"finding": string, "evidence": string}],
  "lead_lag": {"summary": string, "likely_leader": "A"|"B"|"none"|"unclear", "supporting_evidence": [string]},
  "relationship_now": {"corr_now": number, "corr_change_vs_4w": number, "edge_status": "connected"|"not_connected", "interpretation": string},
  "reliability": {"overall_confidence": "high"|"medium"|"low", "notes": [string]},
  "practical_implications": [string],
  "caveats": [string]
}

"portfolio_fit.summary" must explicitly mention at least two holdings from the portfolio list (not just the top correlated one).
"story" should be 3-5 sentences.
"what_changed" must have 2-4 items, each citing specific evidence.
"practical_implications" must have exactly 3 items.
"caveats" must have exactly 3 items."""


def _extract_json(text):
    start = text.find("{")
    if start == -1:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if esc:    esc = False; continue
        if ch == "\\": esc = True; continue
        if ch == '"':  in_str = not in_str; continue
        if in_str:     continue
        if ch == "{":  depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:    return json.loads(text[start:i + 1])
                except: return None
    return None


def _get_client(api_key, provider, custom_base_url=""):
    from openai import OpenAI
    reg      = PROVIDER_REGISTRY.get(provider, PROVIDER_REGISTRY["OpenAI"])
    base_url = custom_base_url.strip() if provider == "Custom (OpenAI-compatible)" else reg["base_url"]
    if provider == "Custom (OpenAI-compatible)" and not base_url:
        raise Exception("Custom provider requires a base URL in the sidebar.")
    return OpenAI(api_key=api_key, base_url=base_url or None, timeout=120)


def _chat(client, provider, model, messages, temperature=0.2, max_tokens=1400):
    if provider == "OpenAI" and model.startswith("gpt-5"):
        system_text = "\n\n".join([m["content"] for m in messages if m["role"] == "system"])
        user_text = "\n\n".join([m["content"] for m in messages if m["role"] == "user"])
        input_text = (f"System:\n{system_text}\n\nUser:\n{user_text}").strip()
        resp = client.responses.create(
            model=model,
            input=input_text,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=resp.output_text or ""))])
    kwargs = {"extra_body": {"chat_template_kwargs": {"thinking": False}}} if provider == "NVIDIA NIM" else {}
    return client.chat.completions.create(
        model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, **kwargs
    )


def _portfolio_fit_mentions_ok(portfolio, text, min_mentions=2):
    if not portfolio or not text:
        return False
    text_l = text.lower()
    matches = 0
    for t in portfolio:
        if t and t.lower() in text_l:
            matches += 1
    return matches >= min_mentions


def verify_api_key(provider, model, api_key, custom_base_url=""):
    if not api_key:
        return {"ok": False, "message": "API key is empty."}
    try:
        client     = _get_client(api_key, provider, custom_base_url)
        t0         = time.perf_counter()
        resp       = _chat(client, provider, model, [{"role": "user", "content": "Reply with exactly: OK"}], temperature=0, max_tokens=8)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        return {
            "ok": True,
            "message": (resp.choices[0].message.content or "").strip(),
            "latency_ms": elapsed_ms,
            "provider": provider, "model": model,
            "checked_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "key_fp": hashlib.sha256(api_key.encode()).hexdigest()[:12],
        }
    except Exception as e:
        return {
            "ok": False, "message": str(e), "provider": provider, "model": model,
            "checked_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "key_fp": hashlib.sha256(api_key.encode()).hexdigest()[:12] if api_key else "",
        }


def list_available_models(api_key, custom_base_url=""):
    if not api_key:
        return {"ok": False, "message": "API key is empty."}
    try:
        client = _get_client(api_key, "OpenAI", custom_base_url)
        models = client.models.list().data
        ids = sorted([m.id for m in models if getattr(m, "id", None)])
        return {"ok": True, "models": ids}
    except Exception as e:
        return {"ok": False, "message": str(e)}


def generate_ai_brief(evidence, provider, model, api_key, custom_base_url=""):
    if not api_key:
        raise Exception("API key required")
    user_prompt = f"Analyze this evidence and return valid JSON:\n\n{json.dumps(evidence, indent=1)}"
    max_tokens  = 2500
    client      = _get_client(api_key, provider, custom_base_url)
    messages    = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}]
    try:
        resp   = _chat(client, provider, model, messages, max_tokens=max_tokens)
        result = _extract_json(resp.choices[0].message.content or "")
        if result is None:
            raise ValueError("No valid JSON returned")
        portfolio = evidence.get("portfolio_context", {}).get("portfolio", [])
        pf_summary = (result.get("portfolio_fit") or {}).get("summary", "")
        if portfolio and not _portfolio_fit_mentions_ok(portfolio, pf_summary, min_mentions=2):
            raise ValueError("Portfolio fit summary missing holdings")
        return result
    except Exception:
        # Retry with stricter instruction
        portfolio = evidence.get("portfolio_context", {}).get("portfolio", [])
        must_list = ", ".join(portfolio) if portfolio else "N/A"
        messages[0]["content"] = SYSTEM_PROMPT + "\nReturn ONLY valid JSON. No markdown fences."
        messages[1]["content"] = (
            user_prompt
            + "\n\nOutput a single JSON object."
            + f"\n\nIMPORTANT: In portfolio_fit.summary, explicitly reference at least two of these holdings: {must_list}."
        )
        resp   = _chat(client, provider, model, messages, temperature=0.1, max_tokens=max_tokens)
        result = _extract_json(resp.choices[0].message.content or "")
        if result is None:
            raise ValueError("Model returned no valid JSON after retry.")
        return result


def generate_pair_explanation(a, b, corr_val, provider, model, api_key, custom_base_url=""):
    if not api_key:
        return {}
    corr_str = f"{corr_val:+.2f}" if corr_val is not None else "unknown"
    prompt = f"""You are a clear financial educator explaining stock market relationships to a non-expert.

Two assets show a correlation of {corr_str}: {a} and {b}.

Write a short explanation covering:
1. What is {a}? (1-2 sentences)
2. What is {b}? (1-2 sentences)
3. Why are they correlated? Be specific — name actual industries, products, or economic forces. If correlation is negative, explain why they move in opposite directions.

Rules: plain English, no jargon, 4-6 sentences total, flowing prose (no bullet points).

Return ONLY JSON: {{"asset_a_description": "...", "asset_b_description": "...", "correlation_explanation": "...", "confidence": "high"|"medium"|"low"}}"""
    try:
        client = _get_client(api_key, provider, custom_base_url)
        resp   = _chat(client, provider, model, [{"role": "user", "content": prompt}], temperature=0.3, max_tokens=400)
        return _extract_json(resp.choices[0].message.content or "") or {}
    except Exception as e:
        return {"error": str(e)}


def fetch_news(ticker, api_key):
    if not api_key:
        return []
    try:
        r = requests.get(
            f"https://newsapi.org/v2/everything?q={ticker}&sortBy=publishedAt&pageSize=5&apiKey={api_key}",
            timeout=10,
        )
        if r.status_code == 200:
            return r.json().get("articles", [])
    except Exception:
        pass
    return []


def analyze_news_with_ai(news_items, ticker, evidence, api_key, provider, model, custom_base_url=""):
    """Given recent headlines, answer: should I add {ticker} to this specific portfolio?"""
    if not api_key or not news_items:
        return None
    try:
        max_news   = 5
        news_text  = "\n".join([f"- {n.get('title', '')}: {(n.get('description') or '')[:150]}" for n in news_items[:max_news]])
        port_ctx   = evidence.get("portfolio_context", {})
        portfolio  = port_ctx.get("portfolio", [])
        cand_corrs = port_ctx.get("candidate_correlations", {})
        corrs_str  = ", ".join([f"{k}: {v:+.2f}" for k, v in list(cand_corrs.items())[:5]]) if cand_corrs else "N/A"

        prompt = f"""You are a portfolio analyst. A manager holds [{', '.join(portfolio)}] and is deciding whether to add {ticker}.

RECENT NEWS for {ticker}:
{news_text}

QUANTITATIVE CONTEXT:
- {ticker} correlation to current holdings: {corrs_str}
- Portfolio internal avg |ρ|: {port_ctx.get('portfolio_avg_abs_corr', 'N/A')}

Based on the news and the quantitative fit data above, give a structured answer to: "Should I add {ticker} to this portfolio right now?"

Return JSON:
{{"recommendation":"add"|"wait"|"avoid","conviction":"high"|"medium"|"low","headline_summary":"What is happening with this stock right now in 1-2 sentences","investment_case":"Specific reasons to add given this portfolio's existing holdings","key_risks":["risk1","risk2","risk3"],"timing":"Is now a good entry point and why","catalyst_watch":["specific upcoming event or data point to monitor"]}}
Output only valid JSON."""
        client = _get_client(api_key, provider, custom_base_url)
        resp   = _chat(client, provider, model, [{"role": "user", "content": prompt}], max_tokens=700)
        return _extract_json(resp.choices[0].message.content or "")
    except Exception as e:
        return {"error": str(e)}


def web_search_deep_analysis(candidate, top_asset, evidence, api_key, model="gpt-4.1-mini", portfolio=None):
    """OpenAI Responses API with web_search — answers: should I add {candidate} to this portfolio?"""
    from openai import OpenAI
    port_ctx   = evidence.get("portfolio_context", {})
    portfolio  = portfolio or port_ctx.get("portfolio", [])
    cand_corrs = port_ctx.get("candidate_correlations", {})
    corrs_str  = ", ".join([f"{k}: {v:+.2f}" for k, v in cand_corrs.items()]) if cand_corrs else "N/A"

    prompt = f"""You are a senior portfolio analyst with live internet access.

A portfolio manager currently holds: {', '.join(portfolio) if portfolio else 'N/A'}
They are deciding whether to ADD {candidate} to this portfolio.

QUANTITATIVE FIT (already computed — do not recalculate):
- {candidate} rolling correlation to each holding: {corrs_str}
- Strongest correlated holding: {top_asset}
- Portfolio internal avg |ρ|: {port_ctx.get('portfolio_avg_abs_corr', 'N/A')}

Use web search to find the following for {candidate}:
1. Most recent earnings, revenue, or guidance news
2. Analyst rating or price target changes in the last 60 days
3. Key macro or sector tailwinds/headwinds affecting {candidate} right now
4. Any catalyst or event coming up that could affect volatility or correlation

Then give a clear investment recommendation: should this manager add {candidate} to their portfolio RIGHT NOW?

Return JSON:
{{"recommendation":"add"|"wait"|"avoid","conviction":"high"|"medium"|"low","headline_summary":"What is happening with {candidate} right now in 1-2 sentences","investment_case":"Specific reasons to add given this portfolio's existing holdings","key_risks":["risk1","risk2","risk3"],"timing":"Is now a good entry point and why","catalyst_watch":["specific upcoming event or data point to monitor"],"recent_headlines":["headline1","headline2","headline3"],"sentiment":"positive"|"negative"|"neutral","sector_context":"Brief macro or sector commentary relevant to {candidate}"}}
Output ONLY valid JSON."""

    client = OpenAI(api_key=api_key)
    try:
        response = client.responses.create(model=model, tools=[{"type": "web_search"}], input=prompt)
        text = response.output_text
    except Exception:
        response = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}], temperature=0.2, max_tokens=1200,
        )
        text = response.choices[0].message.content or ""
    result = _extract_json(text)
    return result if result else {"raw_analysis": text[:2000]}


# =============================================================================
# CHARTS
# =============================================================================
def chart_corr_series(df, threshold, a, b, lookback):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["week"], y=df["corr"], mode="lines",
        line=dict(color="#3b82f6", width=2),
        fill="tozeroy", fillcolor="rgba(59,130,246,0.08)",
        hovertemplate="<b>%{x|%b %Y}</b><br>ρ = %{y:.2f}<extra></extra>",
    ))
    if not df.empty:
        fig.add_trace(go.Scatter(
            x=[df["week"].iloc[-1]], y=[df["corr"].iloc[-1]], mode="markers",
            marker=dict(size=10, color="#a78bfa", line=dict(width=2, color="white")),
            hovertemplate="<b>Current</b><br>ρ = %{y:.3f}<extra></extra>", showlegend=False,
        ))
    fig.add_hline(y=threshold,  line_dash="dash", line_color="green", line_width=1,
                  annotation_text=f"threshold {threshold:+.1f}", annotation_font_size=11)
    fig.add_hline(y=-threshold, line_dash="dash", line_color="red",   line_width=1)
    fig.add_hline(y=0,          line_color="gray", line_width=0.5)
    fig.update_layout(
        height=260, margin=dict(l=0, r=40, t=30, b=0),
        yaxis=dict(range=[-1, 1], title="ρ", tickformat=".1f"),
        xaxis=dict(showgrid=False),
        showlegend=False, hovermode="x unified",
        title=dict(text=f"{a} vs {b} — rolling {lookback}d correlation", font_size=13),
    )
    return fig


def chart_heatmap(corr, order):
    cm = corr.reindex(index=order, columns=order)
    colorscale = [[0, "#dc2626"], [0.35, "#6b7280"], [0.5, "#f8fafc"], [0.65, "#6b7280"], [1, "#16a34a"]]
    fig = go.Figure(go.Heatmap(
        z=cm.values, x=order, y=order, zmin=-1, zmax=1,
        colorscale=colorscale,
        text=np.around(cm.values, 2).astype(str),
        texttemplate="%{text}", textfont=dict(size=9),
        hovertemplate="%{x} vs %{y}: ρ = %{z:.2f}<extra></extra>",
        colorbar=dict(thickness=10, tickvals=[-1, -0.5, 0, 0.5, 1]),
    ))
    fig.update_layout(
        height=380, margin=dict(l=0, r=0, t=30, b=0),
        xaxis=dict(tickangle=45, tickfont_size=10, side="bottom"),
        yaxis=dict(tickfont_size=10, autorange="reversed"),
        title=dict(text="Portfolio correlation matrix", font_size=13),
    )
    return fig


def chart_network(corr, tickers, threshold):
    n      = len(tickers)
    angles = [2 * math.pi * i / n for i in range(n)]
    nx_    = [0.5 + 0.38 * math.cos(a) for a in angles]
    ny_    = [0.5 + 0.38 * math.sin(a) for a in angles]
    traces = []
    for i, a in enumerate(tickers):
        for j, b in enumerate(tickers):
            if i >= j: continue
            v = corr.loc[a, b] if a in corr.index and b in corr.columns else np.nan
            if pd.isna(v) or abs(v) < threshold: continue
            traces.append(go.Scatter(
                x=[nx_[i], nx_[j]], y=[ny_[i], ny_[j]], mode="lines",
                line=dict(width=1 + 4 * abs(v), color="#3b82f6" if v > 0 else "#ef4444"),
                opacity=0.3 + 0.5 * abs(v), hoverinfo="skip", showlegend=False,
            ))
    traces.append(go.Scatter(
        x=nx_, y=ny_, mode="markers+text", text=tickers, textposition="middle center",
        textfont=dict(size=9),
        marker=dict(size=32, color="white", line=dict(width=1.5, color="#94a3b8")),
        hovertemplate="%{text}<extra></extra>", showlegend=False,
    ))
    fig = go.Figure(traces)
    fig.update_layout(
        height=360, margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=dict(visible=False, range=[0, 1], scaleanchor="x"),
        showlegend=False,
    )
    return fig


def chart_price_overlay(prices, tickers, lookback_days=None):
    df = prices[tickers].dropna()
    if lookback_days:
        df = df.tail(lookback_days)
    if df.empty:
        return go.Figure()
    norm    = df / df.iloc[0] * 100
    palette = ["#3b82f6", "#a78bfa", "#22c55e", "#f59e0b", "#ef4444", "#38bdf8", "#e879f9"]
    fig = go.Figure()
    for i, t in enumerate(tickers):
        if t in norm.columns:
            fig.add_trace(go.Scatter(
                x=norm.index, y=norm[t], mode="lines", name=t,
                line=dict(color=palette[i % len(palette)], width=2),
                hovertemplate=f"<b>{t}</b><br>%{{x}}<br>Index: %{{y:.1f}}<extra></extra>",
            ))
    fig.add_hline(y=100, line_color="rgba(148,163,184,0.3)", line_width=1, line_dash="dot")
    fig.update_layout(
        height=260, margin=dict(l=0, r=10, t=30, b=0),
        title=dict(text="Normalized prices (indexed to 100)", font_size=13),
        yaxis=dict(gridcolor="rgba(148,163,184,0.1)"),
        xaxis=dict(showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    return fig


def chart_gauge(value, title):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value, domain={"x": [0, 1], "y": [0, 1]},
        title={"text": title, "font": {"size": 13}},
        number={"font": {"size": 26}, "valueformat": ".3f"},
        gauge={
            "axis": {"range": [-1, 1]},
            "bar":  {"color": "#3b82f6", "thickness": 0.6},
            "steps": [
                {"range": [-1,   -0.5], "color": "rgba(248,113,113,0.15)"},
                {"range": [-0.5,  0.5], "color": "rgba(148,163,184,0.08)"},
                {"range": [0.5,    1],  "color": "rgba(96,165,250,0.15)"},
            ],
        },
    ))
    fig.update_layout(height=220, margin=dict(l=20, r=20, t=45, b=15))
    return fig


# =============================================================================
# SIDEBAR
# =============================================================================
def render_sidebar():
    with st.sidebar:
        st.title("🌊 Contagion Explorer")
        st.caption("Should I add this asset to my portfolio — or is it just more of the same risk?")
        st.divider()

        st.subheader("My Portfolio")
        custom = [t for t in st.session_state.get("custom_tickers", []) if t not in DEFAULT_TICKERS]
        all_tickers = list(DEFAULT_TICKERS.keys()) + custom
        st.multiselect(
            "Current holdings",
            options=all_tickers,
            default=st.session_state["portfolio"],
            format_func=lambda t: f"{t} — {_ticker_label(t)}",
            help="The assets you already hold.",
            key="portfolio",
        )
        st.caption("Add more tickers (comma or space separated):")
        st.text_input(
            "Add holdings",
            key="portfolio_add_input",
            label_visibility="collapsed",
            placeholder="e.g. AMD, NFLX, BRK-B",
        )
        if st.button("＋ Add", key="portfolio_add_btn", use_container_width=True):
            raw = st.session_state.get("portfolio_add_input", "")
            to_add = _resolve_ticker_input(raw)
            if to_add:
                newly_added = []
                for t in to_add:
                    if t not in st.session_state["custom_tickers"] and t not in DEFAULT_TICKERS:
                        st.session_state["custom_tickers"].append(t)
                    if t not in st.session_state["portfolio"]:
                        st.session_state["portfolio"].append(t)
                        newly_added.append(t)
                st.session_state["last_added"] = newly_added if newly_added else "already_in"
                st.rerun()
            else:
                st.session_state["last_added"] = "not_found"

        last = st.session_state.get("last_added")
        if last == "already_in":
            st.caption("✓ Already selected.")
        elif last == "not_found":
            st.caption("⚠️ Not recognised — try a ticker symbol.")
        elif isinstance(last, list) and last:
            st.caption(f"✓ Added: {', '.join(last)}")

        st.subheader("Candidate Asset")
        candidate_opts = [t for t in all_tickers if t not in st.session_state["portfolio"]]
        if st.session_state["candidate"] not in candidate_opts and candidate_opts:
            st.session_state["candidate"] = candidate_opts[0]
        st.session_state["candidate"] = st.selectbox(
            "Asset you're considering adding",
            options=candidate_opts,
            index=candidate_opts.index(st.session_state["candidate"]) if st.session_state["candidate"] in candidate_opts else 0,
            format_func=lambda t: f"{t} — {_ticker_label(t)}",
        )

        st.divider()
        st.subheader("Settings")
        st.session_state["lookback"] = st.select_slider(
            "Lookback window",
            options=[20, 60, 120],
            value=st.session_state["lookback"],
            help="Trading days used for correlation calculation.",
        )
        st.session_state["threshold"] = st.slider(
            "Contagion threshold |ρ|", 0.2, 0.9,
            value=st.session_state["threshold"], step=0.05,
            help="Pairs above this are considered 'linked'.",
        )
        st.session_state["use_demo"] = st.toggle(
            "Demo mode (synthetic data)",
            value=st.session_state["use_demo"],
        )

        st.divider()
        st.subheader("AI Settings")
        provider = st.selectbox(
            "Provider", list(PROVIDER_REGISTRY.keys()),
            index=list(PROVIDER_REGISTRY.keys()).index(st.session_state["llm_provider"]),
        )
        st.session_state["llm_provider"] = provider
        default_model = PROVIDER_REGISTRY[provider]["default_model"]
        if st.session_state.get("_last_provider") != provider:
            st.session_state["llm_model"]      = default_model
            st.session_state["_last_provider"] = provider
        if provider == "OpenAI":
            current = st.session_state["llm_model"]
            options = OPENAI_MODELS + ["Custom…"]
            index = options.index(current) if current in OPENAI_MODELS else options.index("Custom…")
            selected = st.selectbox("Model", options=options, index=index)
            if selected == "Custom…":
                st.session_state["llm_model"] = st.text_input("Custom model", value=current)
            else:
                st.session_state["llm_model"] = selected
        else:
            st.session_state["llm_model"] = st.text_input("Model", value=st.session_state["llm_model"])

        if provider == "Custom (OpenAI-compatible)":
            st.session_state["llm_custom_base_url"] = st.text_input(
                "Custom Base URL", value=st.session_state.get("llm_custom_base_url", ""),
                placeholder="https://your-api.example.com/v1",
            )

        st.session_state["llm_api_key"] = st.text_input(
            "API Key", value=st.session_state["llm_api_key"], type="password",
            help="Stored in session only.",
        )

        if st.button("🧪 Test API Key", use_container_width=True):
            with st.spinner(f"Testing {provider}…"):
                st.session_state["llm_last_key_check"] = verify_api_key(
                    provider, st.session_state["llm_model"],
                    st.session_state["llm_api_key"],
                    st.session_state.get("llm_custom_base_url", ""),
                )
        check = st.session_state.get("llm_last_key_check")
        if check:
            if check.get("ok"):
                st.success(f"✅ OK — {check.get('latency_ms')}ms ({check.get('provider')})")
            else:
                st.error(f"❌ {str(check.get('message', 'Failed'))[:80]}")

        if provider == "OpenAI":
            if st.button("📋 Check available models", use_container_width=True):
                with st.spinner("Fetching model list…"):
                    st.session_state["llm_models_check"] = list_available_models(
                        st.session_state["llm_api_key"],
                        st.session_state.get("llm_custom_base_url", ""),
                    )
            models_check = st.session_state.get("llm_models_check")
            if models_check:
                if models_check.get("ok"):
                    with st.expander("Available models", expanded=False):
                        st.code("\n".join(models_check.get("models", [])[:200]))
                else:
                    st.error(f"❌ {str(models_check.get('message', 'Failed'))[:120]}")

        st.session_state["ai_analysis_depth"] = "Deep"

        st.session_state["news_api_key"] = st.text_input(
            "NewsAPI Key (optional)", value=st.session_state["news_api_key"], type="password",
            help="For real news headlines in Deep mode (newsapi.org).",
        )


# =============================================================================
# MAIN
# =============================================================================
def main():
    render_sidebar()

    portfolio  = st.session_state["portfolio"]
    candidate  = st.session_state["candidate"]
    lookback   = st.session_state["lookback"]
    threshold  = st.session_state["threshold"]
    api_key    = st.session_state["llm_api_key"]
    provider   = st.session_state["llm_provider"]
    model      = st.session_state["llm_model"]
    depth      = "Deep"
    custom_url = st.session_state.get("llm_custom_base_url", "")

    if len(portfolio) < 2:
        st.warning("Add at least 2 assets to your portfolio in the sidebar.")
        return

    # Load data
    with st.spinner("Loading market data…"):
        md = load_data(portfolio + [candidate], demo=st.session_state["use_demo"])

    if st.session_state["use_demo"]:
        st.info("Running in demo mode with synthetic data.")
    if md.tickers_failed:
        st.warning(f"Could not load: {', '.join(md.tickers_failed)}")
    if len(md.tickers_ok) < 2:
        st.error("Not enough data. Try enabling demo mode.")
        return
    if candidate not in md.tickers_ok:
        st.error(f"Could not load data for {candidate}.")
        return

    portfolio_ok = [t for t in portfolio if t in md.tickers_ok]

    # =========================================================================
    # HEADER
    # =========================================================================
    st.markdown(f"## Should I add **{candidate}** to my portfolio?")
    st.caption(
        f"You hold: {', '.join(portfolio_ok)}. "
        f"This tool checks whether {candidate} genuinely diversifies your portfolio "
        f"or just adds more of the same risk."
    )
    st.divider()

    # =========================================================================
    # SECTION 1 — Current portfolio
    # =========================================================================
    st.subheader("1 · How correlated is my current portfolio?")
    st.caption("Before evaluating the candidate, understand what you already hold.")

    corr_portfolio = full_corr_matrix(md.returns, portfolio_ok, lookback)
    order = cluster_order(corr_portfolio)

    col_net, col_heat = st.columns(2)
    portfolio_key = hashlib.sha1(",".join(portfolio_ok).encode()).hexdigest()[:8]
    with col_net:
        st.plotly_chart(chart_network(corr_portfolio, portfolio_ok, threshold),
                        use_container_width=True, key=f"network_{portfolio_key}", config={"responsive": True})
        st.caption("Blue = positive correlation · Red = negative (hedge) · Thicker = stronger.")
    with col_heat:
        st.plotly_chart(chart_heatmap(corr_portfolio, order),
                        use_container_width=True, key=f"heatmap_{portfolio_key}", config={"responsive": True})

    n         = len(portfolio_ok)
    max_edges = n * (n - 1) // 2
    linked    = sum(
        1 for i in range(n) for j in range(i + 1, n)
        if not pd.isna(corr_portfolio.iloc[i, j]) and abs(corr_portfolio.iloc[i, j]) >= threshold
    )
    avg_corr      = np.nanmean([abs(corr_portfolio.iloc[i, j]) for i in range(n) for j in range(i + 1, n)]) if n > 1 else 0
    contagion_pct = linked / max_edges if max_edges > 0 else 0

    if contagion_pct >= 0.60:
        risk_label = "🔴 HIGH"
    elif contagion_pct >= 0.30:
        risk_label = "🟡 MODERATE"
    else:
        risk_label = "🟢 LOW"

    with st.expander("Portfolio summary", expanded=True):
        m1, m2 = st.columns(2)
        m1.metric("Avg |ρ| in portfolio", f"{avg_corr:.2f}",
                  help="Average absolute correlation. Higher = less diversified.")
        m2.metric("Linked pairs", f"{linked} / {max_edges}",
                  help=f"Pairs with |ρ| ≥ {threshold}.")
        m3, m4 = st.columns(2)
        m3.metric("Contagion level", f"{risk_label} ({contagion_pct:.0%})")
        m4.metric("Lookback window", f"{lookback} days")

    st.divider()

    # =========================================================================
    # SECTION 2 — Candidate fit
    # =========================================================================
    st.subheader(f"2 · Does {candidate} add real diversification?")
    st.caption(
        f"The question is not just how {candidate} compares to one other stock — "
        f"it's whether it moves independently of your **whole portfolio**."
    )

    candidate_corrs = {
        asset: rolling_corr(md.returns, candidate, asset, lookback)
        for asset in portfolio_ok
    }
    valid_corrs = [(a, c) for a, c in candidate_corrs.items() if c is not None]

    # Portfolio-level fit metrics
    cand_avg_corr = np.mean([abs(c) for _, c in valid_corrs]) if valid_corrs else 0
    overlapping   = [a for a, c in valid_corrs if abs(c) >= threshold]
    diversifying  = [a for a, c in valid_corrs if abs(c) < 0.2 or c < 0]

    if cand_avg_corr < avg_corr - 0.05:
        fit_color  = "🟢"
        fit_label  = "Strong diversifier"
        fit_detail = (
            f"Avg |ρ| to your portfolio = **{cand_avg_corr:.2f}**, below the portfolio's own internal avg "
            f"of {avg_corr:.2f}. {candidate} moves more independently than your holdings move with each other."
        )
    elif cand_avg_corr > avg_corr + 0.10:
        fit_color  = "🔴"
        fit_label  = "Risk amplifier"
        fit_detail = (
            f"Avg |ρ| to your portfolio = **{cand_avg_corr:.2f}**, above the portfolio's internal avg "
            f"of {avg_corr:.2f}. Adding {candidate} concentrates risk rather than spreading it."
        )
    else:
        fit_color  = "🟡"
        fit_label  = "Neutral fit"
        fit_detail = (
            f"Avg |ρ| to your portfolio = **{cand_avg_corr:.2f}**, similar to the portfolio's internal avg "
            f"of {avg_corr:.2f}. Modest diversification benefit at best."
        )

    st.markdown(f"#### {fit_color} {candidate} is a **{fit_label}** for this portfolio")
    st.caption(fit_detail)

    # Three headline metrics
    fa, fb, fc = st.columns(3)
    fa.metric(
        "Avg |ρ| to portfolio", f"{cand_avg_corr:.2f}",
        delta=f"{cand_avg_corr - avg_corr:+.2f} vs portfolio internal avg",
        delta_color="inverse",
        help="Lower means the candidate moves more independently of what you already hold.",
    )
    fb.metric(
        "Overlapping holdings", f"{len(overlapping)} / {len(portfolio_ok)}",
        help=f"|ρ| ≥ {threshold}: {', '.join(overlapping) if overlapping else 'None'}.",
    )
    fc.metric(
        "Independent holdings", f"{len(diversifying)} / {len(portfolio_ok)}",
        help=f"|ρ| < 0.2 or negative: {', '.join(diversifying) if diversifying else 'None'}.",
    )

    # Horizontal bar chart — candidate vs each holding, color-coded
    if valid_corrs:
        sorted_corrs = sorted(valid_corrs, key=lambda x: x[1])
        asset_labels = [a for a, _ in sorted_corrs]
        corr_values  = [c for _, c in sorted_corrs]
        bar_colors   = [
            "#ef4444" if abs(c) >= threshold else
            "#22c55e" if (abs(c) < 0.2 or c < 0) else
            "#94a3b8"
            for c in corr_values
        ]
        bar_fig = go.Figure(go.Bar(
            x=corr_values, y=asset_labels, orientation="h",
            marker_color=bar_colors,
            text=[f"{c:+.2f}" for c in corr_values],
            textposition="outside",
            hovertemplate="%{y}: ρ = %{x:.3f}<extra></extra>",
        ))
        bar_fig.add_vline(x=threshold,  line_dash="dash", line_color="#ef4444", line_width=1,
                          annotation_text=f"threshold +{threshold}", annotation_position="top right",
                          annotation_font_size=10)
        bar_fig.add_vline(x=-threshold, line_dash="dash", line_color="#22c55e", line_width=1,
                          annotation_text=f"−{threshold}", annotation_position="top left",
                          annotation_font_size=10)
        bar_fig.add_vline(x=0, line_color="gray", line_width=0.5)
        bar_fig.update_layout(
            height=max(200, len(asset_labels) * 44),
            margin=dict(l=0, r=70, t=40, b=0),
            xaxis=dict(range=[-1, 1], title="Rolling correlation ρ", tickformat=".1f"),
            title=dict(text=f"How correlated is {candidate} with each holding?", font_size=13),
            showlegend=False,
        )
        st.plotly_chart(bar_fig, use_container_width=True, key="fit_bar_chart")
        st.caption(
            f"🔴 Overlaps (|ρ| ≥ {threshold})  ·  🟢 Diversifies (|ρ| < 0.2 or negative)  ·  ⚪ Neutral"
        )

    # Find strongest pair (needed for downstream sections)
    top_asset, top_corr = max(valid_corrs, key=lambda x: abs(x[1]), default=(None, None))

    # Before vs After heatmap — the clearest way to see portfolio impact
    st.markdown("**What does the portfolio look like if you add it?**")
    extended_tickers = portfolio_ok + [candidate]
    corr_extended    = full_corr_matrix(md.returns, extended_tickers, lookback)
    order_extended   = cluster_order(corr_extended)
    ext_key          = hashlib.sha1((",".join(extended_tickers)).encode()).hexdigest()[:8]
    col_before, col_after = st.columns(2)
    with col_before:
        st.caption("**Before** — current portfolio")
        st.plotly_chart(chart_heatmap(corr_portfolio, order),
                        use_container_width=True, key="heatmap_before")
    with col_after:
        st.caption(f"**After** — adding {candidate}")
        st.plotly_chart(chart_heatmap(corr_extended, order_extended),
                        use_container_width=True, key=f"heatmap_after_{ext_key}")

    # Rolling correlation trend + reliability for the strongest link
    if top_asset:
        hist_s2 = corr_series(md, candidate, top_asset, lookback)
        if not hist_s2.empty:
            st.caption(f"Rolling correlation trend — {candidate} vs its strongest link **{top_asset}**")
            st.plotly_chart(
                chart_corr_series(hist_s2, threshold, candidate, top_asset, lookback),
                use_container_width=True, key="corr_series_s2",
            )

        ci = bootstrap_ci(md.returns, candidate, top_asset, lookback)
        if ci:
            width       = ci[1] - ci[0]
            reliability = "reliable" if width < 0.3 else "noisy — treat with caution"
            st.info(
                f"**Signal reliability (strongest link):** 95% CI for ρ({candidate}, {top_asset}) "
                f"= [{ci[0]:+.2f}, {ci[1]:+.2f}] — {reliability}."
            )

        if md.weekly_anchors:
            sens       = window_sensitivity(md.returns, candidate, top_asset, md.weekly_anchors[-1], threshold)
            sens_parts = [
                f"{s['lookback']}d: {s['corr']:+.2f} ({s['edge'] or '—'})" if s["corr"] is not None
                else f"{s['lookback']}d: N/A"
                for s in sens
            ]
            st.caption(f"Correlation stability ({candidate}↔{top_asset}) — {' | '.join(sens_parts)}")

    st.divider()

    # =========================================================================
    # SECTION 3 — AI verdict
    # =========================================================================
    st.subheader(f"3 · Should I add {candidate}?")
    st.caption("AI reads all the evidence above and gives you a structured verdict.")

    if not api_key:
        st.warning("Add your API key in the sidebar to unlock the AI verdict.")
        return

    if not top_asset:
        st.error("Could not determine a comparison pair.")
        return

    # Build rich evidence payload
    evidence = build_evidence_json(md, candidate, top_asset, lookback, threshold)
    evidence["portfolio_context"] = {
        "portfolio": portfolio_ok,
        "candidate": candidate,
        "candidate_correlations": {a: round(c, 4) for a, c in candidate_corrs.items() if c is not None},
        "portfolio_avg_abs_corr": round(float(avg_corr), 4),
        "portfolio_linked_pairs": f"{linked}/{max_edges}",
    }

    # Scenario hint
    pair_key = (candidate, top_asset) if (candidate, top_asset) in SCENARIO_HINTS else \
               (top_asset, candidate)  if (top_asset, candidate)  in SCENARIO_HINTS else None
    if pair_key:
        st.info(f"💡 **Why this pair matters:** {SCENARIO_HINTS[pair_key]}")

    # Run button
    use_web   = provider == "OpenAI"
    run_label = (
        f"🌐 Should I add {candidate}? — Run AI analysis with live web search" if use_web else
        f"🚀 Should I add {candidate}? — Run AI analysis"
    )

    if st.button(run_label, type="primary", use_container_width=True):
        t_start = time.perf_counter()
        try:
            with st.status(f"Running {depth} AI analysis…", expanded=True) as status_box:
                status_box.write(f"**Provider:** `{provider}` · **Model:** `{model}` · **Depth:** {depth}")

                status_box.write("Step 1: Generating analysis brief…")
                brief = generate_ai_brief(evidence, provider, model, api_key, custom_url)
                st.session_state["last_brief"] = brief
                # Clear cached pair explanation so it regenerates after fresh analysis
                st.session_state.pop(f"pair_exp_{candidate}_{top_asset}", None)
                status_box.write(f"✅ Brief done in {time.perf_counter() - t_start:.1f}s")

                web_research = None
                analysis_a, analysis_b = None, None

                if use_web:
                    status_box.write(f"Step 2: 🌐 Searching for latest news on `{candidate}`…")
                    web_research = web_search_deep_analysis(candidate, top_asset, evidence, api_key, model, portfolio_ok)
                    status_box.write(f"✅ Investment research done in {time.perf_counter() - t_start:.1f}s")
                else:
                    news_items = fetch_news(candidate, st.session_state["news_api_key"])
                    if news_items:
                        status_box.write(f"Step 2: Building investment case for `{candidate}` from {len(news_items)} headlines…")
                        analysis_a = analyze_news_with_ai(news_items, candidate, evidence, api_key, provider, model, custom_url)
                        status_box.write("✅ Investment analysis done")
                    else:
                        status_box.write("ℹ️ No news found — add a NewsAPI key in the sidebar for real headlines.")
                analysis_b = None

                st.session_state["last_news_analysis"] = {
                    "asset_a": candidate, "asset_b": top_asset,
                    "web_research": web_research,
                    "analysis_a": analysis_a, "analysis_b": analysis_b,
                    "source": "web_search" if use_web else "chat",
                    "depth": depth,
                }
                total_s = time.perf_counter() - t_start
                status_box.update(label=f"✅ Analysis complete in {total_s:.1f}s", state="complete")
                st.session_state["last_ai_run_meta"] = {
                    "provider": provider, "model": model, "depth": depth,
                    "duration_s": round(total_s, 2),
                    "finished_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
        except Exception as e:
            st.error(f"Analysis failed: {e}")

    run_meta = st.session_state.get("last_ai_run_meta")
    if run_meta:
        st.caption(
            f"Last run: {run_meta.get('finished_at')} · "
            f"{run_meta.get('provider')} / {run_meta.get('model')} · "
            f"{run_meta.get('depth')} · {run_meta.get('duration_s')}s"
        )

    # ── Display results ───────────────────────────────────────────────────────
    brief = st.session_state.get("last_brief")
    if not brief:
        return

    # Extract all fields up front
    rel          = brief.get("relationship_now", {})
    reliability  = brief.get("reliability", {})
    lead_lag_ai  = brief.get("lead_lag", {})
    corr_val     = rel.get("corr_now") or 0
    corr_chg     = rel.get("corr_change_vs_4w") or 0
    confidence   = reliability.get("overall_confidence", "medium")
    leader_raw   = lead_lag_ai.get("likely_leader", "unclear")
    leader_label = {"A": candidate, "B": top_asset, "none": "Neither", "unclear": "Unclear"}.get(leader_raw, leader_raw)
    edge         = rel.get("edge_status", "unknown").replace("_", " ").title()
    regime       = brief.get("regime_summary", "")

    # Compute verdict
    if confidence == "high" and edge == "Connected":
        verdict_icon = "✅"
        verdict_text = (
            f"Strong, reliable co-movement detected. {candidate} and {top_asset} move together "
            f"(ρ = {corr_val:+.2f}). Adding {candidate} adds limited diversification — it largely "
            f"tracks what you already hold."
        )
    elif confidence == "low" or abs(corr_val) < 0.3:
        verdict_icon = "✅"
        verdict_text = (
            f"Weak or uncorrelated signal (ρ = {corr_val:+.2f}). {candidate} moves largely "
            f"independently of your portfolio — a genuine diversifier."
        )
    else:
        verdict_icon = "🔶"
        verdict_text = (
            f"Moderate co-movement (ρ = {corr_val:+.2f}, {confidence} confidence). "
            f"Monitor over the next few weeks before acting on this signal."
        )

    st.divider()

    # ── 1. VERDICT FIRST ─────────────────────────────────────────────────────
    st.markdown(f"### {verdict_icon} Bottom Line")
    st.markdown(f"**{verdict_text}**")
    if regime:
        st.caption(f"Market regime: {regime}")

    # ── 2. KEY METRICS ROW ───────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("ρ Now",      f"{corr_val:+.2f}")
    m2.metric("4w Change",  f"{corr_chg:+.2f}" if corr_chg else "—")
    m3.metric("Edge",       edge)
    m4.metric("Confidence", confidence.upper())
    m5.metric("Leader",     leader_label)

    # ── 3. CHARTS — rolling corr + gauge ─────────────────────────────────────
    chart_col, gauge_col = st.columns([3, 1])
    with chart_col:
        hist = corr_series(md, candidate, top_asset, lookback)
        if not hist.empty:
            st.plotly_chart(
                chart_corr_series(hist, threshold, candidate, top_asset, lookback),
                use_container_width=True, key="results_corr_series",
            )
    with gauge_col:
        st.plotly_chart(
            chart_gauge(corr_val, f"{candidate} vs {top_asset}"),
            use_container_width=True, key="results_gauge",
        )

    # ── 4. PORTFOLIO FIT + STORY + WHAT CHANGED ──────────────────────────────
    portfolio_fit = brief.get("portfolio_fit", {})
    if portfolio_fit:
        st.markdown(f"**Portfolio fit:** {portfolio_fit.get('summary', '')}")
        overlaps     = portfolio_fit.get("overlaps", [])
        diversifiers = portfolio_fit.get("diversifiers", [])
        fit_cols = st.columns(2)
        with fit_cols[0]:
            if overlaps:
                st.caption("Overlaps with: " + " · ".join(overlaps))
        with fit_cols[1]:
            if diversifiers:
                st.caption("Diversifies against: " + " · ".join(diversifiers))

    story = brief.get("story", "")
    if story:
        st.markdown(story)

    what_changed = brief.get("what_changed", [])
    if what_changed:
        st.markdown("**Key findings:**")
        for item in what_changed:
            st.markdown(f"**{item.get('finding', '')}** — {item.get('evidence', '')}")

    # ── 5. TABS ───────────────────────────────────────────────────────────────
    st.divider()
    tabs = st.tabs(["🎯 Should I add it?", "⚠️ Risks & Caveats"])

    with tabs[0]:
        last_na  = st.session_state.get("last_news_analysis")
        analysis = None
        if last_na and last_na.get("asset_a") == candidate:
            if last_na.get("source") == "web_search":
                analysis = last_na.get("web_research")
            else:
                analysis = last_na.get("analysis_a")

        if analysis and analysis.get("error"):
            st.warning(f"Research failed: {analysis['error']}")
            analysis = None

        if analysis:
            # Raw fallback (model returned unstructured text)
            if analysis.get("raw_analysis") and not analysis.get("recommendation"):
                st.markdown("**🌐 AI Research**")
                st.markdown(analysis["raw_analysis"][:2000])
            else:
                rec       = (analysis.get("recommendation") or "wait").lower()
                conviction = (analysis.get("conviction") or "medium").upper()
                rec_icon  = {"add": "🟢", "wait": "🟡", "avoid": "🔴"}.get(rec, "🔘")
                rec_label = {"add": "ADD", "wait": "WAIT — monitor first", "avoid": "AVOID"}.get(rec, rec.upper())

                st.markdown(f"### {rec_icon} {rec_label}")
                st.caption(f"Conviction: **{conviction}** · Source: {'🌐 Live web search' if last_na.get('source') == 'web_search' else '📰 NewsAPI'}")

                if analysis.get("headline_summary"):
                    st.markdown(f"**What's happening now:** {analysis['headline_summary']}")

                if analysis.get("sector_context"):
                    st.caption(f"Sector / macro: {analysis['sector_context']}")

                st.divider()
                inv_col, risk_col = st.columns(2)
                with inv_col:
                    st.markdown("**Investment case**")
                    st.markdown(analysis.get("investment_case") or "—")
                    if analysis.get("timing"):
                        st.caption(f"Timing: {analysis['timing']}")
                with risk_col:
                    st.markdown("**Key risks**")
                    for r in analysis.get("key_risks") or []:
                        st.markdown(f"- ⚠️ {r}")

                catalyst = analysis.get("catalyst_watch") or []
                if catalyst:
                    st.markdown("**Watch for:**")
                    for c in catalyst:
                        st.markdown(f"- 👁 {c}")

                headlines = analysis.get("recent_headlines") or []
                if headlines:
                    st.divider()
                    st.markdown("**Recent headlines**")
                    for h in headlines[:5]:
                        st.markdown(f"📰 {h}")
        else:
            st.info(
                f"Click the button above to get a live investment case for **{candidate}**. "
                "OpenAI provider uses live web search. Other providers need a NewsAPI key in the sidebar."
            )

    with tabs[1]:
        impl_col, cav_col = st.columns(2)
        with impl_col:
            st.markdown("**Implications**")
            for imp in brief.get("practical_implications", []):
                st.markdown(f"- ✅ {imp}")
        with cav_col:
            st.markdown("**Caveats**")
            for cav in brief.get("caveats", []):
                st.markdown(f"- ⚠️ {cav}")
        notes = reliability.get("notes", [])
        if notes:
            with st.expander("Reliability details"):
                for n in notes:
                    st.markdown(f"- {n}")

    # ── 6. PAIR EXPLANATION ───────────────────────────────────────────────────
    st.divider()
    pair_exp_key = f"pair_exp_{candidate}_{top_asset}"
    if pair_exp_key not in st.session_state:
        with st.spinner(f"Generating plain-English explanation for {candidate} vs {top_asset}…"):
            st.session_state[pair_exp_key] = generate_pair_explanation(
                candidate, top_asset, corr_val,
                provider, model, api_key, custom_url,
            )
    pair_exp = st.session_state.get(pair_exp_key, "")
    if pair_exp:
        st.markdown("**What does this relationship mean in practice?**")
        st.markdown(pair_exp)

    # ── 7. ANALYSIS LOG ───────────────────────────────────────────────────────
    st.divider()
    st.subheader("Analysis log")
    st.caption("Track what you've evaluated this session. Add your own notes.")

    log   = st.session_state["analysis_log"]
    entry = {
        "Candidate":      candidate,
        "Portfolio":      ", ".join(portfolio_ok),
        "Confidence":     confidence.upper(),
        "Strongest link": f"{top_asset} ρ={top_corr:+.2f}" if top_asset else "—",
        "Leader":         leader_label,
        "Notes":          "",
    }
    if not log or log[-1]["Candidate"] != candidate or log[-1]["Portfolio"] != entry["Portfolio"]:
        log.append(entry)
        st.session_state["analysis_log"] = log[-10:]

    edited = st.data_editor(
        pd.DataFrame(st.session_state["analysis_log"]),
        use_container_width=True,
        hide_index=True,
        disabled=["Candidate", "Portfolio", "Confidence", "Strongest link", "Leader"],
        column_config={"Notes": st.column_config.TextColumn("Notes", max_chars=200)},
        key="log_editor",
    )
    st.session_state["analysis_log"] = edited.to_dict("records")

    if st.button("Clear log"):
        st.session_state["analysis_log"] = []
        st.rerun()


if __name__ == "__main__":
    main()
