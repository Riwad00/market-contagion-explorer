from __future__ import annotations

from datetime import date
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data import generate_sample_market_data, load_market_data
from src.llm import LLMError, generate_ai_brief
from src.metrics import (
    bootstrap_corr_ci,
    compute_window_stats,
    corr_cluster_order,
    extreme_move_overlap,
    lagged_correlations,
    pair_corr_at_anchor,
    window_sensitivity,
)
from src.network_viz import build_network_figure
from src.utils import clean_ticker, dedupe_preserve_order, is_plausible_ticker


DEFAULT_UNIVERSE = [
    # ETFs
    "SPY",
    "QQQ",
    "IWM",
    "IEF",
    "HYG",
    "GLD",
    "VGK",
    # Stocks (US + EU)
    "AAPL",
    "MSFT",
    "TSLA",
    "SAP.DE",
    "ASML.AS",
]

FALLBACK_EU = ["AIR.PA", "SIE.DE"]


def _edge_count(corr: pd.DataFrame, threshold: float) -> int:
    if corr is None or corr.empty:
        return 0
    m = corr.to_numpy()
    n = m.shape[0]
    cnt = 0
    for i in range(n):
        for j in range(i + 1, n):
            v = m[i, j]
            if np.isfinite(v) and abs(v) >= threshold:
                cnt += 1
    return int(cnt)


def _pair_weekly_corr_series(
    returns: pd.DataFrame,
    anchors: list[pd.Timestamp],
    asset_a: str,
    asset_b: str,
    lookback: int,
) -> pd.DataFrame:
    rows = []
    for a in anchors:
        c = pair_corr_at_anchor(returns, asset_a, asset_b, a, lookback)
        rows.append({"week": a, "corr": c})
    s = pd.DataFrame(rows).dropna()
    return s


def _format_week_label(ts: pd.Timestamp) -> str:
    return ts.date().isoformat()


def _avg_abs_corr_offdiag(corr: pd.DataFrame) -> float | None:
    if corr is None or corr.empty or corr.shape[0] < 2:
        return None
    m = corr.to_numpy(dtype=float)
    n = m.shape[0]
    vals = []
    for i in range(n):
        for j in range(i + 1, n):
            v = m[i, j]
            if np.isfinite(v):
                vals.append(abs(float(v)))
    return float(np.mean(vals)) if vals else None


def _edge_count_from_corr(corr: pd.DataFrame, threshold: float) -> int:
    if corr is None or corr.empty:
        return 0
    m = corr.to_numpy(dtype=float)
    n = m.shape[0]
    cnt = 0
    for i in range(n):
        for j in range(i + 1, n):
            v = m[i, j]
            if np.isfinite(v) and abs(float(v)) >= threshold:
                cnt += 1
    return int(cnt)


def _top_pair_by_abs_corr(corr: pd.DataFrame) -> tuple[str, str, float] | None:
    if corr is None or corr.empty or corr.shape[0] < 2:
        return None
    tickers = list(corr.index)
    best = None
    best_abs = -1.0
    for i, a in enumerate(tickers):
        for j in range(i + 1, len(tickers)):
            b = tickers[j]
            v = corr.loc[a, b]
            if pd.isna(v):
                continue
            av = abs(float(v))
            if av > best_abs:
                best_abs = av
                best = (a, b, float(v))
    return best


def _top_pair_by_abs_change(corr_now: pd.DataFrame, corr_then: pd.DataFrame) -> tuple[str, str, float] | None:
    if corr_now is None or corr_now.empty or corr_now.shape[0] < 2:
        return None
    tickers = [t for t in corr_now.index if t in corr_then.index]
    if len(tickers) < 2:
        return None
    best = None
    best_abs = -1.0
    for i, a in enumerate(tickers):
        for j in range(i + 1, len(tickers)):
            b = tickers[j]
            v0 = corr_now.loc[a, b] if (a in corr_now.index and b in corr_now.columns) else np.nan
            v1 = corr_then.loc[a, b] if (a in corr_then.index and b in corr_then.columns) else np.nan
            if pd.isna(v0) or pd.isna(v1):
                continue
            d = float(v0 - v1)
            ad = abs(d)
            if ad > best_abs:
                best_abs = ad
                best = (a, b, d)
    return best


def _corr_pairs_change_table(corr_now: pd.DataFrame, corr_then: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    rows = []
    tickers = list(corr_now.index)
    for i, a in enumerate(tickers):
        for j in range(i + 1, len(tickers)):
            b = tickers[j]
            v0 = corr_now.loc[a, b]
            v1 = corr_then.loc[a, b] if (a in corr_then.index and b in corr_then.columns) else np.nan
            if pd.isna(v0) or pd.isna(v1):
                continue
            d = float(v0 - v1)
            rows.append({"a": a, "b": b, "corr_then": float(v1), "corr_now": float(v0), "delta": d, "abs_delta": abs(d)})
    df = pd.DataFrame(rows).sort_values("abs_delta", ascending=False).head(int(top_k))
    if not df.empty:
        df = df.drop(columns=["abs_delta"])
    return df


def _build_pair_evidence_json(
    *,
    asset_a: str,
    asset_b: str,
    asof_week: pd.Timestamp,
    lookback: int,
    history_months: int,
    threshold: float,
    returns: pd.DataFrame,
    anchors: list[pd.Timestamp],
    universe: list[str],
) -> dict:
    stats = compute_window_stats(returns[[asset_a, asset_b]], asof_week, lookback)
    corr_now = stats.corr.loc[asset_a, asset_b] if (asset_a in stats.corr.index and asset_b in stats.corr.columns) else np.nan
    corr_now = None if pd.isna(corr_now) else float(corr_now)

    idx = anchors.index(asof_week) if asof_week in anchors else None
    corr_4w = None
    if idx is not None and idx >= 4:
        corr_4w = pair_corr_at_anchor(returns, asset_a, asset_b, anchors[idx - 4], lookback)

    corr_change = None
    if corr_now is not None and corr_4w is not None:
        corr_change = float(corr_now - corr_4w)

    edge_status = None
    if corr_now is not None:
        edge_status = "connected" if abs(corr_now) >= threshold else "not_connected"

    lag = lagged_correlations(stats.returns_window, asset_a, asset_b, max_lag=5)
    overlap = extreme_move_overlap(stats.returns_window, asset_a, asset_b, top_n=10)
    sens = window_sensitivity(returns, asset_a, asset_b, asof_week, [20, 60, 120], threshold)
    ci = bootstrap_corr_ci(stats.returns_window, asset_a, asset_b, n_boot=400)

    missing = stats.returns_window[[asset_a, asset_b]].isna().mean().to_dict()
    n_effective = int(stats.returns_window[[asset_a, asset_b]].dropna().shape[0])

    return {
        "universe": list(universe),
        "pair": {"asset_a": asset_a, "asset_b": asset_b},
        "time_context": {
            "asof_week": _format_week_label(asof_week),
            "lookback_window_trading_days": int(lookback),
            "history_months": int(history_months),
        },
        "relationship": {
            "corr_now": corr_now,
            "corr_4w_ago": corr_4w,
            "corr_change_vs_4w": corr_change,
            "edge_threshold": float(threshold),
            "edge_status": edge_status,
        },
        "lead_lag": lag,
        "extreme_move_overlap": overlap,
        "stability": {
            "window_sensitivity": sens,
            "bootstrap_ci_95": ci,
            "missing_frac_in_window": missing,
            "effective_sample_size": n_effective,
        },
    }


def main() -> None:
    st.set_page_config(page_title="Contagion Explorer", layout="wide")

    st.title("Contagion Explorer")
    st.caption(
        "Evidence-based cross-market connectivity (EU + US) using live market data. "
        "Correlations, lead‑lag patterns, and uncertainty checks — **no causation claims**."
    )

    with st.expander("Method summary (read this first)", expanded=False):
        st.markdown(
            """
- **Weekly snapshots**: each point is the *last trading day of the week* for interpretability.
- **Rolling window**: correlations/volatility are computed on trailing **20/60/120** trading days ending on the selected week.
- **Connectivity**: we track how dense the correlation graph is over time (avg |corr| and threshold edge count).
- **Heatmap**: we reorder the correlation matrix by hierarchical clustering to reveal groups.
- **Lead‑lag**: lagged correlations (1–5 days) show *predictive association patterns only* (not causality).
- **Reliability**: window sensitivity + **bootstrap 95% CI** + missing data warnings are always shown.
            """
        )

    # --------------------
    # Sidebar controls (fast access)
    # --------------------
    st.sidebar.header("Controls")
    months = st.sidebar.radio("History length", options=[12, 24], index=0, horizontal=True)
    # Global defaults used across tabs; controlled in the Overview (pair-first).
    if "global_lookback" not in st.session_state:
        st.session_state["global_lookback"] = 60
    if "global_threshold" not in st.session_state:
        st.session_state["global_threshold"] = 0.60
    lookback = int(st.session_state["global_lookback"])
    threshold = float(st.session_state["global_threshold"])
    cap = 18
    st.sidebar.caption(f"Ticker cap: **{cap}** (to keep charts readable).")

    st.sidebar.subheader("Universe")
    add_raw = st.sidebar.text_input(
        "Add tickers (comma-separated)",
        value="",
        help="Validated for plausible format (e.g., SAP.DE, ASML.AS). Data availability is checked on download.",
    )

    base = DEFAULT_UNIVERSE[:]
    added = []
    if add_raw.strip():
        for t in [x.strip() for x in add_raw.split(",")]:
            if t and is_plausible_ticker(t):
                added.append(clean_ticker(t))

    tickers = dedupe_preserve_order(base + added)[: int(cap)]

    st.sidebar.subheader("Data source")
    # Apply any pending provider switch BEFORE creating the selectbox widget.
    if "_pending_live_provider" in st.session_state:
        st.session_state["live_provider"] = st.session_state.pop("_pending_live_provider")

    provider_options = ["Yahoo (default, yfinance)", "Stooq (backup)"]
    provider = st.sidebar.selectbox(
        "Live data provider",
        options=provider_options,
        index=0,
        help="Stooq is live daily closes without API keys. Yahoo (yfinance) fetches OHLCV from Yahoo Finance via an unofficial wrapper and may be blocked on some networks.",
        key="live_provider",
    )
    provider = st.session_state.get("live_provider", provider_options[0])
    provider_key = "yahoo" if str(provider).startswith("Yahoo") else "stooq"

    yahoo_refresh_clicked = False
    if provider_key == "yahoo":
        st.sidebar.caption("Yahoo can rate-limit/block; this app avoids spamming it during slider changes.")
        yahoo_refresh_clicked = st.sidebar.button("Refresh Yahoo data", type="secondary")
        if "yahoo_refresh_nonce" not in st.session_state:
            st.session_state["yahoo_refresh_nonce"] = 0
        if yahoo_refresh_clicked:
            st.session_state["yahoo_refresh_nonce"] += 1

    use_sample = st.sidebar.toggle(
        "Use sample data (offline demo)",
        value=bool(st.session_state.get("use_sample_data", False)),
        help="Synthetic fallback (not live). Keep OFF for the assignment unless your environment cannot access live data at all.",
    )
    st.session_state["use_sample_data"] = use_sample

    st.sidebar.caption("Tip: If Yahoo is blocked on your network, switch provider to **Stooq (default)**.")

    with st.spinner("Downloading prices and preparing weekly anchors…"):
        if use_sample:
            md = generate_sample_market_data(tickers, months=int(months))
        else:
            md = load_market_data(
                tickers,
                months=int(months),
                provider=provider_key,
                yahoo_refresh_nonce=int(st.session_state.get("yahoo_refresh_nonce", 0)),
            )

            # Automatic backup: if Yahoo fails on this network, use Stooq so the app still runs.
            used_backup = False
            if provider_key == "yahoo" and (not md.tickers_ok or md.prices.empty or md.returns.empty or not md.weekly_anchors):
                md = load_market_data(tickers, months=int(months), provider="stooq")
                used_backup = True
            st.session_state["used_stooq_backup"] = bool(used_backup)

    if st.session_state.get("used_stooq_backup"):
        st.warning(
            "Yahoo (yfinance) failed to load on this run (network blocking / throttling). "
            "Using **Stooq backup** so the app remains functional.",
            icon="⚠️",
        )

    if use_sample:
        st.warning(
            "Using **sample data (offline demo)**. This is synthetic data (not live). Turn this OFF for the assignment.",
            icon="⚠️",
        )

    if md.tickers_failed and not use_sample:
        st.warning(
            "Some tickers failed to load and were excluded: "
            + ", ".join(md.tickers_failed)
        )
        if any(t.endswith(".DE") or t.endswith(".AS") for t in md.tickers_failed):
            st.info("If EU tickers fail, try fallback suggestions: " + ", ".join(FALLBACK_EU))

    if not md.tickers_ok or md.prices.empty or md.returns.empty or not md.weekly_anchors:
        if not use_sample:
            st.error(
                "Not enough data to compute contagion metrics. "
                "This often happens when the selected provider is blocked or rate-limited on your network."
            )
            st.info(
                "Try: switch provider to **Stooq (default)** (still live data) or change networks/VPN."
            )

            cta1, cta2 = st.columns([1, 3], vertical_alignment="center")
            with cta1:
                if st.button("Switch to Stooq and reload", type="primary"):
                    st.session_state["use_sample_data"] = False
                    st.session_state["_pending_live_provider"] = "Stooq (default)"
                    st.rerun()
            with cta2:
                st.caption(
                    "Tip: if your sidebar is hidden, expand it (top-left arrow) to find the data source toggle."
                )
        else:
            st.error("Not enough sample data to compute contagion metrics. Try fewer tickers or extend history.")
        st.stop()

    # Weekly anchors
    week_labels = [_format_week_label(x) for x in md.weekly_anchors]
    default_idx = len(md.weekly_anchors) - 1

    tabs = st.tabs(["Contagion Overview", "Pair Explorer", "AI Brief", "Accuracy & Method", "Settings"])

    # --------------------
    # Contagion Overview
    # --------------------
    with tabs[0]:
        st.subheader("Contagion Overview")

        with st.expander("How to read this", expanded=False):
            st.markdown(
                """
- **Pick Asset A/B**: the main chart shows weekly rolling correlation for this pair.
- **Threshold**: the dotted horizontal line marks the edge threshold ($|corr| \\ge$ threshold).
- **NOW / THEN**: use compare to see how the relationship map changed between two weeks.
- **Evidence & reliability**: lead‑lag, window sensitivity (20/60/120), and bootstrap CI help judge stability.
                """
            )

        # Initialize session state defaults (only if missing)
        if "overview_a" not in st.session_state:
            st.session_state["overview_a"] = md.tickers_ok[0]
        if "overview_b" not in st.session_state:
            st.session_state["overview_b"] = md.tickers_ok[1] if len(md.tickers_ok) > 1 else md.tickers_ok[0]
        if "overview_lookback" not in st.session_state:
            st.session_state["overview_lookback"] = int(st.session_state.get("global_lookback", 60))
        if "overview_threshold" not in st.session_state:
            st.session_state["overview_threshold"] = float(st.session_state.get("global_threshold", 0.60))
        if "overview_now_week" not in st.session_state:
            st.session_state["overview_now_week"] = week_labels[default_idx]
        if "overview_compare" not in st.session_state:
            st.session_state["overview_compare"] = False
        if "overview_then_week" not in st.session_state:
            st.session_state["overview_then_week"] = week_labels[max(0, default_idx - 8)]

        # Apply any pending pair suggestion BEFORE widgets instantiate (avoids StreamlitAPIException).
        if "_pending_overview_pair" in st.session_state:
            pending = st.session_state.pop("_pending_overview_pair") or {}
            pa = pending.get("a")
            pb = pending.get("b")
            if pa in md.tickers_ok and pb in md.tickers_ok and pa != pb:
                st.session_state["overview_a"] = pa
                st.session_state["overview_b"] = pb

        # Controls on-page (pair-first)
        c1, c2, c3, c4 = st.columns([1.1, 1.1, 1.0, 1.2], vertical_alignment="bottom")
        with c1:
            a = st.selectbox("Asset A", options=md.tickers_ok, key="overview_a")
        with c2:
            b_opts = [t for t in md.tickers_ok if t != a]
            # Keep prior B if valid, else first option.
            if st.session_state.get("overview_b") not in b_opts and b_opts:
                st.session_state["overview_b"] = b_opts[0]
            b = st.selectbox("Asset B", options=b_opts, key="overview_b")
        with c3:
            st.select_slider("Rolling window (days)", options=[20, 60, 120], key="overview_lookback")
        with c4:
            st.slider("Threshold (|corr| ≥)", 0.30, 0.90, step=0.05, key="overview_threshold")

        # Apply global lookback/threshold for other tabs
        st.session_state["global_lookback"] = int(st.session_state["overview_lookback"])
        st.session_state["global_threshold"] = float(st.session_state["overview_threshold"])
        lookback = int(st.session_state["global_lookback"])
        threshold = float(st.session_state["global_threshold"])

        s1, s2 = st.columns([1, 1], vertical_alignment="center")
        with s1:
            if st.button("Suggest most correlated pair (Now)"):
                stats_now = compute_window_stats(md.returns[md.tickers_ok], md.weekly_anchors[week_labels.index(st.session_state["overview_now_week"])], lookback)
                corr_now = stats_now.corr.reindex(index=md.tickers_ok, columns=md.tickers_ok)
                best = _top_pair_by_abs_corr(corr_now)
                if best:
                    st.session_state["_pending_overview_pair"] = {"a": best[0], "b": best[1]}
                    st.rerun()
        with s2:
            if st.button("Suggest biggest change vs 4w"):
                now_idx = week_labels.index(st.session_state["overview_now_week"])
                if now_idx < 4:
                    st.warning("Need at least 4 prior weeks to compute the change.")
                else:
                    now_week = md.weekly_anchors[now_idx]
                    then_week = md.weekly_anchors[now_idx - 4]
                    corr_now = compute_window_stats(md.returns[md.tickers_ok], now_week, lookback).corr.reindex(index=md.tickers_ok, columns=md.tickers_ok)
                    corr_then = compute_window_stats(md.returns[md.tickers_ok], then_week, lookback).corr.reindex(index=md.tickers_ok, columns=md.tickers_ok)
                    best = _top_pair_by_abs_change(corr_now, corr_then)
                    if best:
                        st.session_state["_pending_overview_pair"] = {"a": best[0], "b": best[1]}
                        st.rerun()

        st.divider()

        # Week selectors on-page (not sidebar)
        w1, w2, w3 = st.columns([1.2, 0.8, 1.2], vertical_alignment="bottom")
        with w1:
            now_label = st.select_slider("Now week", options=week_labels, key="overview_now_week")
        with w2:
            st.toggle("Compare", key="overview_compare")
        with w3:
            then_label = None
            if st.session_state["overview_compare"]:
                then_label = st.select_slider("Then week", options=week_labels, key="overview_then_week")

        now_week = md.weekly_anchors[week_labels.index(now_label)]
        then_week = md.weekly_anchors[week_labels.index(then_label)] if (then_label and st.session_state["overview_compare"]) else None
        if then_week is not None and then_week >= now_week:
            st.warning("Pick a THEN week earlier than NOW.")
            then_week = None

        # Main chart: rolling correlation (weekly)
        hist = _pair_weekly_corr_series(md.returns, md.weekly_anchors, a, b, lookback)
        corr_now = pair_corr_at_anchor(md.returns, a, b, now_week, lookback)
        corr_ref = pair_corr_at_anchor(md.returns, a, b, then_week, lookback) if then_week is not None else (
            pair_corr_at_anchor(md.returns, a, b, md.weekly_anchors[week_labels.index(now_label) - 4], lookback) if week_labels.index(now_label) >= 4 else None
        )
        delta = None if (corr_now is None or corr_ref is None) else float(corr_now - corr_ref)

        m1, m2 = st.columns(2)
        m1.metric("corr (Now)", "n/a" if corr_now is None else f"{corr_now:+.2f}")
        m2.metric("Δcorr", "n/a" if delta is None else f"{delta:+.2f}")

        fig = go.Figure()
        if not hist.empty:
            fig.add_trace(go.Scatter(x=hist["week"], y=hist["corr"], mode="lines", line=dict(color="#2563EB", width=2), name="Rolling corr"))
        fig.add_hline(y=threshold, line_dash="dot", line_color="rgba(15,23,42,0.35)")
        fig.add_hline(y=-threshold, line_dash="dot", line_color="rgba(15,23,42,0.20)")
        fig.add_vline(x=now_week, line_dash="dot", line_color="rgba(37,99,235,0.45)")
        if then_week is not None:
            fig.add_vline(x=then_week, line_dash="dot", line_color="rgba(15,23,42,0.30)")
        fig.update_layout(
            template="plotly_white",
            height=420,
            margin=dict(l=10, r=10, t=50, b=10),
            title=f"Rolling weekly correlation: {a} vs {b} (window={lookback}d)",
            yaxis=dict(range=[-1, 1]),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Evidence & reliability (collapsed)
        with st.expander("Evidence & reliability", expanded=False):
            stats_pair = compute_window_stats(md.returns[[a, b]], now_week, lookback)
            lag = lagged_correlations(stats_pair.returns_window, a, b, max_lag=5)
            lag_df = pd.DataFrame(
                {
                    "lag_days": [x["lag_days"] for x in lag["a_leads_b"]],
                    f"{a} leads {b}": [x["corr"] for x in lag["a_leads_b"]],
                    f"{b} leads {a}": [x["corr"] for x in lag["b_leads_a"]],
                }
            )
            st.markdown("**Lead‑lag (1–5 days)**")
            st.dataframe(lag_df, use_container_width=True, hide_index=True)

            st.markdown("**Window sensitivity (20/60/120)**")
            sens_rows = window_sensitivity(md.returns, a, b, now_week, [20, 60, 120], threshold)
            st.dataframe(pd.DataFrame(sens_rows), use_container_width=True, hide_index=True)

            st.markdown("**Bootstrap 95% CI (corr Now)**")
            ci = bootstrap_corr_ci(stats_pair.returns_window, a, b, n_boot=400)
            if ci["ci_95"] is None:
                st.warning("Bootstrap CI: sample too small for a stable estimate.")
            else:
                lo, hi = ci["ci_95"]
                st.success(f"[{lo:.2f}, {hi:.2f}] (n={ci['n']}, boot={ci['n_boot']})")

        st.divider()

        # Heatmap below main chart (exploration)
        st.subheader("Relationship map (clustered heatmap)")
        show_abs = st.toggle("Show abs(corr)", value=False, key="overview_heatmap_abs")

        stats_now = compute_window_stats(md.returns[md.tickers_ok], now_week, lookback)
        corr_mat_now = stats_now.corr.reindex(index=md.tickers_ok, columns=md.tickers_ok)
        order = corr_cluster_order(corr_mat_now)
        cm_now = corr_mat_now.reindex(index=order, columns=order)
        z_now = cm_now.abs().to_numpy() if show_abs else cm_now.to_numpy()
        zmin, zmax, zmid = (0.0, 1.0, None) if show_abs else (-1.0, 1.0, 0.0)
        colorscale = "Viridis" if show_abs else "RdBu"

        def heatmap_fig(z, title: str):
            f = go.Figure(
                data=go.Heatmap(
                    z=z,
                    x=order,
                    y=order,
                    zmin=zmin,
                    zmax=zmax,
                    zmid=zmid,
                    colorscale=colorscale,
                    hovertemplate="x=%{x}<br>y=%{y}<br>value=%{z:.2f}<extra></extra>",
                )
            )
            f.update_layout(template="plotly_white", height=520, margin=dict(l=10, r=10, t=50, b=10), title=title)
            return f

        if then_week is not None:
            stats_then = compute_window_stats(md.returns[md.tickers_ok], then_week, lookback)
            corr_mat_then = stats_then.corr.reindex(index=md.tickers_ok, columns=md.tickers_ok).reindex(index=order, columns=order)
            cm_then = corr_mat_then.abs() if show_abs else corr_mat_then
            cm_now_plot = cm_now.abs() if show_abs else cm_now
            diff = (cm_now_plot - cm_then).to_numpy()

            h1, h2, h3 = st.columns(3)
            with h1:
                st.plotly_chart(heatmap_fig(z_now, f"Now ({now_label})"), use_container_width=True)
            with h2:
                st.plotly_chart(heatmap_fig(cm_then.to_numpy(), f"Then ({then_label})"), use_container_width=True)
            with h3:
                fd = go.Figure(
                    data=go.Heatmap(
                        z=diff,
                        x=order,
                        y=order,
                        zmin=-1.0,
                        zmax=1.0,
                        zmid=0.0,
                        colorscale="RdBu",
                        hovertemplate="x=%{x}<br>y=%{y}<br>Δ=%{z:.2f}<extra></extra>",
                    )
                )
                fd.update_layout(template="plotly_white", height=520, margin=dict(l=10, r=10, t=50, b=10), title="Now - Then")
                st.plotly_chart(fd, use_container_width=True)
        else:
            st.plotly_chart(heatmap_fig(z_now, f"Now ({now_label})"), use_container_width=True)

        # Optional legacy network
        with st.expander("Optional: correlation network (legacy view)", expanded=False):
            latest_prices = md.prices.loc[:now_week].tail(1).T.iloc[:, 0] if not md.prices.loc[:now_week].empty else None
            fig_net = build_network_figure(
                corr=corr_mat_now,
                vol_annualized=stats_now.vol_annualized,
                latest_prices=latest_prices,
                order=order,
                threshold=float(threshold),
                title=f"Correlation Network (legacy) — {now_label}",
            )
            st.plotly_chart(fig_net, use_container_width=True)

    # --------------------
    # Pair Explorer
    # --------------------
    with tabs[1]:
        st.subheader("Explain a Link (pair evidence)")
        left, right = st.columns([1, 2], vertical_alignment="top")

        with left:
            a = st.selectbox("Asset A", options=md.tickers_ok, index=0)
            b = st.selectbox("Asset B", options=[t for t in md.tickers_ok if t != a], index=0)
            asof_label = st.select_slider("As-of week", options=week_labels, value=week_labels[default_idx])
            asof_week = md.weekly_anchors[week_labels.index(asof_label)]

        evidence_json = _build_pair_evidence_json(
            asset_a=a,
            asset_b=b,
            asof_week=asof_week,
            lookback=int(lookback),
            history_months=int(months),
            threshold=float(threshold),
            returns=md.returns,
            anchors=md.weekly_anchors,
            universe=md.tickers_ok,
        )

        rel = evidence_json["relationship"]
        stab = evidence_json["stability"]

        with right:
            c1, c2, c3 = st.columns(3)
            c1.metric("Corr now", "n/a" if rel["corr_now"] is None else f'{rel["corr_now"]:.2f}')
            c2.metric("Corr 4w ago", "n/a" if rel["corr_4w_ago"] is None else f'{rel["corr_4w_ago"]:.2f}')
            c3.metric("Δ vs 4w", "n/a" if rel["corr_change_vs_4w"] is None else f'{rel["corr_change_vs_4w"]:+.2f}')

            st.caption(
                f"Edge status at threshold {threshold:.2f}: **{rel['edge_status'] or 'n/a'}**. "
                "This is association only — no causation claims."
            )

            # Rolling correlation timeline (weekly)
            hist = _pair_weekly_corr_series(md.returns, md.weekly_anchors, a, b, int(lookback))
            if not hist.empty:
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=hist["week"], y=hist["corr"], mode="lines", line=dict(color="#2563EB", width=2)))
                fig2.add_hline(y=threshold, line_dash="dot", line_color="rgba(37,99,235,0.35)")
                fig2.add_hline(y=-threshold, line_dash="dot", line_color="rgba(239,68,68,0.35)")
                fig2.update_layout(
                    title="Weekly rolling correlation (pair)",
                    height=280,
                    margin=dict(l=10, r=10, t=45, b=10),
                    yaxis=dict(range=[-1, 1]),
                )
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("Not enough aligned data to plot the correlation timeline for this pair.")

        st.divider()
        cL, cM, cR = st.columns([1, 1, 1], vertical_alignment="top")

        with cL:
            st.subheader("Lead‑lag (1–5 days)")
            lag = evidence_json["lead_lag"]
            lag_df = pd.DataFrame(
                {
                    "lag_days": [x["lag_days"] for x in lag["a_leads_b"]],
                    f"{a} leads {b}": [x["corr"] for x in lag["a_leads_b"]],
                    f"{b} leads {a}": [x["corr"] for x in lag["b_leads_a"]],
                }
            )
            st.dataframe(lag_df, use_container_width=True, hide_index=True)
            st.caption("Interpretation: higher absolute values may indicate a *predictive association pattern*, not causality.")

        with cM:
            st.subheader("Extreme-move overlap (current window)")
            ov = evidence_json["extreme_move_overlap"]
            st.metric("Overlap days", ov["overlap_days"])
            st.metric("Same-direction overlap", ov["same_direction"])
            st.metric("Opposite-direction overlap", ov["opposite_direction"])
            st.caption("Computed from each asset’s top-10 absolute return days within the current window.")

        with cR:
            st.subheader("Reliability checks (mandatory)")
            st.markdown("**Window sensitivity (20/60/120):**")
            st.dataframe(pd.DataFrame(stab["window_sensitivity"]), use_container_width=True, hide_index=True)

            ci = stab["bootstrap_ci_95"]
            if ci["ci_95"] is None:
                st.warning("Bootstrap CI: sample too small for a stable estimate.")
            else:
                lo, hi = ci["ci_95"]
                st.success(f"Bootstrap 95% CI for corr: [{lo:.2f}, {hi:.2f}] (n={ci['n']}, boot={ci['n_boot']})")

            miss = stab["missing_frac_in_window"]
            st.caption(
                f"Effective sample size: **{stab['effective_sample_size']}**. "
                f"Missing fraction in window — {a}: {miss.get(a, 0):.0%}, {b}: {miss.get(b, 0):.0%}."
            )

        st.session_state["last_pair_evidence_json"] = evidence_json

    # --------------------
    # AI Brief
    # --------------------
    with tabs[2]:
        st.subheader("AI Brief (grounded, optional)")
        st.caption("The AI uses ONLY the computed evidence JSON. No outside knowledge, no macro/news invented.")

        evidence_json = st.session_state.get("last_pair_evidence_json")
        if not evidence_json:
            st.info("Go to **Pair Explorer** first and select a pair to generate an AI brief.")
            st.stop()

        with st.expander("Evidence JSON (what the model receives)", expanded=False):
            st.json(evidence_json)

        provider = st.session_state.get("llm_provider", "OpenAI")
        model = st.session_state.get("llm_model", "gpt-4.1-mini")
        api_key = st.session_state.get("llm_api_key", "")

        if not api_key:
            st.warning("AI is disabled until you provide an API key in the **Settings** tab.")
            st.stop()

        if st.button("Generate AI Brief", type="primary"):
            with st.spinner("Calling LLM and validating strict JSON output…"):
                try:
                    brief = generate_ai_brief(
                        evidence_json=evidence_json,
                        provider=provider,
                        model=model,
                        api_key=api_key,
                    )
                    st.success("AI Brief generated (validated JSON).")
                    st.session_state["last_ai_brief"] = brief
                except LLMError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"Unexpected error: {type(e).__name__}: {e}")

        brief = st.session_state.get("last_ai_brief")
        if brief:
            c1, c2 = st.columns([1, 2], vertical_alignment="top")
            with c1:
                st.markdown("**Regime**")
                st.write(brief["regime_summary"]["label"])
                st.markdown("**Confidence**")
                st.write(brief["reliability"]["overall_confidence"])
                st.markdown("**Edge status**")
                st.write(brief["relationship_now"]["edge_status"])
            with c2:
                st.markdown("**One‑sentence summary**")
                st.write(brief["regime_summary"]["one_sentence"])
                st.markdown("**What changed**")
                st.dataframe(pd.DataFrame(brief["what_changed"]), use_container_width=True, hide_index=True)

            st.markdown("**Lead‑lag**")
            st.write(brief["lead_lag"]["summary"])
            st.write(f"Likely leader: **{brief['lead_lag']['likely_leader']}**")
            st.dataframe(pd.DataFrame({"supporting_evidence": brief["lead_lag"]["supporting_evidence"]}), hide_index=True)

            st.markdown("**Reliability notes**")
            st.dataframe(pd.DataFrame({"notes": brief["reliability"]["reliability_notes"]}), hide_index=True)

            st.markdown("**Practical implications (exactly 3)**")
            st.dataframe(pd.DataFrame({"item": brief["practical_implications"]}), hide_index=True)

            st.markdown("**Caveats (exactly 3)**")
            st.dataframe(pd.DataFrame({"item": brief["caveats"]}), hide_index=True)

    # --------------------
    # Accuracy & Method
    # --------------------
    with tabs[3]:
        st.subheader("Accuracy & Method (mandatory)")
        st.markdown(
            """
**Key limitations to keep visible:**
- **Correlation instability / non‑stationarity**: relationships can change quickly; signals may not persist.
- **Window sensitivity**: 20/60/120 day windows can disagree; this app shows that explicitly.
- **Bootstrap CI**: wide intervals mean high uncertainty.
- **Missing data & alignment**: holidays / ticker gaps can distort estimates; the app warns when sample size is small.
- **Lead‑lag does not imply causality**: lag correlation is a predictive association pattern only.
            """
        )

        st.markdown("### Sanity check widget: connectivity vs window")
        sel_week_label = st.select_slider("Week for sanity check", options=week_labels, value=week_labels[default_idx], key="sanity_week")
        sel_week = md.weekly_anchors[week_labels.index(sel_week_label)]
        rows = []
        for w in [20, 60, 120]:
            stats = compute_window_stats(md.returns[md.tickers_ok], sel_week, w)
            rows.append({"window": w, "edge_count": _edge_count(stats.corr, float(threshold))})
        df = pd.DataFrame(rows)
        st.dataframe(df, hide_index=True, use_container_width=True)

        spread = int(df["edge_count"].max() - df["edge_count"].min())
        if spread >= 10:
            st.warning("Connectivity changes a lot across windows here → treat conclusions as **unstable**.")
        else:
            st.info("Connectivity is relatively consistent across windows here (still not a guarantee of stability).")

        st.markdown("### How bootstrap CI is computed")
        st.markdown(
            "Within the selected trailing window, we resample daily return pairs with replacement "
            "and recompute correlation repeatedly to form a 95% confidence interval."
        )

    # --------------------
    # Settings
    # --------------------
    with tabs[4]:
        st.subheader("Settings (LLM + data controls)")

        st.markdown("### LLM provider (optional)")
        provider = st.selectbox("Provider", options=["OpenAI"], index=0)
        model = st.text_input("Model", value=st.session_state.get("llm_model", "gpt-4.1-mini"))
        api_key = st.text_input(
            "API key",
            value=st.session_state.get("llm_api_key", os.getenv("OPENAI_API_KEY", "")),
            type="password",
            help="Stored only in session state. Never written to disk.",
        )
        st.session_state["llm_provider"] = provider
        st.session_state["llm_model"] = model
        st.session_state["llm_api_key"] = api_key

        st.markdown("### Data controls (read-only summary)")
        st.write(
            {
                "history_months": int(months),
                "rolling_window_days": int(lookback),
                "edge_threshold": float(threshold),
                "tickers_requested": md.tickers_requested,
                "tickers_ok": md.tickers_ok,
                "asof": date.today().isoformat(),
            }
        )

        st.markdown("### Security notes")
        st.markdown(
            """
- No API keys are hardcoded or committed.
- If you deploy to Streamlit Cloud, set secrets in the app settings (optional).
- Quantitative features work without any API key.
            """
        )


if __name__ == "__main__":
    main()

