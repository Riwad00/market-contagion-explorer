from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def _normalize_radius(vol: pd.Series, r_min: float = 0.25, r_max: float = 1.0) -> pd.Series:
    v = vol.copy()
    v = v.replace([np.inf, -np.inf], np.nan).astype(float)
    if v.dropna().empty:
        return pd.Series(index=v.index, data=(r_min + r_max) / 2.0)
    lo = float(v.min(skipna=True))
    hi = float(v.max(skipna=True))
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-12:
        return pd.Series(index=v.index, data=(r_min + r_max) / 2.0)
    scaled = (v - lo) / (hi - lo)
    return r_min + scaled * (r_max - r_min)


def radial_layout_coords(order: list[str], radius_by_ticker: pd.Series) -> pd.DataFrame:
    n = len(order)
    if n == 0:
        return pd.DataFrame(columns=["ticker", "x", "y", "r", "theta"])
    thetas = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    r = radius_by_ticker.reindex(order).to_numpy()
    x = r * np.cos(thetas)
    y = r * np.sin(thetas)
    return pd.DataFrame({"ticker": order, "x": x, "y": y, "r": r, "theta": thetas})


def build_network_figure(
    *,
    corr: pd.DataFrame,
    vol_annualized: pd.Series,
    latest_prices: pd.Series | None,
    order: list[str],
    threshold: float,
    title: str,
) -> go.Figure:
    c = corr.reindex(index=order, columns=order).copy()
    vol = vol_annualized.reindex(order).copy()

    radius = _normalize_radius(vol)
    coords = radial_layout_coords(order, radius)

    fig = go.Figure()

    # Edges
    for i, a in enumerate(order):
        for j in range(i + 1, len(order)):
            b = order[j]
            val = c.loc[a, b] if (a in c.index and b in c.columns) else np.nan
            if pd.isna(val):
                continue
            if abs(val) < threshold:
                continue

            xa = float(coords.loc[coords["ticker"] == a, "x"].iloc[0])
            ya = float(coords.loc[coords["ticker"] == a, "y"].iloc[0])
            xb = float(coords.loc[coords["ticker"] == b, "x"].iloc[0])
            yb = float(coords.loc[coords["ticker"] == b, "y"].iloc[0])

            width = 1.0 + 5.0 * float(abs(val))
            dash = "solid" if val >= 0 else "dot"
            color = "rgba(37,99,235,0.55)" if val >= 0 else "rgba(239,68,68,0.55)"

            fig.add_trace(
                go.Scatter(
                    x=[xa, xb],
                    y=[ya, yb],
                    mode="lines",
                    line=dict(width=width, color=color, dash=dash),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    # Nodes
    hover = []
    for t in order:
        v = vol.get(t, np.nan)
        p = latest_prices.get(t, np.nan) if latest_prices is not None else np.nan
        hv = "n/a" if pd.isna(v) else f"{float(v):.1%}"
        hp = "n/a" if pd.isna(p) else f"{float(p):.2f}"
        hover.append(f"<b>{t}</b><br>Vol (ann.): {hv}<br>Latest price: {hp}")

    fig.add_trace(
        go.Scatter(
            x=coords["x"],
            y=coords["y"],
            mode="markers+text",
            text=coords["ticker"],
            textposition="middle center",
            marker=dict(size=30, color="rgba(15,23,42,0.90)", line=dict(width=1, color="white")),
            hovertext=hover,
            hoverinfo="text",
            showlegend=False,
        )
    )

    fig.update_layout(
        title=title,
        height=650,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def _edge_traces(
    *,
    corr: pd.DataFrame,
    coords_by_ticker: dict[str, tuple[float, float]],
    order: list[str],
    threshold: float,
) -> list[go.Scatter]:
    traces: list[go.Scatter] = []
    for i, a in enumerate(order):
        for j in range(i + 1, len(order)):
            b = order[j]
            if a not in corr.index or b not in corr.columns:
                continue
            val = corr.loc[a, b]
            if pd.isna(val) or abs(val) < threshold:
                continue
            xa, ya = coords_by_ticker[a]
            xb, yb = coords_by_ticker[b]

            width = 1.0 + 5.0 * float(abs(val))
            dash = "solid" if val >= 0 else "dot"
            color = "rgba(37,99,235,0.55)" if val >= 0 else "rgba(239,68,68,0.55)"

            traces.append(
                go.Scatter(
                    x=[xa, xb],
                    y=[ya, yb],
                    mode="lines",
                    line=dict(width=width, color=color, dash=dash),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
    return traces


def build_animated_network_figure(
    *,
    snapshots: list[dict[str, Any]],
    order: list[str],
    threshold: float,
    title: str,
) -> go.Figure:
    """
    Plotly animated network:
    - Fixed angles from `order` for smooth animation
    - Per-frame radii update from volatility (normalized per frame)
    - Per-frame edges update from correlation thresholding

    snapshots item schema:
      {"week_label": str, "corr": DataFrame, "vol": Series, "latest_prices": Series|None}
    """
    if not snapshots:
        return go.Figure()

    def node_trace(vol: pd.Series, latest_prices: pd.Series | None):
        radius = _normalize_radius(vol.reindex(order))
        coords = radial_layout_coords(order, radius)
        coords_by = {r["ticker"]: (float(r["x"]), float(r["y"])) for _, r in coords.iterrows()}

        hover = []
        for t in order:
            v = vol.get(t, np.nan)
            p = latest_prices.get(t, np.nan) if latest_prices is not None else np.nan
            hv = "n/a" if pd.isna(v) else f"{float(v):.1%}"
            hp = "n/a" if pd.isna(p) else f"{float(p):.2f}"
            hover.append(f"<b>{t}</b><br>Vol (ann.): {hv}<br>Latest price: {hp}")

        trace = go.Scatter(
            x=coords["x"],
            y=coords["y"],
            mode="markers+text",
            text=coords["ticker"],
            textposition="middle center",
            marker=dict(size=30, color="rgba(15,23,42,0.90)", line=dict(width=1, color="white")),
            hovertext=hover,
            hoverinfo="text",
            showlegend=False,
        )
        return trace, coords_by

    # Initial frame
    s0 = snapshots[0]
    node0, coords0 = node_trace(s0["vol"], s0.get("latest_prices"))
    edges0 = _edge_traces(corr=s0["corr"], coords_by_ticker=coords0, order=order, threshold=threshold)

    fig = go.Figure(data=[*edges0, node0])

    frames = []
    for s in snapshots:
        node, coords = node_trace(s["vol"], s.get("latest_prices"))
        edges = _edge_traces(corr=s["corr"], coords_by_ticker=coords, order=order, threshold=threshold)
        frames.append(go.Frame(name=s["week_label"], data=[*edges, node]))

    fig.frames = frames

    steps = []
    for s in snapshots:
        steps.append(
            dict(
                method="animate",
                args=[[s["week_label"]], {"mode": "immediate", "frame": {"duration": 250, "redraw": True}, "transition": {"duration": 150}}],
                label=s["week_label"],
            )
        )

    fig.update_layout(
        title=title,
        height=680,
        margin=dict(l=20, r=20, t=80, b=40),
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                x=0.02,
                y=1.08,
                xanchor="left",
                yanchor="top",
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[None, {"fromcurrent": True, "frame": {"duration": 250, "redraw": True}, "transition": {"duration": 150}}],
                    ),
                    dict(label="Pause", method="animate", args=[[None], {"mode": "immediate", "frame": {"duration": 0}, "transition": {"duration": 0}}]),
                ],
            )
        ],
        sliders=[
            dict(
                active=0,
                y=0.02,
                x=0.06,
                len=0.9,
                pad={"t": 25, "b": 0},
                currentvalue={"prefix": "Week: "},
                steps=steps,
            )
        ],
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def compare_edge_changes(
    corr_now: pd.DataFrame,
    corr_then: pd.DataFrame,
    *,
    order: list[str],
    threshold: float,
) -> dict[str, Any]:
    def edge_set(c: pd.DataFrame) -> dict[tuple[str, str], float]:
        out: dict[tuple[str, str], float] = {}
        for i, a in enumerate(order):
            for j in range(i + 1, len(order)):
                b = order[j]
                if a not in c.index or b not in c.columns:
                    continue
                v = c.loc[a, b]
                if pd.isna(v) or abs(v) < threshold:
                    continue
                out[(a, b)] = float(v)
        return out

    e_now = edge_set(corr_now)
    e_then = edge_set(corr_then)

    now_keys = set(e_now.keys())
    then_keys = set(e_then.keys())

    added = sorted(list(now_keys - then_keys))
    removed = sorted(list(then_keys - now_keys))

    changed = []
    for k in sorted(list(now_keys & then_keys)):
        d = e_now[k] - e_then[k]
        changed.append({"pair": k, "corr_then": e_then[k], "corr_now": e_now[k], "delta": float(d)})
    changed_sorted = sorted(changed, key=lambda x: abs(x["delta"]), reverse=True)[:10]

    avg_abs_now = float(np.nanmean(np.abs(list(e_now.values())))) if e_now else 0.0
    avg_abs_then = float(np.nanmean(np.abs(list(e_then.values())))) if e_then else 0.0

    return {
        "edges_added": [{"a": a, "b": b, "corr_now": e_now[(a, b)]} for (a, b) in added[:12]],
        "edges_removed": [{"a": a, "b": b, "corr_then": e_then[(a, b)]} for (a, b) in removed[:12]],
        "top_corr_changes": [
            {
                "a": p[0],
                "b": p[1],
                "corr_then": x["corr_then"],
                "corr_now": x["corr_now"],
                "delta": x["delta"],
            }
            for x in changed_sorted
            for p in [x["pair"]]
        ],
        "edge_count_now": int(len(e_now)),
        "edge_count_then": int(len(e_then)),
        "avg_abs_corr_now": avg_abs_now,
        "avg_abs_corr_then": avg_abs_then,
        "avg_abs_corr_delta": float(avg_abs_now - avg_abs_then),
    }

