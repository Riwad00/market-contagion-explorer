from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform


@dataclass(frozen=True)
class WindowStats:
    end_date: pd.Timestamp
    lookback: int
    returns_window: pd.DataFrame
    corr: pd.DataFrame
    vol_annualized: pd.Series


def window_slice(returns: pd.DataFrame, end_date: pd.Timestamp, lookback: int) -> pd.DataFrame:
    w = returns.loc[:end_date].tail(int(lookback))
    return w


def compute_window_stats(returns: pd.DataFrame, end_date: pd.Timestamp, lookback: int) -> WindowStats:
    w = window_slice(returns, end_date, lookback)

    # pairwise corr; keep columns ordering stable
    corr = w.corr(min_periods=max(10, int(0.6 * lookback)))
    vol = w.std(skipna=True) * np.sqrt(252.0)

    return WindowStats(end_date=end_date, lookback=lookback, returns_window=w, corr=corr, vol_annualized=vol)


def corr_cluster_order(corr: pd.DataFrame) -> list[str]:
    """
    Hierarchical clustering order to place similar assets adjacent on the circle.
    Distance = 1 - corr (so negatively correlated assets are far apart).
    """
    if corr is None or corr.empty or len(corr) <= 2:
        return list(corr.index) if corr is not None else []

    c = corr.copy().fillna(0.0).clip(-1.0, 1.0)
    dist = 1.0 - c
    np.fill_diagonal(dist.values, 0.0)

    condensed = squareform(dist.values, checks=False)
    Z = linkage(condensed, method="average")
    order_idx = leaves_list(Z)
    return c.index.to_numpy()[order_idx].tolist()


def pair_corr_at_anchor(
    returns: pd.DataFrame,
    asset_a: str,
    asset_b: str,
    end_date: pd.Timestamp,
    lookback: int,
) -> float | None:
    w = window_slice(returns[[asset_a, asset_b]], end_date, lookback).dropna()
    if len(w) < max(10, int(0.6 * lookback)):
        return None
    return float(w[asset_a].corr(w[asset_b]))


def lagged_correlations(
    returns_window: pd.DataFrame,
    asset_a: str,
    asset_b: str,
    max_lag: int = 5,
) -> dict[str, Any]:
    """
    Computes corr(A_t, B_{t+k}) for k=1..max_lag (A leads B)
    and corr(B_t, A_{t+k}) (B leads A).
    """
    df = returns_window[[asset_a, asset_b]].dropna().copy()
    out = {"a_leads_b": [], "b_leads_a": []}
    if len(df) < 15:
        return out

    a = df[asset_a]
    b = df[asset_b]

    for k in range(1, max_lag + 1):
        # A_t vs B_{t+k} -> align A with forward-shifted B
        corr_ab = a.corr(b.shift(-k))
        corr_ba = b.corr(a.shift(-k))
        out["a_leads_b"].append({"lag_days": k, "corr": None if pd.isna(corr_ab) else float(corr_ab)})
        out["b_leads_a"].append({"lag_days": k, "corr": None if pd.isna(corr_ba) else float(corr_ba)})

    return out


def extreme_move_overlap(
    returns_window: pd.DataFrame,
    asset_a: str,
    asset_b: str,
    top_n: int = 10,
) -> dict[str, Any]:
    df = returns_window[[asset_a, asset_b]].dropna().copy()
    if df.empty:
        return {"top_n": top_n, "same_direction": 0, "opposite_direction": 0, "overlap_days": 0}

    a = df[asset_a]
    b = df[asset_b]
    a_top = a.abs().nlargest(min(top_n, len(a))).index
    b_top = b.abs().nlargest(min(top_n, len(b))).index
    overlap = a_top.intersection(b_top)

    same = 0
    opp = 0
    for d in overlap:
        sa = np.sign(a.loc[d])
        sb = np.sign(b.loc[d])
        if sa == 0 or sb == 0:
            continue
        if sa == sb:
            same += 1
        else:
            opp += 1

    return {
        "top_n": top_n,
        "overlap_days": int(len(overlap)),
        "same_direction": int(same),
        "opposite_direction": int(opp),
    }


def bootstrap_corr_ci(
    returns_window: pd.DataFrame,
    asset_a: str,
    asset_b: str,
    *,
    n_boot: int = 400,
    seed: int = 7,
) -> dict[str, Any]:
    df = returns_window[[asset_a, asset_b]].dropna().copy()
    n = len(df)
    if n < 30:
        return {"n": int(n), "n_boot": int(n_boot), "ci_95": None, "note": "sample_too_small"}

    rng = np.random.default_rng(seed)
    a = df[asset_a].to_numpy()
    b = df[asset_b].to_numpy()

    cors = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        cors[i] = np.corrcoef(a[idx], b[idx])[0, 1]

    lo, hi = np.nanpercentile(cors, [2.5, 97.5])
    return {"n": int(n), "n_boot": int(n_boot), "ci_95": [float(lo), float(hi)], "note": "ok"}


def window_sensitivity(
    returns: pd.DataFrame,
    asset_a: str,
    asset_b: str,
    end_date: pd.Timestamp,
    windows: list[int],
    threshold: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for w in windows:
        c = pair_corr_at_anchor(returns, asset_a, asset_b, end_date, w)
        connected = None if c is None else (abs(c) >= threshold)
        rows.append(
            {
                "lookback": int(w),
                "corr": c,
                "edge_status": None if connected is None else ("connected" if connected else "not_connected"),
            }
        )
    return rows

