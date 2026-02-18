# Contagion Explorer (Streamlit Prototype)

“Contagion Explorer” is an interactive tool that monitors **cross-market connectivity** (“contagion”) across EU + US assets using **live market data (Yahoo Finance via `yfinance`)**.

It answers:
- Where is contagion building or fading over time?
- Which assets are connected strongly right now, and how does that change week to week?
- For any pair: what does the evidence say about co-movement and **lead‑lag predictive patterns** (no causation claims)?
- How reliable are these signals (window sensitivity, confidence intervals, missing data warnings)?

The app has two layers:
- **Quantitative evidence (deterministic)**: correlations, rolling windows, weekly snapshots, lead‑lag lag correlations, bootstrap CI, stability checks.
- **AI Brief (optional, user‑provided API key)**: an LLM generates a structured brief using **ONLY** the computed evidence JSON (no outside knowledge; no macro/news invented).

## Run locally

From the repo root:

```bash
cd assignment_1
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy (Streamlit Community Cloud)

- Set the app entry point to `assignment_1/app.py`
- Add dependencies from `assignment_1/requirements.txt`
- Do **not** commit secrets

## API keys (safe)

The AI Brief is **disabled by default**. To enable:
- Open the **Settings** tab in the app
- Paste an API key (stored **only in session state**, never written to disk)

If you want to use Streamlit secrets locally:
- Create `.streamlit/secrets.toml` (this repo ignores it)
- Example:

```toml
OPENAI_API_KEY="..."
```

## Method (Accuracy & limitations)

- **Correlations are unstable**: relationships can change quickly (non‑stationary).
- **Rolling windows**: 20/60/120 trading days produce different results; the app surfaces **window sensitivity** explicitly.
- **Weekly snapshots**: the network updates on last trading day of each week for interpretability.
- **Radial layout**: node **radius = rolling annualized volatility** (normalized per week); node **angle = hierarchical clustering order** to place similar assets adjacent.
- **Bootstrap confidence interval**: a 95% CI is computed for the current-window correlation to show uncertainty.
- **Lead‑lag**: lagged correlations suggest predictive association only; **not causation** and not trading advice.

## Data source

Price history is pulled from Yahoo Finance via `yfinance` and uses **Adjusted Close** when available.

### If Yahoo Finance is blocked on your network

Some campus/corporate networks (or temporary Yahoo throttling) can cause `yfinance` downloads to fail.
This prototype includes **live provider options**:
- **Stooq (default)**: live daily closes without API keys (may not match “adjusted close”).
- **Yahoo (yfinance)**: live OHLCV via `yfinance` (unofficial Yahoo wrapper); may be blocked on some networks.

