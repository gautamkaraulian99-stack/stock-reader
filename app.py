"""
VCP screener (Mark Minervini style) for NSE stocks.
 
A stock is suggested when ALL of these are true:
  1. EMA(21) crossed ABOVE EMA(63) within the last `cross_bars` candles ("just crossed")
  2. ROC is near zero  ->  |ROC| <= roc_tol (%)
        ROC length = 18 for small caps, 20 for mid/large caps
  3. A VCP is visible: >= `min_contractions` pullbacks, each shallower than the
     one before, last one tight, price near the pivot, volume drying up.
 
Single-file version (backend + web page + stock list in one file).
 
Run:
    pip install flask pandas numpy yfinance
    python vcp_screener.py     ->  http://127.0.0.1:5000
 
To use your own stock list, create a file named universe.csv (columns: symbol,cap)
next to this file. Otherwise the DEFAULT_UNIVERSE below is used.
"""
import io
import os
import time
 
import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, Response, jsonify, request
 
app = Flask(__name__)
 
CACHE = {}          # params -> (timestamp, results)
CACHE_TTL = 15 * 60  # seconds
BATCH = 40          # tickers per yfinance download call
 
 
# --------------------------------------------------------------------------
# Universe
# --------------------------------------------------------------------------
DEFAULT_UNIVERSE = '''symbol,cap
RELIANCE,large
TCS,large
HDFCBANK,large
INFY,large
ICICIBANK,large
LT,large
ITC,large
TATAMOTORS,large
ASIANPAINT,large
TITAN,large
POLYCAB,mid
PERSISTENT,mid
TRENT,mid
CUMMINSIND,mid
ASTRAL,mid
AIAENG,mid
KPITTECH,mid
CDSL,small
CAMPUS,small
RADICO,small
RATNAMANI,small
SUPRAJIT,small
JKPAPER,small
GRSE,small
'''
 
 
def load_universe():
    """universe.csv (optional) columns: symbol,cap   (cap = small | mid | large)"""
    if os.path.exists("universe.csv"):
        df = pd.read_csv("universe.csv")
    else:
        df = pd.read_csv(io.StringIO(DEFAULT_UNIVERSE))
    df["symbol"] = df["symbol"].str.strip().str.upper()
    df["cap"] = df["cap"].str.strip().str.lower()
    return df
 
 
def roc_length(cap):
    return 18 if cap == "small" else 20  # mid & large -> 20
 
 
# --------------------------------------------------------------------------
# Indicators
# --------------------------------------------------------------------------
def add_indicators(df, cap):
    n = roc_length(cap)
    df = df.copy()
    df["ema21"] = df["Close"].ewm(span=21, adjust=False).mean()
    df["ema63"] = df["Close"].ewm(span=63, adjust=False).mean()
    df["roc"] = (df["Close"] / df["Close"].shift(n) - 1) * 100
    return df
 
 
def bars_since_cross_up(df, lookback):
    """Return how many candles ago EMA21 crossed above EMA63 (0 = today), else None."""
    diff = df["ema21"] - df["ema63"]
    if diff.iloc[-1] <= 0:
        return None
    crossed = ((diff > 0) & (diff.shift(1) <= 0)).iloc[-lookback:]
    idx = np.where(crossed.values)[0]
    if len(idx) == 0:
        return None
    return int(len(crossed) - 1 - idx[-1])
 
 
def detect_vcp(df, lookback=120, order=5, min_contractions=2,
               last_max_depth=12.0, pivot_dist=8.0):
    """
    Volatility Contraction Pattern detector.
    - Finds swing highs (local max over +/- `order` candles).
    - For each swing high, depth = drop to the lowest low before the next swing high.
    - Needs depths to shrink contraction after contraction.
    """
    d = df.iloc[-lookback:]
    high, low, close, vol = d["High"].values, d["Low"].values, d["Close"].values, d["Volume"].values
    n = len(d)
 
    swing_highs = []
    for i in range(order, n - order):
        if high[i] == high[i - order:i + order + 1].max():
            swing_highs.append(i)
    if len(swing_highs) < min_contractions:
        return None
 
    contractions = []
    for k, hi in enumerate(swing_highs):
        end = swing_highs[k + 1] if k + 1 < len(swing_highs) else n
        seg_low = low[hi:end].min()
        depth = (1 - seg_low / high[hi]) * 100
        contractions.append((hi, depth))
 
    # take the longest run of shrinking depths that ends at the latest contraction
    run = [contractions[-1]]
    for c in reversed(contractions[:-1]):
        if run[0][1] < c[1]:       # earlier pullback was deeper -> still contracting
            run.insert(0, c)
        else:
            break
    run = run[-4:]                  # classic VCPs have 2-4 contractions
    if len(run) < min_contractions:
        return None
 
    depths = [round(x[1], 1) for x in run]
    if depths[0] > 40 or depths[-1] > last_max_depth:
        return None
 
    pivot = max(high[run[0][0]:])                       # base high
    dist = (pivot / close[-1] - 1) * 100
    if dist < 0 or dist > pivot_dist:                   # must sit just below the pivot
        return None
 
    if vol[-10:].mean() >= vol[-50:].mean():            # volume dry-up
        return None
 
    return {"contractions": depths, "pivot": round(float(pivot), 2),
            "dist_to_pivot": round(float(dist), 1)}
 
 
# --------------------------------------------------------------------------
# Scan
# --------------------------------------------------------------------------
def scan(cross_bars, roc_tol, min_contractions, require_vcp):
    uni = load_universe()
    results = []
 
    for start in range(0, len(uni), BATCH):
        chunk = uni.iloc[start:start + BATCH]
        tickers = [f"{s}.NS" for s in chunk["symbol"]]
        try:
            data = yf.download(tickers, period="1y", interval="1d", group_by="ticker",
                               auto_adjust=True, progress=False, threads=True)
        except Exception:
            continue
 
        for sym, cap in zip(chunk["symbol"], chunk["cap"]):
            try:
                df = data[f"{sym}.NS"] if len(tickers) > 1 else data
                df = df.dropna(subset=["Close", "High", "Low", "Volume"])
                if len(df) < 130:
                    continue
                df = add_indicators(df, cap)
 
                since = bars_since_cross_up(df, cross_bars)
                if since is None:
                    continue
 
                roc = float(df["roc"].iloc[-1])
                if not abs(roc) <= roc_tol:
                    continue
 
                vcp = detect_vcp(df, min_contractions=min_contractions)
                if require_vcp and vcp is None:
                    continue
 
                results.append({
                    "symbol": sym,
                    "cap": cap,
                    "close": round(float(df["Close"].iloc[-1]), 2),
                    "roc_len": roc_length(cap),
                    "roc": round(roc, 2),
                    "ema21": round(float(df["ema21"].iloc[-1]), 2),
                    "ema63": round(float(df["ema63"].iloc[-1]), 2),
                    "crossed_days_ago": since,
                    "vcp": vcp,
                })
            except Exception:
                continue
 
    results.sort(key=lambda r: (r["crossed_days_ago"], abs(r["roc"])))
    return results
 
 
INDEX_HTML = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VCP Screener</title>
<style>
  :root {
    --bg: #f3f5f2;
    --surface: #ffffff;
    --ink: #17211b;
    --muted: #5f6b63;
    --line: #d9dfd9;
    --accent: #0f6b4a;
    --accent-ink: #ffffff;
    --warn: #a4471b;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #101613;
      --surface: #18201b;
      --ink: #e6ede8;
      --muted: #94a39a;
      --line: #2a352e;
      --accent: #4fc08d;
      --accent-ink: #0d1a13;
      --warn: #e0925f;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--ink);
    font: 15px/1.5 "IBM Plex Sans", system-ui, sans-serif;
  }
  main { max-width: 1040px; margin: 0 auto; padding: 32px 20px 64px; }
  h1 { font: 600 28px/1.2 "IBM Plex Sans", system-ui, sans-serif; margin: 0 0 6px; letter-spacing: -0.01em; }
  .sub { color: var(--muted); margin: 0 0 24px; max-width: 70ch; }
  form {
    display: flex; flex-wrap: wrap; gap: 16px 20px; align-items: end;
    background: var(--surface); border: 1px solid var(--line); border-radius: 10px;
    padding: 16px 18px; margin-bottom: 24px;
  }
  label { display: flex; flex-direction: column; gap: 4px; font-size: 13px; color: var(--muted); }
  input[type=number] {
    width: 110px; padding: 7px 9px; font: inherit; color: var(--ink);
    background: var(--bg); border: 1px solid var(--line); border-radius: 6px;
  }
  .check { flex-direction: row; align-items: center; gap: 8px; color: var(--ink); padding-bottom: 7px; }
  button {
    font: 600 14px "IBM Plex Sans", system-ui, sans-serif; padding: 9px 18px; border: 0; border-radius: 6px;
    background: var(--accent); color: var(--accent-ink); cursor: pointer;
  }
  button:disabled { opacity: .6; cursor: wait; }
  :focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
  #status { color: var(--muted); margin: 0 0 12px; min-height: 22px; }
  #status.error { color: var(--warn); }
  .wrap { overflow-x: auto; background: var(--surface); border: 1px solid var(--line); border-radius: 10px; }
  table { width: 100%; border-collapse: collapse; min-width: 760px; }
  th, td { padding: 11px 14px; text-align: left; border-bottom: 1px solid var(--line); white-space: nowrap; }
  th { font-weight: 600; font-size: 13px; color: var(--muted); }
  tr:last-child td { border-bottom: 0; }
  td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
  a { color: var(--accent); font-weight: 600; text-decoration: none; }
  a:hover { text-decoration: underline; }
  .tag { font-size: 12px; padding: 2px 8px; border: 1px solid var(--line); border-radius: 99px; color: var(--muted); }
  .note { color: var(--muted); font-size: 13px; margin-top: 20px; max-width: 75ch; }
</style>
</head>
<body>
<main>
  <h1>VCP screener</h1>
  <p class="sub">Stocks where the 21 EMA just crossed above the 63 EMA, ROC sits near zero, and price is tightening into a Minervini-style volatility contraction pattern.</p>
 
  <form id="f">
    <label>Crossed within (candles)
      <input type="number" id="cross_bars" value="3" min="1" max="20">
    </label>
    <label>ROC tolerance (± %)
      <input type="number" id="roc_tol" value="3" min="0.1" step="0.1">
    </label>
    <label>Min contractions
      <input type="number" id="min_contractions" value="2" min="2" max="4">
    </label>
    <label class="check"><input type="checkbox" id="require_vcp" checked> Require VCP</label>
    <button id="go" type="submit">Run scan</button>
  </form>
 
  <p id="status" role="status">Set your filters and run the scan.</p>
 
  <div class="wrap" id="tablewrap" hidden>
    <table>
      <thead>
        <tr>
          <th>Stock</th><th>Cap</th><th class="num">Close</th>
          <th class="num">ROC (len)</th><th class="num">EMA 21 / 63</th>
          <th class="num">Crossed</th><th>Contractions (%)</th><th class="num">Pivot</th>
        </tr>
      </thead>
      <tbody id="rows"></tbody>
    </table>
  </div>
 
  <p class="note">ROC length is 18 for small caps and 20 for mid and large caps. Data comes from Yahoo Finance daily candles and may lag. This is a screening aid, not investment advice.</p>
</main>
 
<script>
const $ = id => document.getElementById(id);
const capLabel = { small: "Small", mid: "Mid", large: "Large" };
 
$("f").addEventListener("submit", async e => {
  e.preventDefault();
  const status = $("status"), btn = $("go");
  status.className = "";
  status.textContent = "Scanning… this can take a minute for a large list.";
  btn.disabled = true;
  $("tablewrap").hidden = true;
 
  const q = new URLSearchParams({
    cross_bars: $("cross_bars").value,
    roc_tol: $("roc_tol").value,
    min_contractions: $("min_contractions").value,
    require_vcp: $("require_vcp").checked ? "1" : "0",
  });
 
  try {
    const res = await fetch("/api/scan?" + q);
    if (!res.ok) throw new Error("Server returned " + res.status);
    const { results } = await res.json();
    render(results);
  } catch (err) {
    status.className = "error";
    status.textContent = "Scan failed: " + err.message + ". Check that the server is running and try again.";
  } finally {
    btn.disabled = false;
  }
});
 
function render(list) {
  const status = $("status"), rows = $("rows");
  rows.replaceChildren();
  if (!list.length) {
    status.textContent = "No matches. Try a wider ROC tolerance, more candles since the cross, or turn off “Require VCP”.";
    return;
  }
  status.textContent = list.length + " stock" + (list.length > 1 ? "s" : "") + " match.";
  for (const r of list) {
    const tr = document.createElement("tr");
    const vcp = r.vcp ? r.vcp.contractions.join(" → ") : "–";
    const pivot = r.vcp ? r.vcp.pivot.toFixed(2) + " (" + r.vcp.dist_to_pivot + "% away)" : "–";
    const when = r.crossed_days_ago === 0 ? "Today" : r.crossed_days_ago + "d ago";
    tr.innerHTML =
      `<td><a target="_blank" rel="noopener" href="https://www.tradingview.com/chart/?symbol=NSE:${encodeURIComponent(r.symbol)}">${r.symbol}</a></td>` +
      `<td><span class="tag">${capLabel[r.cap] || r.cap}</span></td>` +
      `<td class="num">${r.close.toFixed(2)}</td>` +
      `<td class="num">${r.roc.toFixed(2)}% (${r.roc_len})</td>` +
      `<td class="num">${r.ema21.toFixed(2)} / ${r.ema63.toFixed(2)}</td>` +
      `<td class="num">${when}</td>` +
      `<td>${vcp}</td>` +
      `<td class="num">${pivot}</td>`;
    rows.appendChild(tr);
  }
  $("tablewrap").hidden = false;
}
</script>
</body>
</html>
'''
 
 
# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.route("/")
def index():
    return Response(INDEX_HTML, mimetype="text/html")
 
 
@app.route("/api/scan")
def api_scan():
    cross_bars = int(request.args.get("cross_bars", 3))
    roc_tol = float(request.args.get("roc_tol", 3))
    min_c = int(request.args.get("min_contractions", 2))
    require_vcp = request.args.get("require_vcp", "1") == "1"
 
    key = (cross_bars, roc_tol, min_c, require_vcp)
    hit = CACHE.get(key)
    if hit and time.time() - hit[0] < CACHE_TTL:
        return jsonify({"cached": True, "results": hit[1]})
 
    results = scan(cross_bars, roc_tol, min_c, require_vcp)
    CACHE[key] = (time.time(), results)
    return jsonify({"cached": False, "results": results})
 
 
if __name__ == "__main__":
    app.run(debug=True)
