"""Stock Setup Reader: type a stock name, get live data, signal, levels and a measured hit rate.
Run:  pip install streamlit yfinance pandas numpy   then   streamlit run app.py
"""
import numpy as np
import pandas as pd

# ---------- indicators ----------
def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def rsi(c, n=14):
    d = c.diff(); up = d.clip(lower=0); dn = -d.clip(upper=0)
    rs = up.ewm(alpha=1/n, adjust=False).mean() / dn.ewm(alpha=1/n, adjust=False).mean().replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def atr(df, n=14):
    pc = df.Close.shift()
    tr = pd.concat([df.High - df.Low, (df.High - pc).abs(), (df.Low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def adx(df, n=14):
    up, dn = df.High.diff(), -df.Low.diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    ndm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    a = atr(df, n)
    pdi = 100 * pdm.ewm(alpha=1/n, adjust=False).mean() / a
    ndi = 100 * ndm.ewm(alpha=1/n, adjust=False).mean() / a
    dx = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    return dx.ewm(alpha=1/n, adjust=False).mean()

PARAMS = {
    "Intraday": dict(interval="15m", period="30d", f=9,  s=21, sl=1.0, tg=1.5, hold=16),
    "Swing":    dict(interval="1d",  period="3y",  f=20, s=50, sl=1.5, tg=3.0, hold=10),
}

def build(df, mode):
    P = PARAMS[mode]; d = df.copy()
    c = d.Close
    d["ef"], d["es"] = ema(c, P["f"]), ema(c, P["s"])
    macd = ema(c, 12) - ema(c, 26); d["mh"] = macd - ema(macd, 9)
    d["rsi"], d["atr"], d["adx"] = rsi(c), atr(d), adx(d)
    d["vol_ok"] = d.Volume > 1.2 * d.Volume.rolling(20).mean()
    s = np.sign(d.ef - d.es) + np.sign(c - d.es)
    s += np.where((d.mh > 0) & (d.mh > d.mh.shift()), 1, np.where((d.mh < 0) & (d.mh < d.mh.shift()), -1, 0))
    s += np.where((d.rsi >= 50) & (d.rsi <= 70), 1, np.where((d.rsi >= 30) & (d.rsi < 50), -1, 0))
    s += np.where(d.vol_ok, np.sign(c - d.Open), 0)
    if mode == "Intraday":
        day = d.index.date
        tp = (d.High + d.Low + d.Close) / 3
        vwap = (tp * d.Volume).groupby(day).cumsum() / d.Volume.groupby(day).cumsum().replace(0, np.nan)
        d["vwap"] = vwap; s += np.sign(c - vwap).fillna(0)
    d["score"] = s
    trending = d.adx >= 20                      # skip choppy markets
    d["sig"] = np.where((s >= 3) & trending, 1, np.where((s <= -3) & trending, -1, 0))
    return d

def backtest(d, mode, cost=0.0005):
    """Enter next open after a fresh signal; ATR stop/target; stop assumed first if both hit."""
    P = PARAMS[mode]; rows = []; i = 60; n = len(d)
    sig, atrv = d.sig.values, d.atr.values
    o, h, l, c = d.Open.values, d.High.values, d.Low.values, d.Close.values
    while i < n - 2:
        if sig[i] != 0 and sig[i] != sig[i-1] and atrv[i] > 0:
            k = sig[i]; en = o[i+1]; sd = P["sl"] * atrv[i]
            st, tg = en - k * sd, en + k * P["tg"] * atrv[i]; r = None; j = i + 1
            while j <= min(i + P["hold"], n - 1):
                hs = l[j] <= st if k > 0 else h[j] >= st
                ht = h[j] >= tg if k > 0 else l[j] <= tg
                if hs: r = -1.0; break
                if ht: r = P["tg"] / P["sl"]; break
                j += 1
            if r is None:
                j = min(i + P["hold"], n - 1); r = k * (c[j] - en) / sd
            r -= cost * en / sd
            rows.append((d.index[i], k, r)); i = j + 1; continue
        i += 1
    return pd.DataFrame(rows, columns=["when", "dir", "R"])

def stats(t):
    if len(t) == 0: return None
    w, lo = t.R[t.R > 0], t.R[t.R <= 0]
    eq = t.R.cumsum()
    return dict(trades=len(t), win=100 * len(w) / len(t), avgR=t.R.mean(),
                pf=(w.sum() / -lo.sum()) if lo.sum() < 0 else float("inf"),
                dd=float((eq - eq.cummax()).min()))

def verdict(d, mode):
    P = PARAMS[mode]; x = d.iloc[-1]; k = int(x.sig)
    lv = None
    if k:
        e = float(x.Close); a = float(x.atr)
        lv = dict(entry=e, stop=e - k * P["sl"] * a, target=e + k * P["tg"] * a)
    return k, lv

# ---------- data ----------
def resolve(q):
    import yfinance as yf
    q = q.strip()
    try:
        for r in yf.Search(q, max_results=10).quotes:
            if r.get("symbol", "").endswith((".NS", ".BO")): return r["symbol"], r.get("shortname", q)
    except Exception: pass
    return (q.upper() if "." in q else q.upper() + ".NS"), q

def load(sym, mode):
    import yfinance as yf
    P = PARAMS[mode]
    df = yf.download(sym, period=P["period"], interval=P["interval"], auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    return df.dropna()

# ---------- UI ----------
def main():
    import streamlit as st
    st.set_page_config(page_title="Setup Reader", layout="centered")
    st.title("Setup Reader")
    st.caption("Type a stock name. Live data from Yahoo Finance (can lag a few minutes). Not investment advice.")
    q = st.text_input("Stock name", placeholder="e.g. Reliance, TCS, Asahi India Glass")
    mode = st.radio("Style", ["Intraday", "Swing"], horizontal=True)
    if not (q and st.button("Analyse", type="primary")): return
    sym, name = resolve(q)
    with st.spinner(f"Fetching {sym}"):
        df = load(sym, mode)
    if len(df) < 120: st.error(f"Not enough data for {sym}. Try the exact NSE symbol."); return
    d = build(df, mode); k, lv = verdict(d, mode); x = d.iloc[-1]
    word = {1: "Long", -1: "Short", 0: "Wait"}[k]
    st.header(f"{sym}: {word}")
    st.write(f"Last price {x.Close:.2f} at {d.index[-1]}. Score {x.score:+.0f}, ADX {x.adx:.0f}, RSI {x.rsi:.0f}.")
    if lv:
        a, b, c3 = st.columns(3)
        a.metric("Entry", f"{lv['entry']:.2f}"); b.metric("Stop", f"{lv['stop']:.2f}"); c3.metric("Target", f"{lv['target']:.2f}")
    else:
        st.info("Indicators conflict or the market is not trending (ADX below 20). No trade is a valid call.")
    t = backtest(d, mode)
    cut = int(len(t) * 0.7); a_, b_ = stats(t.iloc[:cut]), stats(t.iloc[cut:])
    st.subheader("How these rules did on this stock")
    for lab, s in [("All history", stats(t)), ("Older 70% (rules tuned here)", a_), ("Newest 30% (honest test)", b_)]:
        if s: st.write(f"**{lab}:** {s['trades']} trades, {s['win']:.0f}% winners, avg {s['avgR']:+.2f}R, profit factor {s['pf']:.2f}, worst drawdown {s['dd']:.1f}R")
    st.caption("R = amount risked per trade. Includes 0.05% cost per trade. Few trades means the numbers are unreliable.")
    st.line_chart(d[["Close", "ef", "es"]].tail(150))

if __name__ == "__main__":
    main()
