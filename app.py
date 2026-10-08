"""Stock Setup Reader: single-stock read plus a Top 10 bullish / bearish scanner.
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

# ---------- scanner ----------
# Yahoo symbols without ".NS". "core" = large caps (fast), "extra" = mid caps. Edit freely.
SECTORS = {
 "Banks & Financials": dict(core="HDFCBANK ICICIBANK SBIN KOTAKBANK AXISBANK INDUSINDBK BAJFINANCE BAJAJFINSV SBILIFE HDFCLIFE SHRIRAMFIN JIOFIN BANKBARODA PNB",
    extra="FEDERALBNK IDFCFIRSTB AUBANK CHOLAFIN MUTHOOTFIN PFC RECLTD CANBK LICHSGFIN"),
 "IT": dict(core="TCS INFY HCLTECH WIPRO TECHM", extra="LTIM PERSISTENT COFORGE MPHASIS OFSS"),
 "Auto": dict(core="MARUTI M&M EICHERMOT HEROMOTOCO BAJAJ-AUTO", extra="TVSMOTOR ASHOKLEY BOSCHLTD MOTHERSON BHARATFORG"),
 "Pharma & Healthcare": dict(core="SUNPHARMA CIPLA DRREDDY APOLLOHOSP", extra="LUPIN ZYDUSLIFE TORNTPHARM AUROPHARMA MAXHEALTH DIVISLAB"),
 "FMCG": dict(core="HINDUNILVR ITC NESTLEIND BRITANNIA TATACONSUM", extra="DABUR GODREJCP MARICO COLPAL VBL"),
 "Oil, Gas & Energy": dict(core="RELIANCE ONGC BPCL COALINDIA", extra="IOC GAIL HINDPETRO OIL"),
 "Metals": dict(core="JSWSTEEL TATASTEEL HINDALCO", extra="VEDL JINDALSTEL SAIL NMDC"),
 "Power & Utilities": dict(core="NTPC POWERGRID", extra="TATAPOWER ADANIPOWER ADANIGREEN NHPC"),
 "Capital Goods, Infra & Defence": dict(core="LT ULTRACEMCO GRASIM ADANIENT ADANIPORTS BEL",
    extra="SIEMENS ABB HAL BHEL CUMMINSIND POLYCAB DLF AMBUJACEM SHREECEM"),
 "Consumer, Retail & Travel": dict(core="TITAN ASIANPAINT TRENT ETERNAL INDIGO",
    extra="DMART PAGEIND HAVELLS VOLTAS JUBLFOOD BERGEPAINT INDHOTEL NAUKRI PIDILITIND"),
 "Telecom": dict(core="BHARTIARTL", extra="INDUSTOWER"),
}
SECTOR_OF, CORE, EXTRA = {}, [], []
for _sec, _g in SECTORS.items():
    for _k, _lst in (("core", CORE), ("extra", EXTRA)):
        for _s in _g[_k].split():
            SECTOR_OF[_s + ".NS"] = _sec; _lst.append(_s + ".NS")

def scan(mode, syms):
    import yfinance as yf
    P = PARAMS[mode]
    raw = yf.download(list(syms), period=P["period"], interval=P["interval"], auto_adjust=True,
                      progress=False, group_by="ticker", threads=True)
    rows = []
    for s in syms:
        try:
            df = raw[s].dropna()
            if len(df) < 120: continue
            d = build(df, mode); x = d.iloc[-1]; ts = stats(backtest(d, mode))
            k = int(np.sign(x.score)); e, a = float(x.Close), float(x.atr)
            fired = int(x.sig) != 0
            rows.append(dict(Stock=s.replace(".NS", ""), Sector=SECTOR_OF.get(s, "Other"),
                Status="Signal" if fired else "Leaning", Score=float(x.score), Price=round(e, 2),
                ADX=round(float(x.adx)), RSI=round(float(x.rsi)),
                Entry=round(e, 2) if fired else None,
                Stop=round(e - k * P["sl"] * a, 2) if fired else None,
                Target=round(e + k * P["tg"] * a, 2) if fired else None,
                PastTrades=ts["trades"] if ts else 0, PastWin=round(ts["win"]) if ts else None,
                PastAvgR=round(float(ts["avgR"]), 2) if ts else None, AsOf=str(d.index[-1])))
        except Exception:
            continue
    return pd.DataFrame(rows)

def top(df, side, n=10, need_edge=True, per_sector=3, sectors=None):
    d = df[df.Score > 0] if side > 0 else df[df.Score < 0]
    if sectors: d = d[d.Sector.isin(sectors)]
    if need_edge: d = d[(d.PastAvgR > 0) & (d.PastTrades >= 5)]
    d = d.assign(a=d.Score.abs()).sort_values(["a", "ADX"], ascending=False)
    d = d.groupby("Sector", sort=False).head(per_sector).head(n)
    return d.drop(columns=["a", "AsOf"]).reset_index(drop=True)

# ---------- UI ----------
def single(st):
    q = st.text_input("Stock name", placeholder="e.g. Reliance, TCS, Asahi India Glass")
    mode = st.radio("Style", ["Intraday", "Swing"], horizontal=True, key="m1")
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

def scanner(st):
    @st.cache_data(ttl=600, show_spinner=False)
    def cached(mode, big):
        return scan(mode, CORE + EXTRA if big else CORE)
    c1, c2 = st.columns(2)
    mode = c1.radio("Style", ["Swing", "Intraday"], horizontal=True, key="m2")
    big = c2.radio("Stocks to scan", [f"Large caps ({len(CORE)})", f"Large + mid caps ({len(CORE) + len(EXTRA)})"], horizontal=True) != f"Large caps ({len(CORE)})"
    sectors = st.multiselect("Sectors (leave empty for all)", list(SECTORS))
    c3, c4 = st.columns(2)
    cap = c3.slider("Max stocks per sector in each list", 1, 10, 3)
    need = c4.checkbox("Hide stocks where these rules lost money in the past", value=True)
    st.caption("Intraday works best with large caps. Scanning mid caps takes a few minutes.")
    if not st.button("Scan now", type="primary"): return
    with st.spinner("Downloading and scoring stocks"):
        df = cached(mode, big)
    if df.empty: st.error("No data came back. Yahoo may be blocking for a few minutes. Try again later."); return
    st.write(f"Scanned {len(df)} stocks. Latest data point: {df.AsOf.max()}")
    cols = {"PastTrades": "Past trades", "PastWin": "Past win %", "PastAvgR": "Past avg R"}
    for side, title in [(1, "Top 10 bullish"), (-1, "Top 10 bearish (short or avoid)")]:
        t = top(df, side, 10, need, cap, sectors)
        st.subheader(title)
        if t.empty: st.info("Nothing passes the filters right now.")
        else: st.dataframe(t.rename(columns=cols), use_container_width=True, hide_index=True)
    st.caption("Signal = all checks agree and the market is trending, so entry, stop and target are shown. Leaning = direction is positive or negative but not all checks agree. Past columns show how the same rules did on that stock before. Short selling in delivery is not allowed in India, so bearish names are mainly for intraday shorts or to avoid. Symbols Yahoo does not recognise are skipped.")

def main():
    import streamlit as st
    st.set_page_config(page_title="Setup Reader", layout="wide")
    st.title("Setup Reader")
    st.caption("Live data from Yahoo Finance (can lag a few minutes). Not investment advice.")
    t1, t2 = st.tabs(["Top 10 scanner", "Single stock"])
    with t1: scanner(st)
    with t2: single(st)

if __name__ == "__main__":
    main()
