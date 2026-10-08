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

def mkt_trend(nifty, idx, mode):
    P = PARAMS[mode]; c = nifty.Close; ef, es = ema(c, P["f"]), ema(c, P["s"])
    t = pd.Series(np.where((c > es) & (ef > es), 1, np.where((c < es) & (ef < es), -1, 0)), index=c.index)
    return t.reindex(idx, method="ffill").values

def htf_trend(df, mode):
    c = df.Close
    w, a, b = (c.resample("W-FRI").last().dropna(), 10, 20) if mode == "Swing" else (c.resample("D").last().dropna(), 5, 10)
    ea, eb = ema(w, a), ema(w, b)
    t = pd.Series(np.where((w > eb) & (ea > eb), 1, np.where((w < eb) & (ea < eb), -1, 0)), index=w.index)
    if mode != "Swing": t = t.shift(1)          # use only completed days for intraday bars
    key = c.index if mode == "Swing" else c.index.normalize()
    return t.reindex(key, method="ffill").values

def build(df, mode, nifty=None, opts=None):
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
    raw = np.where((s >= 3) & trending, 1, np.where((s <= -3) & trending, -1, 0))
    o = opts or {}
    d["mkt"] = d["rs"] = d["htf"] = np.nan
    try:
        if nifty is not None:
            d["mkt"] = mkt_trend(nifty, d.index, mode)
            nc = nifty.Close.reindex(d.index, method="ffill"); L = 60 if mode == "Swing" else 125
            d["rs"] = c.pct_change(L) - nc.pct_change(L)
    except Exception: pass
    try: d["htf"] = htf_trend(d, mode)
    except Exception: pass
    okl = pd.Series(True, index=d.index); oks = okl.copy()
    for key, col in (("market", "mkt"), ("htf", "htf")):
        if o.get(key) and d[col].notna().any(): okl &= d[col] > 0; oks &= d[col] < 0
    if o.get("rs") and d.rs.notna().any(): okl &= d.rs > 0; oks &= d.rs < 0
    d["sig"] = np.where((raw == 1) & okl, 1, np.where((raw == -1) & oks, -1, 0))
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
    lv = None; kk = k or int(np.sign(x.score))      # weak setups still get reference levels
    if kk:
        e = float(x.Close); a = float(x.atr)
        lv = dict(entry=e, stop=e - kk * P["sl"] * a, target=e + kk * P["tg"] * a)
    return k, lv

# ---------- data ----------
def resolve(q, market="India (NSE)"):
    import yfinance as yf
    q = q.strip(); us = market == "US"
    try:
        for r in yf.Search(q, max_results=10).quotes:
            sym = r.get("symbol", "")
            if us and r.get("quoteType") == "EQUITY" and "." not in sym: return sym, r.get("shortname", q)
            if not us and sym.endswith((".NS", ".BO")): return sym, r.get("shortname", q)
    except Exception: pass
    return (q.upper() if us or "." in q else q.upper() + ".NS"), q

def load(sym, mode):
    import yfinance as yf
    P = PARAMS[mode]
    df = yf.download(sym, period=P["period"], interval=P["interval"], auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    return df.dropna()

# ---------- scanner ----------
# Yahoo symbols without ".NS". "core" = large caps (fast), "extra" = mid caps. Edit freely.
SECTORS_IN = {
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
SECTORS_US = {
 "Technology": dict(core="AAPL MSFT NVDA AVGO ORCL CRM ADBE AMD CSCO INTC QCOM TXN IBM", extra="NOW INTU AMAT MU LRCX KLAC PANW SNPS CDNS ANET"),
 "Communication": dict(core="GOOGL META NFLX DIS CMCSA T VZ TMUS", extra="CHTR EA TTWO"),
 "Consumer Discretionary": dict(core="AMZN TSLA HD MCD NKE SBUX LOW BKNG", extra="TJX ABNB CMG MAR ORLY GM F"),
 "Consumer Staples": dict(core="WMT COST PG KO PEP PM MDLZ", extra="CL MO TGT KHC GIS"),
 "Financials": dict(core="JPM BAC WFC GS MS V MA AXP BLK BRK-B C", extra="SCHW SPGI PNC USB COF CB PGR MMC"),
 "Healthcare": dict(core="LLY UNH JNJ ABBV MRK TMO ABT PFE AMGN", extra="ISRG DHR BMY GILD VRTX MDT REGN CVS"),
 "Energy": dict(core="XOM CVX COP", extra="SLB EOG OXY PSX MPC"),
 "Industrials": dict(core="CAT GE RTX HON UPS BA DE LMT", extra="UNP ETN WM ADP NOC FDX EMR"),
 "Utilities & Real Estate": dict(core="NEE DUK SO AMT PLD", extra="D EXC EQIX SPG O PSA"),
 "Materials": dict(core="LIN SHW", extra="FCX NEM APD ECL DOW"),
}

def make_market(sectors, suffix, bench):
    so, core, extra = {}, [], []
    for sec, g in sectors.items():
        for k, lst in (("core", core), ("extra", extra)):
            for t in g[k].split():
                so[t + suffix] = sec; lst.append(t + suffix)
    return dict(sectors=sectors, suffix=suffix, bench=bench, sector_of=so, core=core, extra=extra)

MK = {"India (NSE)": make_market(SECTORS_IN, ".NS", "^NSEI"), "US": make_market(SECTORS_US, "", "^GSPC")}

def scan(mode, syms, opts=None, market="India (NSE)"):
    import yfinance as yf
    P = PARAMS[mode]
    nifty = None
    if opts and any(opts.values()):
        try: nifty = load(MK[market]["bench"], mode)
        except Exception: nifty = None
    raw = yf.download(list(syms), period=P["period"], interval=P["interval"], auto_adjust=True,
                      progress=False, group_by="ticker", threads=True)
    rows = []
    for s in syms:
        try:
            df = raw[s].dropna()
            if len(df) < 120: continue
            d = build(df, mode, nifty, opts); x = d.iloc[-1]; ts = stats(backtest(d, mode))
            k = int(np.sign(x.score)); e, a = float(x.Close), float(x.atr)
            fired = int(x.sig) != 0
            rows.append(dict(Stock=s.removesuffix(MK[market]["suffix"]) if MK[market]["suffix"] else s, Sector=MK[market]["sector_of"].get(s, "Other"),
                Status="Signal" if fired else "Leaning", Score=float(x.score), Price=round(e, 2),
                ADX=round(float(x.adx)), RSI=round(float(x.rsi)),
                RSvsIndex=round(float(x.rs) * 100, 1) if pd.notna(x.rs) else None,
                Entry=round(e, 2), Chase=round(e + k * 0.25 * a, 2),
                Stop=round(e - k * P["sl"] * a, 2),
                Target=round(e + k * P["tg"] * a, 2),
                PastTrades=ts["trades"] if ts else 0, PastWin=round(ts["win"]) if ts else None,
                PastAvgR=round(float(ts["avgR"]), 2) if ts else None, AsOf=str(d.index[-1])))
        except Exception:
            continue
    return pd.DataFrame(rows)

def top(df, side, n=10, need_edge=True, per_sector=3, sectors=None):
    d = df[df.Score > 0] if side > 0 else df[df.Score < 0]
    if sectors: d = d[d.Sector.isin(sectors)]
    if need_edge: d = d[(d.PastAvgR > 0) & (d.PastTrades >= 5)]
    d = d.assign(a=d.Score.abs(), f=d.Status == "Signal").sort_values(["f", "a", "ADX"], ascending=False)
    d = d.groupby("Sector", sort=False).head(per_sector).head(n)
    order = ["Stock", "Sector", "Status", "Score", "Entry", "Chase", "Stop", "Target", "Price", "ADX", "RSI", "RSvsIndex", "PastTrades", "PastWin", "PastAvgR"]
    return d[order].reset_index(drop=True)

def live_price(sym):
    import yfinance as yf
    try:
        p = yf.Ticker(sym).fast_info["last_price"]
        return float(p) if p and p == p else None
    except Exception:
        return None

# ---------- next-day outlook ----------
def nextday(df):
    """Measured next-day odds for this stock after days with a similar score. Not a forecast."""
    d = build(df, "Swing")
    ret = d.Close.pct_change().shift(-1)                      # next-day close-to-close return
    bucket = pd.Series(np.where(d.score >= 3, 1, np.where(d.score <= -3, -1, 0)), index=d.index)
    r, b = ret.iloc[60:-1], bucket.iloc[60:-1]
    cur = int(bucket.iloc[-1]); x = d.iloc[-1]
    base = float((r > 0).mean()); m = b == cur; n = int(m.sum())
    pos = np.arange(len(r)); cut = int(len(r) * 0.7)
    def pu(mask): return float((r[mask] > 0).mean()) if mask.sum() >= 5 else float("nan")
    p, p_old, p_new = (pu(m), pu(m & (pos < cut)), pu(m & (pos >= cut))) if n else (float("nan"),) * 3
    bias = "No clear edge"
    if cur != 0 and n >= 40 and not np.isnan(p_old) and not np.isnan(p_new):
        if cur == 1 and p - base >= 0.04 and p_old > base and p_new > base: bias = "Up bias"
        if cur == -1 and base - p >= 0.04 and p_old < base and p_new < base: bias = "Down bias"
    h, l, c = float(x.High), float(x.Low), float(x.Close); pv = (h + l + c) / 3
    return dict(bias=bias, cur=cur, n=n, p=p, p_old=p_old, p_new=p_new, base=base,
                avg=float(r[m].mean()) if n else float("nan"), close=c, pivot=pv, r1=2 * pv - l, s1=2 * pv - h,
                atr_pct=float(x.atr) / c * 100, asof=str(d.index[-1])[:10])

def scan_next(syms, market):
    import yfinance as yf
    P = PARAMS["Swing"]; raw = yf.download(list(syms), period=P["period"], interval=P["interval"],
                                           auto_adjust=True, progress=False, group_by="ticker", threads=True)
    M = MK[market]; rows = []
    for s in syms:
        try:
            df = raw[s].dropna()
            if len(df) < 150: continue
            r = nextday(df)
            rows.append(dict(Stock=s.removesuffix(M["suffix"]) if M["suffix"] else s, Sector=M["sector_of"].get(s, "Other"),
                Bias=r["bias"], P_up=round(r["p"] * 100) if r["n"] and r["p"] == r["p"] else None,
                Normal=round(r["base"] * 100), Days=r["n"], AvgNextDay=round(r["avg"] * 100, 2) if r["avg"] == r["avg"] else None,
                Close=round(r["close"], 2), Support=round(r["s1"], 2), Pivot=round(r["pivot"], 2), Resistance=round(r["r1"], 2),
                ATRpct=round(r["atr_pct"], 1), AsOf=r["asof"]))
        except Exception:
            continue
    return pd.DataFrame(rows)

def show_next(st, df):
    r = nextday(df)
    st.subheader("Next-day outlook")
    st.write(f"**{r['bias']}** (based on the candle dated {r['asof']})")
    if r["cur"] == 0 or r["n"] == 0:
        st.write("Today's indicators are mixed, so there is no basis for a next-day call.")
    else:
        f = lambda v: "n/a" if v != v else f"{v * 100:.0f}%"
        st.write(f"After {r['n']} past days with a similar setup, the stock closed higher the next day {f(r['p'])} of the time, versus {f(r['base'])} on all days. Older 70%: {f(r['p_old'])}. Newest 30%: {f(r['p_new'])}. Average next-day move after such days: {r['avg'] * 100:+.2f}%.")
    c = st.columns(4)
    c[0].metric("Support (S1)", f"{r['s1']:.2f}"); c[1].metric("Pivot", f"{r['pivot']:.2f}")
    c[2].metric("Resistance (R1)", f"{r['r1']:.2f}"); c[3].metric("Typical daily range", f"{r['atr_pct']:.1f}%")
    st.caption("Odds come from this stock's own history, not a forecast. An up or down bias is shown only if the odds beat normal by at least 4 points, with 40 or more past days, in both the older and newer parts of the history. Gaps at the open can ignore these levels. Run it after the market closes so the last candle is complete.")

def nextday_tab(st, market):
    M = MK[market]
    @st.cache_data(ttl=600, show_spinner=False)
    def cached(mk, big):
        m = MK[mk]; return scan_next(m["core"] + m["extra"] if big else m["core"], mk)
    pick = st.radio("Stocks to scan", [f"Large caps ({len(M['core'])})", f"Large + mid caps ({len(M['core']) + len(M['extra'])})"], horizontal=True, key="nx")
    sectors = st.multiselect("Sectors (leave empty for all)", list(M["sectors"]), key="nxs")
    st.caption("Looks for stocks whose own history shows a clear lean for tomorrow. Many days there will be few or none.")
    if not st.button("Find next-day leaders", type="primary"): return
    with st.spinner("Downloading daily data and measuring odds"):
        df = cached(market, pick.startswith("Large +"))
    if df.empty: st.error("No data came back. Yahoo may be blocking for a few minutes. Try again later."); return
    if sectors: df = df[df.Sector.isin(sectors)]
    st.write(f"Scanned {len(df)} stocks. Latest candle date in data: {df.AsOf.max()}.")
    cols = {"P_up": "Closed higher after similar days %", "Normal": "Closed higher on all days %", "Days": "Similar days", "AvgNextDay": "Avg next-day move %", "ATRpct": "Typical range %"}
    for bias, title, asc in [("Up bias", "Up bias for tomorrow", False), ("Down bias", "Down bias for tomorrow", True)]:
        t = df[df.Bias == bias].assign(g=lambda x: (x.P_up - x.Normal).abs()).sort_values("g", ascending=False).head(10).drop(columns=["g", "AsOf", "Bias"])
        st.subheader(title)
        if t.empty: st.info("No stock shows a clear edge right now.")
        else: st.dataframe(t.rename(columns=cols), hide_index=True)
    st.caption("A bias means the odds beat normal by 4+ points in both the older and newer history. This is a lean, not a promise, and a lean of 55-60% still loses on many days. Size positions small and use a stop.")

# ---------- UI ----------
def filter_ui(st, key):
    st.markdown("**Extra filters.** Turn them on one at a time and compare the results.")
    c = st.columns(3)
    return dict(market=c[0].checkbox("Nifty trend agrees", key=key + "m"),
                rs=c[1].checkbox("Stronger or weaker than Nifty", key=key + "r"),
                htf=c[2].checkbox("Higher timeframe agrees", key=key + "h"))

def line(st, lab, s):
    if s: st.write(f"**{lab}:** {s['trades']} trades, {s['win']:.0f}% winners, avg {s['avgR']:+.2f}R, profit factor {s['pf']:.2f}, worst drawdown {s['dd']:.1f}R")
    else: st.write(f"**{lab}:** no trades")

def single(st, market):
    q = st.text_input("Stock name", placeholder="e.g. Reliance, TCS (India) or Apple, Tesla (US)")
    mode = st.radio("Style", ["Intraday", "Swing"], horizontal=True, key="m1")
    opts = filter_ui(st, "s")
    if not (q and st.button("Analyse", type="primary")): return
    sym, name = resolve(q, market)
    with st.spinner(f"Fetching {sym}"):
        df = load(sym, mode)
        nifty = None
        if any(opts.values()):
            try: nifty = load(MK[market]["bench"], mode)
            except Exception: st.warning("Could not load Nifty data, so Nifty filters are off.")
    if len(df) < 120: st.error(f"Not enough data for {sym}. Try the exact NSE symbol."); return
    d = build(df, mode, nifty, opts); k, lv = verdict(d, mode); x = d.iloc[-1]
    word = {1: "Long", -1: "Short", 0: "Wait"}[k] + (f" (leaning {'Long' if x.score > 0 else 'Short'})" if k == 0 and x.score != 0 else "")
    st.header(f"{sym}: {word}")
    st.write(f"Last price {x.Close:.2f} at {d.index[-1]}. Score {x.score:+.0f}, ADX {x.adx:.0f}, RSI {x.rsi:.0f}.")
    if any(opts.values()) and int(np.sign(x.score)) != 0 and k == 0 and abs(x.score) >= 3 and x.adx >= 20:
        st.info("The basic checks agree, but one of your extra filters blocks this trade.")
    if lv:
        if k == 0: st.warning("Not all checks agree, so this is a weaker setup. The levels below are for reference only.")
        kk = 1 if lv["target"] > lv["entry"] else -1; atr_v = float(x.atr)
        lp = live_price(sym)
        if lp:
            moved = (lp - lv["entry"]) * kk / atr_v
            sh = lp - lv["entry"]; lv = dict(entry=lp, stop=lv["stop"] + sh, target=lv["target"] + sh)
            st.caption(f"Entry uses the live price {lp:.2f}. The last candle in the data closed at {x.Close:.2f}.")
            if moved > 0.25: st.warning("Price has already moved in the trade's direction by more than a quarter ATR, so chasing it worsens your risk and reward. Consider waiting for a pullback.")
        else:
            st.caption("Live price was not available, so entry uses the last candle's close.")
        a, b, c3, d4 = st.columns(4)
        a.metric("Entry", f"{lv['entry']:.2f}"); d4.metric("Do not enter beyond", f"{lv['entry'] + kk * 0.25 * atr_v:.2f}")
        b.metric("Stop", f"{lv['stop']:.2f}"); c3.metric("Target", f"{lv['target']:.2f}")
    else:
        st.info("Indicators conflict, the market is not trending (ADX below 20), or a filter blocks the trade. No trade is a valid call.")
    t = backtest(d, mode); cut = int(len(t) * 0.7)
    st.subheader("How these rules did on this stock")
    line(st, "All history", stats(t)); line(st, "Older 70% (rules tuned here)", stats(t.iloc[:cut])); line(st, "Newest 30% (honest test)", stats(t.iloc[cut:]))
    if any(opts.values()):
        t0 = backtest(build(df, mode), mode); c0 = int(len(t0) * 0.7)
        st.subheader("Same stock without your extra filters")
        line(st, "All history", stats(t0)); line(st, "Newest 30%", stats(t0.iloc[c0:]))
        st.caption("A filter is only useful if the newest 30% result gets better, not just the older part.")
    st.caption("R = amount risked per trade. Includes 0.05% cost per trade. Few trades means the numbers are unreliable.")
    if mode == "Swing": show_next(st, df)
    st.line_chart(d[["Close", "ef", "es"]].tail(150))

def scanner(st, market):
    M = MK[market]; nc, ne = len(M["core"]), len(M["extra"])
    @st.cache_data(ttl=120, show_spinner=False)
    def cached(mode, big, f, mk):
        m = MK[mk]
        return scan(mode, m["core"] + m["extra"] if big else m["core"], dict(market=f[0], rs=f[1], htf=f[2]), mk)
    c1, c2 = st.columns(2)
    mode = c1.radio("Style", ["Swing", "Intraday"], horizontal=True, key="m2")
    pick = c2.radio("Stocks to scan", [f"Large caps ({nc})", f"Large + mid caps ({nc + ne})"], horizontal=True)
    big = pick.startswith("Large +")
    sectors = st.multiselect("Sectors (leave empty for all)", list(M["sectors"]))
    o = filter_ui(st, "c")
    c3, c4 = st.columns(2)
    cap = c3.slider("Max stocks per sector in each list", 1, 10, 3)
    need = c4.checkbox("Hide stocks where these rules lost money in the past", value=True)
    st.caption("Intraday works best with large caps. Scanning more stocks takes a few minutes.")
    if not st.button("Scan now", type="primary"): return
    with st.spinner("Downloading and scoring stocks"):
        df = cached(mode, big, (o['market'], o['rs'], o['htf']), market)
    if df.empty: st.error("No data came back. Yahoo may be blocking for a few minutes. Try again later."); return
    st.write(f"Scanned {len(df)} stocks. Prices are from the latest candle at {df.AsOf.max()}. Yahoo can be 15 minutes or more behind your broker, so check the live price before acting.")
    cols = {"Entry": "Entry (last price)", "Chase": "Do not enter beyond", "PastTrades": "Past trades", "PastWin": "Past win %", "PastAvgR": "Past avg R"}
    for side, title in [(1, "Top 10 bullish"), (-1, "Top 10 bearish (short or avoid)")]:
        t = top(df, side, 10, need, cap, sectors)
        st.subheader(title)
        if t.empty: st.info("Nothing passes the filters right now.")
        else: st.dataframe(t.rename(columns=cols), hide_index=True)
    short = ("Short selling in delivery is not allowed in India, so bearish names are mainly for intraday shorts or to avoid."
             if market.startswith("India") else "Shorting in the US needs a margin account that allows it, so bearish names can also simply mean avoid.")
    st.caption("Signal = all checks agree and the market is trending (listed first). Leaning = direction is positive or negative but not all checks agree, so treat its levels as a weaker setup. Entry is the last price in the data, not a live quote. If your live price is already beyond the do-not-enter level, skip the trade or wait for a pullback. Past columns show how the same rules did on that stock before. " + short + " Symbols Yahoo does not recognise are skipped.")

def main():
    import streamlit as st
    st.set_page_config(page_title="Setup Reader", layout="wide")
    st.title("Setup Reader")
    st.caption("Live data from Yahoo Finance (can lag a few minutes). Not investment advice.")
    market = st.radio("Market", list(MK), horizontal=True)
    t1, t3, t2 = st.tabs(["Top 10 scanner", "Next-day outlook", "Single stock"])
    with t1: scanner(st, market)
    with t3: nextday_tab(st, market)
    with t2: single(st, market)

if __name__ == "__main__":
    main()
