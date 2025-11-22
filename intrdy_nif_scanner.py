import warnings;
warnings.filterwarnings("ignore")
import yfinance as yf, numpy as np, pandas as pd
from datetime import datetime, timedelta
from statsmodels.robust.scale import mad
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest


def robust_z(x): x = np.asarray(x, float);m = np.nanmedian(x);s = mad(x);return (
                                                                                        x - m) / s if s != 0 else np.zeros_like(
    x)


def skl_z(x): x = np.asarray(x, float).reshape(-1, 1);sc = StandardScaler();return sc.fit_transform(x).flatten() if len(
    x) > 1 else np.zeros(len(x))


def iso_anom(x): x = np.asarray(x, float).reshape(-1, 1);clf = IsolationForest(contamination=0.06,
                                                                               random_state=42);return (
        clf.fit_predict(x) == -1).astype(int)


def vol_z(df):
    v = df["Volume"].values.astype(float)
    if len(v) < 3: return np.zeros(len(v)), np.zeros(len(v)), np.zeros(len(v))
    rz = robust_z(v);
    sz = skl_z(v);
    an = iso_anom(v);
    fz = 0.5 * rz + 0.3 * sz + 0.2 * an
    return fz, rz, sz


def pct(a, b): return (a - b) / b * 100


SY = "^NSEI";
INT = "5m";
LB = 14;
MINW = 6;
MINI = 30;
TOP = 3;
MER = 0.02 / 100;
VZ = 1.25;
ES = 20;
EL = 50
W = {"strength": 0.35, "volume": 0.20, "confluence": 0.25, "htf_trend": 0.10, "gap": 0.10}

u = input("Enter date (dd-mm-yyyy) or press ENTER for yesterday: ").strip()
TD = datetime.strptime(u, "%d-%m-%Y").date() if u else (datetime.today() - timedelta(days=1)).date()

df = yf.download(SY, period=f"{LB}d", interval=INT, progress=False)
if df.empty: raise Exception("No data")
try:
    df = df.tz_localize(None)
    if isinstance(df.columns, pd.MultiIndex): df.columns = [c[0] for c in df.columns]
except:
    pass

pm = df.index.date == TD;
nx = df.index.date == (TD + timedelta(days=1))
pr = df.loc[pm].copy();
nxdf = df.loc[nx].copy()
if pr.empty: raise Exception("No intraday")

d15 = df.resample("15T").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna()
d1 = df.resample("60T").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna()

PDH = float(pr["High"].max());
PDL = float(pr["Low"].min());
PDO = float(pr["Open"].iloc[0]);
PDC = float(pr["Close"].iloc[-1])


def rawzones(df, s=3):
    lo = df["Low"].values.astype(float);
    hi = df["High"].values.astype(float)
    op = df["Open"].values.astype(float);
    cl = df["Close"].values.astype(float);
    n = len(lo)
    d = [];
    sp = []
    for i in range(s, n - s):
        if lo[i] == lo[i - s:i + s + 1].min(): d.append(
            {"low": float(lo[i]), "high": float(max(op[i], cl[i], hi[i])), "idx": df.index[i]})
        if hi[i] == hi[i - s:i + s + 1].max(): sp.append(
            {"low": float(min(op[i], cl[i], lo[i])), "high": float(hi[i]), "idx": df.index[i]})
    return d, sp


rd, rs = rawzones(pr, 3)


def ob(df):
    o = df["Open"].values.astype(float);
    c = df["Close"].values.astype(float)
    h = df["High"].values.astype(float);
    l = df["Low"].values.astype(float);
    o2 = []
    for i in range(1, len(df) - 1):
        pb = c[i - 1] - o[i - 1];
        cb = c[i] - o[i]
        if pb < 0 and cb > 0 and abs(cb) > abs(pb) * 0.6: o2.append(
            {"low": min(o[i - 1], c[i - 1], l[i - 1]), "high": max(o[i - 1], c[i - 1], h[i - 1]),
             "idx": df.index[i - 1]})
        if pb > 0 and cb < 0 and abs(cb) > abs(pb) * 0.6: o2.append(
            {"low": min(o[i - 1], c[i - 1], l[i - 1]), "high": max(o[i - 1], c[i - 1], h[i - 1]),
             "idx": df.index[i - 1]})
    return o2


OBS = ob(pr)


def fvg(df):
    o = df["Open"].values.astype(float);
    c = df["Close"].values.astype(float)
    h = df["High"].values.astype(float);
    l = df["Low"].values.astype(float);
    f = []
    for i in range(1, len(df)):
        pbt = max(o[i - 1], c[i - 1]);
        pbb = min(o[i - 1], c[i - 1])
        cbt = max(o[i], c[i]);
        cbb = min(o[i], c[i])
        if cbb > pbt: f.append({"low": pbt, "high": cbb, "idx": df.index[i]})
        if cbt < pbb: f.append({"low": cbt, "high": pbb, "idx": df.index[i]})
    return f


FVG = fvg(pr)


def ov(a, b): return not (a["high"] < b["low"] or b["high"] < a["low"])


def merge(z):
    if not z: return []
    z = sorted(z, key=lambda x: x["low"]);
    m = [];
    c = z[0].copy()
    for x in z[1:]:
        gp = abs(x["low"] - c["high"]) / (c["high"] if c["high"] != 0 else 1)
        if ov(c, x) or gp <= MER:
            c["low"] = min(c["low"], x["low"]);
            c["high"] = max(c["high"], x["high"])
        else:
            m.append(c);c = x.copy()
    m.append(c);
    return m


def touch(z, df, t=0.002):
    lo, hi = z["low"], z["high"];
    lo2 = lo * (1 - t);
    hi2 = hi * (1 + t)
    return int(((df["Low"] <= hi2) & (df["High"] >= lo2)).sum())


def wbs(i, df):
    try:
        r = df.loc[i]
    except:
        return 0.5
    o, c, h, l = r["Open"], r["Close"], r["High"], r["Low"]
    b = abs(c - o);
    uw = h - max(c, o);
    lw = min(c, o) - l;
    w = max(uw, lw);
    d = b + w
    return w / d if d != 0 else 0.5


def score(rz, df, nd, tp):
    out = [];
    vz, rz2, sz2 = vol_z(df)
    for z in rz:
        lo, hi = z["low"], z["high"];
        idx = z["idx"];
        wd = abs(hi - lo)
        if wd > 80: continue
        tc = touch(z, df);
        wb = wbs(idx, df)
        try:
            p = df.index.get_loc(idx);v = vz[p]
        except:
            v = 0
        imp = 0
        if not nd.empty:
            c = nd["Close"].values.astype(float)
            imp = max(np.max(c) - hi, lo - np.min(c))
        ts = min(tc, 6) / 6;
        vs = 1 if v >= VZ else max(0, 1 - abs(v) / 3);
        iscore = min(max(imp / 100, 0), 1)
        s = 0.35 * ts + 0.25 * wb + 0.25 * vs + 0.15 * iscore
        out.append({"low": lo, "high": hi, "idx": idx, "width": wd, "touches": tc, "vol_zscore": v, "wick_score": wb,
                    "impulse": imp, "strength": s, "type": tp})
    return sorted(out, key=lambda x: x["strength"], reverse=True)


SD = score(rd, pr, nxdf, "demand");
SS = score(rs, pr, nxdf, "supply")
OB = [{"low": x["low"], "high": x["high"], "idx": x["idx"], "source": "ob"} for x in OBS]
FG = [{"low": x["low"], "high": x["high"], "idx": x["idx"], "source": "fvg"} for x in FVG]
ALL = []
for s in SD: ALL.append({"low": s["low"], "high": s["high"], "idx": s["idx"], "source": "demand"})
for s in SS: ALL.append({"low": s["low"], "high": s["high"], "idx": s["idx"], "source": "supply"})
ALL += OB + FG
M = merge(ALL)


def conf(z, SD, SS, OB, FG):
    lo, hi = z["low"], z["high"];
    c = {"demand": 0, "supply": 0, "ob": 0, "fvg": 0}
    for s in SD:
        if ov(z, s): c["demand"] += 1
    for s in SS:
        if ov(z, s): c["supply"] += 1
    for s in OB:
        if ov(z, s): c["ob"] += 1
    for s in FG:
        if ov(z, s): c["fvg"] += 1
    return c, sum(c.values())


F = []
for z in M:
    c, tc = conf(z, SD, SS, OB, FG)
    a = None
    for s in SD + SS:
        if ov(z, s): a = s;break
    t = a["touches"] if a else touch(z, pr);
    w = a["wick_score"] if a else 0.5;
    v = a["vol_zscore"] if a else 0;
    wd = abs(z["high"] - z["low"])
    if wd < MINW: continue
    bs = 0.4 * min(t, 6) / 6 + 0.35 * w + 0.25 * (1 if v >= VZ else max(0, 1 - abs(v) / 3))
    F.append({"low": z["low"], "high": z["high"], "width": wd, "touches": t, "wick_score": w, "vol_zscore": v,
              "confluence": c, "confluence_count": tc, "base_strength": bs})

F = sorted(F, key=lambda z: z["base_strength"] + 0.15 * z["confluence_count"], reverse=True)
TDZ = [z for z in F if (z["low"] + z["high"]) / 2 <= PDC][:TOP]
TSZ = [z for z in F if (z["low"] + z["high"]) / 2 > PDC][:TOP]
if len(TDZ) < TOP: TDZ = F[:TOP]
if len(TSZ) < TOP: TSZ = F[:TOP]

d15["ema"] = d15["Close"].ewm(span=ES, adjust=False).mean()
d1["ema"] = d1["Close"].ewm(span=EL, adjust=False).mean()
st = "up" if d15["Close"].iloc[-1] > d15["ema"].iloc[-1] else "down"
lt = "up" if d1["Close"].iloc[-1] > d1["ema"].iloc[-1] else "down"

op = float(nxdf["Open"].iloc[0]) if not nxdf.empty else PDC
g = op - PDC;
gp = pct(op, PDC)
gb = "bullish" if gp > 0.06 else ("bearish" if gp < -0.06 else "neutral")


def prob(z, st, lt, gb):
    s = min(max(z["base_strength"], 0), 1)
    v = z["vol_zscore"];
    v1 = 1 if v >= VZ else max(0, 1 - abs(v) / 3)
    c = min(z["confluence_count"] / 4, 1)
    mid = (z["low"] + z["high"]) / 2;
    exp = "bear" if mid > PDC else "bull"
    ha = 1 if ((exp == "bull" and lt == "up") or (exp == "bear" and lt == "down")) else 0
    ga = 1 if ((gb == "bullish" and exp == "bull") or (gb == "bearish" and exp == "bear")) else 0
    raw = W["strength"] * s + W["volume"] * v1 + W["confluence"] * c + W["htf_trend"] * ha + W["gap"] * ga
    pr = min(max(raw * 100, 0), 100);
    return pr, 100 - pr


for z in TDZ + TSZ:
    pr, pb = prob(z, st, lt, gb)
    z["prob_reversal"] = round(pr, 1);
    z["prob_breakout"] = round(pb, 1)

sb = 0;
sr = 0
for z in TDZ + TSZ:
    if (z["low"] + z["high"]) / 2 <= PDC:
        sb += z["prob_reversal"]
    else:
        sr += z["prob_reversal"]
if lt == "up":
    sb += 10
else:
    sr += 10
if gb == "bullish":
    sb += 8
elif gb == "bearish":
    sr += 8
bias = "Bullish" if sb > sr else ("Bearish" if sr > sb else "Neutral")


def pz(z, s):
    print(f"--- {s} ---")
    print(f"{z['low']:.2f}-{z['high']:.2f} w:{z['width']:.2f}")
    print(f"t:{z['touches']} w:{z['wick_score']:.2f} v:{z['vol_zscore']:.2f}")
    print(f"c:{z['confluence']} C:{z['confluence_count']}")
    print(f"b:{z['base_strength']:.3f} R:{z['prob_reversal']} B:{z['prob_breakout']}")


def magic_levels(df):
    if not isinstance(df, pd.DataFrame): return []
    if df.empty: return []
    if "High" not in df.columns or "Low" not in df.columns or "Close" not in df.columns: return []

    h=df["High"].values.astype(float)
    l=df["Low"].values.astype(float)
    c=df["Close"].values.astype(float)
    idx=df.index
    ml=[]

    for i in range(2,len(df)-2):
        if abs(h[i]-h[i-1])<=h[i]*0.0003:ml.append({"type":"eq_high","level":h[i],"idx":idx[i]})
        if abs(l[i]-l[i-1])<=l[i]*0.0003:ml.append({"type":"eq_low","level":l[i],"idx":idx[i]})

    last=c[-1] if len(c)>0 else None
    if last is not None:
        rn=[50,100,250,500,1000]
        for r in rn:
            lvl=round(last/r)*r
            if abs(last-lvl)<=last*0.001:ml.append({"type":"round","level":lvl,"idx":idx[-1]})

        qvals=[0.25,0.5,0.75]
        p=round(last)
        for q in qvals:
            ql=p+q
            if abs(last-ql)<=last*0.001:ml.append({"type":"quarter","level":ql,"idx":idx[-1]})

        rng=h-l
        adr=np.mean(rng[-20:]) if len(rng)>=20 else np.mean(rng)
        ml.append({"type":"adr_high","level":last+adr,"idx":idx[-1]})
        ml.append({"type":"adr_low","level":last-adr,"idx":idx[-1]})

        tp=(h+l+c)/3
        v=df["Volume"].values.astype(float)
        vwap=np.sum(tp*v)/np.sum(v) if np.sum(v)>0 else last
        band=adr/2
        ml.append({"type":"vwap","level":vwap,"idx":idx[-1]})
        ml.append({"type":"vwap_up","level":vwap+band,"idx":idx[-1]})
        ml.append({"type":"vwap_dn","level":vwap-band,"idx":idx[-1]})

    return ml



ML = magic_levels(pr)
print("\nMagic Levels:")
for m in ML:
    print(f"{m['type']} : {m['level']:.2f}")

print("============ RESULTS ============")
print(f"{TD}  PDH:{PDH:.2f} PDL:{PDL:.2f} PDO:{PDO:.2f} PDC:{PDC:.2f}")
print(f"HTF 15m:{st}  1h:{lt}")
print(f"Open:{op:.2f} Gap:{g:.2f}({gp:.3f}%)  GapBias:{gb}")
print("Bias:", bias)

print("\nDemand Zones:")
for z in TDZ: pz(z, "Demand")
print("\nSupply Zones:")
for z in TSZ: pz(z, "Supply")

outdf = pd.DataFrame(TDZ + TSZ)
if not outdf.empty:
    outdf = outdf[["low", "high", "width", "touches", "wick_score", "vol_zscore", "confluence_count", "base_strength",
                   "prob_reversal", "prob_breakout"]]
    print(outdf.to_string(index=False))

fn = f"zonesStats_{SY.replace('^', '')}_{TD}.csv"
outdf.drop_duplicates().to_csv(fn, index=False)
print("Saved:", fn)
