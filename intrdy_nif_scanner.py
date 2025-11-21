# nifty_zone_predictor.py
"""
Upgraded Supply/Demand zone extractor + bias & probability model for intraday use.
Author: ChatGPT (adapted for your existing script)
Requirements:
  pip install yfinance pandas numpy scipy
Usage:
  python nifty_zone_predictor.py
It will prompt for a date (dd-mm-yyyy) or ENTER for yesterday.
"""

import warnings
warnings.filterwarnings("ignore")
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.stats import zscore

# -------------------------
# Helper utils
# -------------------------
def to_float(x):
    return float(np.array(x, dtype=float).astype(float).item())

def pct(a, b):
    return (a - b) / b * 100.0

# -------------------------
# CONFIG (tune these)
# -------------------------
SYMBOL = "^NSEI"
INTERVAL = "5m"
LOOKBACK_DAYS = 14

# Minimum quality settings
MIN_ZONE_WIDTH = 6            # minimum absolute points between zone bounds
MIN_IMPULSE = 30              # points price must move after zone (used in filtering)
TOP_ZONES = 3                 # top N demand & supply zones to return
MERGE_PCT = 0.02 / 100.0   # 0.02% = ~5 pts for Nifty
VOLUME_SPIKE_Z = 1.25         # z-score threshold to call a volume spike
WICK_BODY_WEIGHT = 0.6        # weighting between wick vs body for rejection
MIN_TOUCHES = 1               # minimum touches to consider zone
EMA_SHORT = 20                # for HTF trend (on 15m)
EMA_LONG = 50                 # for HTF trend (on 1h)
# Probability model weights (you can tune)
WEIGHTS = {
    "strength": 0.35,
    "volume": 0.20,
    "confluence": 0.25,
    "htf_trend": 0.10,
    "gap": 0.10
}

# -------------------------
# Read target date from user
# -------------------------
user_date = input("Enter date (dd-mm-yyyy) or press ENTER for yesterday: ").strip()

if user_date:
    target_date = datetime.strptime(user_date, "%d-%m-%Y").date()
else:
    target_date = (datetime.today() - timedelta(days=1)).date()

print(f"\nExtracting levels for date: {target_date}\n")

# -------------------------
# Download data
# -------------------------
# We'll fetch a little extra (LOOKBACK_DAYS) to compute HTF etc.
data = yf.download(SYMBOL, period=f"{LOOKBACK_DAYS}d", interval=INTERVAL, progress=False)
if data.empty:
    raise Exception("No data downloaded. Check internet or symbol.")

# Remove tzinfo for consistent date matching
try:
    data = data.tz_localize(None)
    # ---- FIX: Flatten multiindex columns from yfinance ----
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [c[0] for c in data.columns]
    # ---------------------------------------------------------

    # >>> ADD THESE 3 LINES HERE <<<
    print("\n--- DEBUG: Data Preview ---")
    print(data.head())
    print(data.columns)
    print(data.tail())
    print("---------------------------\n")
    # >>> END DEBUG <<<
except Exception:
    pass

# Helper: split prev day and next day (intraday)
prev_day_mask = data.index.date == target_date
next_day_mask = data.index.date == (target_date + timedelta(days=1))

prev = data.loc[prev_day_mask].copy()
next_day = data.loc[next_day_mask].copy()

if prev.empty:
    raise Exception(f"No intraday data for {target_date}")

# Also prepare HTF by resampling
data_15m = data.resample("15T").agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}).dropna()
data_1h  = data.resample("60T").agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}).dropna()

# -------------------------
# Basic PDH/PDL/PDO/PDC
# -------------------------
highs = np.array(prev["High"], dtype=float)
lows = np.array(prev["Low"], dtype=float)
opens = np.array(prev["Open"], dtype=float)
closes = np.array(prev["Close"], dtype=float)

PDH = float(np.max(highs))
PDL = float(np.min(lows))
PDO = float(opens[0])
PDC = float(closes[-1])

# -------------------------
# Raw zone detection (swing-based)
# -------------------------
def detect_raw_zones(df, strength=3):
    """
    Detect swing highs (supply) and swing lows (demand).
    For each index i, if high is local maximum over window -> supply
                       if low is local minimum -> demand
    Return lists of (low_bound, high_bound, center_idx, touch_count_est)
    """
    lows = df["Low"].values.astype(float)
    highs = df["High"].values.astype(float)
    opens = df["Open"].values.astype(float)
    closes = df["Close"].values.astype(float)
    n = len(lows)

    demand = []
    supply = []

    for i in range(strength, n - strength):
        window_lows = lows[i-strength:i+strength+1]
        window_highs = highs[i-strength:i+strength+1]

        # demand (local low)
        if lows[i] == np.min(window_lows):
            # make zone from low wick up to nearby candle open/body high
            low_z = float(lows[i])
            # Use subsequent candle open/close as top of zone (conservative)
            high_z = float(max(opens[i], closes[i], highs[i]))
            demand.append({"low": low_z, "high": high_z, "idx": df.index[i]})

        # supply (local high)
        if highs[i] == np.max(window_highs):
            high_z = float(highs[i])
            low_z = float(min(opens[i], closes[i], lows[i]))
            supply.append({"low": low_z, "high": high_z, "idx": df.index[i]})

    return demand, supply

raw_demand, raw_supply = detect_raw_zones(prev, strength=3)

# -------------------------
# Simple Order Block detection (heuristic)
# -------------------------
def detect_order_blocks(df, lookback=6):
    """
    Heuristic detection of bullish/bearish order blocks:
    - Bullish OB: bearish candle (body down) followed by strong bullish impulse that breaks structure
    - Bearish OB: bullish candle followed by strong bearish impulse
    This is a heuristic to add confluence.
    Returns list of OBs: {'type': 'bull'/'bear', 'low':..., 'high':..., 'idx': ...}
    """
    ob_list = []
    o = df["Open"].values.astype(float)
    c = df["Close"].values.astype(float)
    h = df["High"].values.astype(float)
    l = df["Low"].values.astype(float)

    for i in range(1, len(df)-1):
        prev_body = c[i-1] - o[i-1]
        curr_body = c[i] - o[i]

        # bullish order-block: previous candle bearish and current bullish strong
        if prev_body < 0 and curr_body > 0 and abs(curr_body) > abs(prev_body)*0.6:
            low_z = min(o[i-1], c[i-1], l[i-1])
            high_z = max(o[i-1], c[i-1], h[i-1])
            ob_list.append({"type":"bull", "low": float(low_z), "high": float(high_z), "idx": df.index[i-1]})

        # bearish order-block
        if prev_body > 0 and curr_body < 0 and abs(curr_body) > abs(prev_body)*0.6:
            low_z = min(o[i-1], c[i-1], l[i-1])
            high_z = max(o[i-1], c[i-1], h[i-1])
            ob_list.append({"type":"bear", "low": float(low_z), "high": float(high_z), "idx": df.index[i-1]})

    return ob_list

order_blocks = detect_order_blocks(prev)

# -------------------------
# Fair Value Gaps (FVG) detection
# -------------------------
def detect_fvg(df):
    """
    Detect simple fair value gaps: a gap between two consecutive candles' bodies.
    If candle i low > candle i-1 high -> up gap (FVG)
    If candle i high < candle i-1 low -> down gap
    Returns FVG list
    """
    fvg = []
    o = df["Open"].values.astype(float)
    c = df["Close"].values.astype(float)
    h = df["High"].values.astype(float)
    l = df["Low"].values.astype(float)

    for i in range(1, len(df)):
        prev_body_top = max(o[i-1], c[i-1])
        prev_body_bot = min(o[i-1], c[i-1])
        curr_body_top = max(o[i], c[i])
        curr_body_bot = min(o[i], c[i])

        # Up gap (bullish FVG)
        if curr_body_bot > prev_body_top:
            fvg.append({"type":"bull", "low": prev_body_top, "high": curr_body_bot, "idx": df.index[i]})
        # Down gap (bearish FVG)
        if curr_body_top < prev_body_bot:
            fvg.append({"type":"bear", "low": curr_body_top, "high": prev_body_bot, "idx": df.index[i]})

    return fvg

fvgs = detect_fvg(prev)

# -------------------------
# Zone cleansing, clustering & scoring
# -------------------------
def zone_overlap(z1, z2):
    # True if zones overlap (range overlap)
    return not (z1["high"] < z2["low"] or z2["high"] < z1["low"])

def merge_zones(zones):
    # zones: list of dicts with low, high
    if not zones:
        return []
    # sort by low
    zones_sorted = sorted(zones, key=lambda z: z["low"])
    merged = []
    cur = zones_sorted[0].copy()
    for z in zones_sorted[1:]:
        # if overlap or within MERGE_PCT of each other, merge
        gap_pct = abs(z["low"] - cur["high"]) / cur["high"] if cur["high"] != 0 else 0
        if zone_overlap(cur, z) or gap_pct <= MERGE_PCT:
            cur["low"] = min(cur["low"], z["low"])
            cur["high"] = max(cur["high"], z["high"])
            # merge index list
            cur.setdefault("idxs", set())
            cur.setdefault("sources", set())
            cur["idxs"].add(z.get("idx", None))
            if "source" in z:
                cur["sources"].add(z["source"])
        else:
            merged.append(cur)
            cur = z.copy()
    merged.append(cur)
    return merged

def count_zone_touches(zone, df, tol_pct=0.002):  # tolerance 0.2%
    # count how many times price entered the zone (based on Low/High)
    low, high = zone["low"], zone["high"]
    tol_low = low * (1 - tol_pct)
    tol_high = high * (1 + tol_pct)
    touches = ((df["Low"] <= tol_high) & (df["High"] >= tol_low)).sum()
    return int(touches)

def wick_body_score(idx, df):
    # compute wick vs body dominance for the candle at idx (index label)
    # return value 0..1 where higher = stronger wick rejection
    try:
        row = df.loc[idx]
    except KeyError:
        return 0.5
    o, c, h, l = row["Open"], row["Close"], row["High"], row["Low"]
    body = abs(c - o)
    upper_wick = h - max(c, o)
    lower_wick = min(c, o) - l
    # bigger wick relative to body increases rejection score
    wick = max(upper_wick, lower_wick)
    denom = body + wick
    if denom == 0:
        return 0.5
    return float(wick / denom)

def avg_volume_zscore(df):
    # return volume z-scores for the day's candles
    vols = df["Volume"].values.astype(float)
    if len(vols) < 3:
        return np.zeros_like(vols)
    return zscore(vols, nan_policy='omit')

def score_and_filter(raw_zones, df, next_day_df, zone_type="demand"):
    """
    For each zone produce:
       - width
       - touches
       - wick/body score at creation index (how strong rejection)
       - volume spike at creation (zscore)
       - impulse (how much next day moved away)
    Then compute a final strength metric (0..1)
    """
    results = []
    vol_z = avg_volume_zscore(df)

    for z in raw_zones:
        zone = {"low": z["low"], "high": z["high"], "idx": z.get("idx")}
        # width
        width = abs(zone["high"] - zone["low"])
        if width > 80:  # do not allow huge zones
            continue

        # touches (how many times price visited in prev day)
        touches = count_zone_touches(zone, df, tol_pct=0.002)

        # wick/body score (if idx exists)
        wbs = wick_body_score(zone["idx"], df)

        # volume spike at the creation candle (approx)
        try:
            pos = df.index.get_loc(zone["idx"])
            vol_zscore = float(vol_z[pos]) if pos < len(vol_z) else 0.0
        except Exception:
            vol_zscore = 0.0

        # next-day impulse: how much price in next day moved away from zone
        imp = 0.0
        if not next_day_df.empty:
            closes_next = next_day_df["Close"].values.astype(float)
            # movement above zone_high
            max_move_up = np.max(closes_next) - zone["high"]
            max_move_down = zone["low"] - np.min(closes_next)
            imp = max(max_move_up, max_move_down)

        if imp < MIN_IMPULSE:
            # optional: skip if next day didn't show impulse. But sometimes you want zones for intraday same day.
            # We'll allow but penalize in scoring.
            pass

        # base score components
        # normalize: touches (scale 0..1 maybe 0..5), width scaled (bigger width often weaker but we keep moderate)
        touches_score = min(touches, 6) / 6.0
        vol_score = 1.0 if vol_zscore >= VOLUME_SPIKE_Z else (max(0.0, 1 - abs(vol_zscore)/3.0))
        wick_score = wbs  # 0..1
        imp_score = min(max(imp / 100.0, 0.0), 1.0)  # scale impulse to 0..1 assuming 100 pts = strong

        # combine into strength metric (weights chosen heuristically)
        strength = (0.35 * touches_score) + (0.25 * wick_score) + (0.25 * vol_score) + (0.15 * imp_score)

        results.append({
            "low": zone["low"],
            "high": zone["high"],
            "idx": zone["idx"],
            "width": width,
            "touches": touches,
            "vol_zscore": vol_zscore,
            "wick_score": wick_score,
            "impulse": imp,
            "strength": strength,
            "type": zone_type
        })

    # sort by strength desc and return
    results_sorted = sorted(results, key=lambda x: x["strength"], reverse=True)
    return results_sorted

# Score demand and supply raw zones
scored_demand = score_and_filter(raw_demand, prev, next_day, zone_type="demand")
scored_supply = score_and_filter(raw_supply, prev, next_day, zone_type="supply")

# Also include order blocks and fvgs as potential zones (for confluence)
# Convert order_blocks and fvgs into same dict format and score lightly
ob_zones = []
for ob in order_blocks:
    ob_zones.append({"low": ob["low"], "high": ob["high"], "idx": ob["idx"], "type":"ob"})

fvg_zones = []
for f in fvgs:
    fvg_zones.append({"low": f["low"], "high": f["high"], "idx": f["idx"], "type":"fvg"})

# Merge similar zones across demand/supply + OB + FVG to reduce noise
all_candidate_zones = []
for s in scored_demand:
    s2 = s.copy()
    s2["source"] = "demand"
    all_candidate_zones.append(s2)
for s in scored_supply:
    s2 = s.copy()
    s2["source"] = "supply"
    all_candidate_zones.append(s2)
for ob in ob_zones:
    ob2 = ob.copy()
    ob2["source"] = "ob"
    all_candidate_zones.append(ob2)
for f in fvg_zones:
    f2 = f.copy()
    f2["source"] = "fvg"
    all_candidate_zones.append(f2)

# Consolidate zone dicts (we'll create minimal zone entries for merging)
zone_minimal = []
for z in all_candidate_zones:
    zone_minimal.append({
        "low": z["low"],
        "high": z["high"],
        "idx": z.get("idx"),
        "source": z.get("source", "cand")
    })

merged = merge_zones(zone_minimal)

# After merging, compute combined strength/confluence:
def compute_confluence(merged_zone, scored_demand, scored_supply, ob_zones, fvg_zones):
    low, high = merged_zone["low"], merged_zone["high"]
    conf = {"demand":0, "supply":0, "ob":0, "fvg":0}
    # check overlaps
    for s in scored_demand:
        if not (s["high"] < low or s["low"] > high):
            conf["demand"] += 1
    for s in scored_supply:
        if not (s["high"] < low or s["low"] > high):
            conf["supply"] += 1
    for ob in ob_zones:
        if not (ob["high"] < low or ob["low"] > high):
            conf["ob"] += 1
    for f in fvg_zones:
        if not (f["high"] < low or f["low"] > high):
            conf["fvg"] += 1
    # confluence count
    total_conf = conf["demand"] + conf["supply"] + conf["ob"] + conf["fvg"]
    return conf, total_conf

final_zones = []
for mz in merged:
    conf, total_conf = compute_confluence(mz, scored_demand, scored_supply, ob_zones, fvg_zones)
    # estimate touches and wick_score using closest scored zone if exists
    # find nearest candidate in scored lists
    associated = None
    for s in (scored_demand + scored_supply):
        # overlap test
        if not (s["high"] < mz["low"] or s["low"] > mz["high"]):
            associated = s
            break

    touches = associated["touches"] if associated else count_zone_touches(mz, prev)
    wick_score = associated["wick_score"] if associated else 0.5
    vol_zscore = associated["vol_zscore"] if associated else 0.0
    width = abs(mz["high"] - mz["low"])

    base_strength = (0.4 * min(touches, 6)/6.0) + (0.35 * wick_score) + (0.25 * (1 if vol_zscore >= VOLUME_SPIKE_Z else max(0.0, 1 - abs(vol_zscore)/3.0)))
    # penalize tiny width too small or huge width
    if width < MIN_ZONE_WIDTH:
        continue

    final_zones.append({
        "low": mz["low"],
        "high": mz["high"],
        "width": width,
        "touches": touches,
        "wick_score": wick_score,
        "vol_zscore": vol_zscore,
        "confluence": conf,
        "confluence_count": total_conf,
        "base_strength": base_strength
    })

# Sort zones by combined metric (base_strength + confluence weight)
final_zones_sorted = sorted(final_zones, key=lambda z: (z["base_strength"] + 0.15*z["confluence_count"]), reverse=True)

# Keep top supply & demand separately
top_demands = [z for z in final_zones_sorted if (z["low"] + z["high"]) / 2 <= PDC][:TOP_ZONES]
top_supplies = [z for z in final_zones_sorted if (z["low"] + z["high"]) / 2 > PDC][:TOP_ZONES]

# If not enough by above splitting, just take top by score
if len(top_demands) < TOP_ZONES:
    top_demands = final_zones_sorted[:TOP_ZONES]
if len(top_supplies) < TOP_ZONES:
    top_supplies = final_zones_sorted[:TOP_ZONES]

# -------------------------
# HTF Trend Filter
# -------------------------
def htf_trend(data_15m, data_1h):
    # compute EMA trend direction on HTFs
    data_15m["ema_short"] = data_15m["Close"].ewm(span=EMA_SHORT, adjust=False).mean()
    data_1h["ema_long"] = data_1h["Close"].ewm(span=EMA_LONG, adjust=False).mean()

    # last values
    last_15m = data_15m["Close"].iloc[-1]
    ema_15m = data_15m["ema_short"].iloc[-1]
    last_1h = data_1h["Close"].iloc[-1]
    ema_1h = data_1h["ema_long"].iloc[-1]

    short_trend = "up" if last_15m > ema_15m else "down"
    long_trend = "up"  if last_1h > ema_1h else "down"
    return short_trend, long_trend

short_trend, long_trend = htf_trend(data_15m.copy(), data_1h.copy())

# -------------------------
# Pre-market gap & open bias
# -------------------------
# Use next_day first candle open if available as 'market open' reference
if not next_day.empty:
    next_day_open = float(next_day["Open"].iloc[0])
else:
    # fallback to today's first available or previous close
    next_day_open = float(PDC)

gap = next_day_open - PDC
gap_pct = pct(next_day_open, PDC)

# simple gap bias
gap_bias = None
if gap_pct > 0.06:   # >0.06% up gap
    gap_bias = "bullish_gap"
elif gap_pct < -0.06:
    gap_bias = "bearish_gap"
else:
    gap_bias = "neutral_gap"

# -------------------------
# Probability model (very simple linear combination -> 0..100)
# -------------------------
def zone_probability(zone, short_trend, long_trend, gap_bias):
    """
    Combine signals:
     - strength (base_strength)
     - volume presence (vol_zscore)
     - confluence_count
     - HTF trend alignment
     - gap alignment
    Return: probability_of_reversal (0..100), probability_of_breakout (0..100)
    """
    # normalize components to 0..1
    strength = min(max(zone["base_strength"], 0.0), 1.0)
    vol = 1.0 if zone["vol_zscore"] >= VOLUME_SPIKE_Z else max(0.0, 1 - abs(zone["vol_zscore"])/3.0)
    conf = min(zone["confluence_count"] / 4.0, 1.0)  # 4 or more confluence = strong
    # HTF alignment: if HTF (long_trend) matches expected direction (supply vs demand)
    zone_mid = (zone["low"] + zone["high"]) / 2.0
    # if zone is above open/close -> treated as supply zone (likely short)
    expected_direction = "bear" if zone_mid > PDC else "bull"
    htf_align = 1.0 if ((expected_direction == "bull" and long_trend == "up") or (expected_direction == "bear" and long_trend == "down")) else 0.0
    # gap alignment: if gap direction supports expected_direction
    if gap_bias == "bullish_gap" and expected_direction == "bull":
        gap_align = 1.0
    elif gap_bias == "bearish_gap" and expected_direction == "bear":
        gap_align = 1.0
    else:
        gap_align = 0.0

    # raw weighted sum
    raw = (WEIGHTS["strength"] * strength +
           WEIGHTS["volume"] * vol +
           WEIGHTS["confluence"] * conf +
           WEIGHTS["htf_trend"] * htf_align +
           WEIGHTS["gap"] * gap_align)

    # probability of reversal increases with strength and confluence; probability of breakout is inverse-ish
    prob_reversal = min(max(raw * 100.0, 0.0), 100.0)
    prob_breakout = max(0.0, 100.0 - prob_reversal)  # simple heuristic

    return prob_reversal, prob_breakout

# Compute probabilities for top zones
for z in top_demands + top_supplies:
    pr, pb = zone_probability(z, short_trend, long_trend, gap_bias)
    z["prob_reversal"] = round(pr, 1)
    z["prob_breakout"] = round(pb, 1)

# -------------------------
# FINAL DAILY BIAS (aggregate)
# -------------------------
# Use weighted average of top zones' expected direction and HTF/gap
score_bull = 0.0
score_bear = 0.0
for z in top_demands + top_supplies:
    mid = (z["low"] + z["high"]) / 2.0
    if mid <= PDC:
        score_bull += z["prob_reversal"]
    else:
        score_bear += z["prob_reversal"]

# incorporate HTF
if long_trend == "up":
    score_bull += 10
else:
    score_bear += 10

# incorporate gap
if gap_bias == "bullish_gap":
    score_bull += 8
elif gap_bias == "bearish_gap":
    score_bear += 8

if score_bull > score_bear:
    daily_bias = "Bullish"
elif score_bear > score_bull:
    daily_bias = "Bearish"
else:
    daily_bias = "Neutral"

# -------------------------
# PRINT / OUTPUT
# -------------------------
def print_zone(z, side="Demand"):
    print(data.head())
    print(data.columns)
    print(data.tail())

    print(f"--- {side} Zone ---")
    print(f"Range: {z['low']:.2f} → {z['high']:.2f}  width: {z['width']:.2f}")
    print(f"Touches: {z['touches']}  wick_score:{z['wick_score']:.2f}  vol_z:{z['vol_zscore']:.2f}")
    print(f"Confluence: {z['confluence']}  confluence_count:{z['confluence_count']}")
    print(f"Base strength: {z['base_strength']:.3f}")
    print(f"P(reversal)={z['prob_reversal']}%  P(breakout)={z['prob_breakout']}%")
    print()

print("====================================")
print(f" TOP LEVELS FOR {target_date} (Enhanced)")
print("====================================\n")
print(f"PDH : {PDH:.2f}")
print(f"PDL : {PDL:.2f}")
print(f"PDO : {PDO:.2f}")
print(f"PDC : {PDC:.2f}\n")

print(f"HTF short(15m): {short_trend}   HTF long(1h): {long_trend}")
print(f"Next day open: {next_day_open:.2f}  Gap: {gap:.2f} ({gap_pct:.3f}%)  Gap bias: {gap_bias}")
print(f"Daily Bias (aggregate): {daily_bias}\n")

print("Top Demand Zones (likely buy reactions):")
for z in top_demands:
    print_zone(z, side="Demand")

print("Top Supply Zones (likely sell reactions):")
for z in top_supplies:
    print_zone(z, side="Supply")

# Optional: show merged zones as a table
df_out = pd.DataFrame(top_demands + top_supplies)
if not df_out.empty:
    df_out = df_out[["low","high","width","touches","wick_score","vol_zscore","confluence_count","base_strength","prob_reversal","prob_breakout"]]
    print("\nSummary table:")
    print(df_out.to_string(index=False))

# Save outputs to CSV for later visual inspection
out_csv = f"zones_{SYMBOL.replace('^','')}_{target_date}.csv"
df_out = df_out.drop_duplicates()
df_out.to_csv(out_csv, index=False)
print(f"\nSaved top zones to: {out_csv}")

# End of script
