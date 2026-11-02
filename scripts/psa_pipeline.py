#!/usr/bin/env python3
"""
psa_pipeline.py - Southwest GACC 7-Day Significant Fire Potential pipeline.

Subcommands (run in this order by .github/workflows/forecast.yml):

  fetch-psa   Download SW PSA polygons from the NIFC National_PSA_Current
              feature service -> docs/psa.geojson
  model       Run one Herbie model and reduce it to PSA-day values
              -> partials/<MODEL>.csv
  blend       Median-blend all partials, merge data/fuels.csv, classify each
              PSA-day with config/thresholds.json, apply data/overrides.csv,
              write docs/forecast.json (inputs + spread) and docs/outlook.json
              (categories, triggers, reasons, discussion).

Per PSA-day weather (local MST day; afternoon = 11 am - 6 pm):
  wind  sustained 10 m wind mph (90th pct of PSA grid, afternoon max)
  gust  gust mph (90th pct, afternoon max)
  minrh 2 m RH % (10th pct, afternoon min)
  tmax  2 m temp F (90th pct, daily max)
  qpf   precip in (PSA median, daily total)
  thunder % of PSA with CAPE >= 250 J/kg and measurable precip, afternoon max
  lapse 700-500 mb lapse rate C/km (PSA mean, afternoon max)
  hdwi  Hot-Dry-Windy proxy: VPD(hPa) x wind(m/s), near surface only
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import urllib.parse
import urllib.request
import warnings
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

MST = timezone(timedelta(hours=-7))
AFTERNOON = range(11, 19)
SW_BBOX = (-115.5, -102.5, 30.5, 37.8)
SW_PSAS = ["SW01", "SW02", "SW03", "SW04", "SW05", "SW06N", "SW06S", "SW07",
           "SW08", "SW09", "SW10", "SW11", "SW12", "SW13", "SW14N", "SW14S"]
FIELDS = ["wind", "gust", "minrh", "tmax", "qpf", "thunder", "lapse", "hdwi"]

GRIB = {
    "t2": r":TMP:2 m above ground", "rh2": r":RH:2 m above ground", "d2": r":DPT:2 m above ground",
    "u10": r":UGRD:10 m above ground", "v10": r":VGRD:10 m above ground", "gust": r":GUST:surface",
    "cape": r":CAPE:surface", "t700": r":TMP:700 mb", "t500": r":TMP:500 mb",
    "z700": r":HGT:700 mb", "z500": r":HGT:500 mb", "apcp": r":APCP:surface:",
}
ECMWF = {
    "t2": ":2t:", "d2": ":2d:", "u10": ":10u:", "v10": ":10v:", "gust": ":10fg",
    "cape": ":mucape:", "t700": ":t:700:", "t500": ":t:500:",
    "z700": ":gh:700:", "z500": ":gh:500:", "apcp": ":tp:",
}
NBM = {
    "t2": r":TMP:2 m above ground:\d+ hour fcst:$", "rh2": r":RH:2 m above ground:\d+ hour fcst:$",
    "wspd": r":WIND:10 m above ground:\d+ hour fcst:$", "gust": r":GUST:10 m above ground:\d+ hour fcst:$",
    "apcp": r":APCP:surface:\d+-\d+ hour acc fcst:$",
}
H3 = list(range(0, 61, 3))
MODELS = {
    "HRRR":    (dict(model="hrrr", product="sfc"), list(range(0, 49)), GRIB, 1),
    "RAP":     (dict(model="rap", product="awp130pgrb"), list(range(0, 52, 3)), GRIB, 1),
    "NAMNEST": (dict(model="nam", product="conusnest.hiresf"), H3, GRIB, 6),
    "RRFS":    (dict(model="rrfs", product="prslev", domain="conus", member="control"), H3, GRIB, 6),
    "NBM":     (dict(model="nbm", product="co"), list(range(1, 37)) + list(range(39, 169, 3)), NBM, 1),
    "GFS":     (dict(model="gfs", product="pgrb2.0p25"), list(range(0, 121, 3)) + list(range(126, 181, 6)), GRIB, 6),
    "GEFS":    (dict(model="gefs", product="atmos.5", member="mean"), list(range(0, 181, 6)), GRIB, 6),
    "IFS":     (dict(model="ifs", product="oper"), list(range(0, 145, 3)) + list(range(150, 181, 6)), ECMWF, 6),
    "AIFS":    (dict(model="aifs", product="oper"), list(range(0, 181, 6)), ECMWF, 6),
}
MS_TO_MPH, MM_TO_IN = 2.23694, 1 / 25.4
GRID_MODELS = ("NBM", "GFS")      # these also save daily gridded fields for the discussion maps


# ------------------------------------------------------------------ PSAs
NWCG_PSP = "https://fsapps.nwcg.gov/psp/arcgis/rest/services/npsg/PSA_GACC_KeyRAWS/MapServer"
PSA_SOURCES = [
    # NWCG Predictive Services public map service (layer 1 = PSA boundaries), queried by map area
    ("query", NWCG_PSP + "/1/query"),
    # NIFC feature service (may require a token) and Open Data downloads as fallbacks
    ("query", "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/National_PSA_Current/FeatureServer/0/query"),
    # NIFC Open Data direct downloads (whole country; filtered below)
    ("file", "https://opendata.arcgis.com/api/v3/datasets/national-predictive-service-areas-psa-boundaries/downloads/data?format=geojson&spatialRefId=4326"),
    ("file", "https://data-nifc.opendata.arcgis.com/datasets/nifc::national-predictive-service-areas-psa-boundaries.geojson"),
]


def _psa_code(props):
    """Find an SW PSA id in any attribute, e.g. 'SW06N', 'SW 6N', 'sw06n'."""
    for v in props.values():
        m = re.fullmatch(r"SW[\s_-]*0?(\d{1,2})([NS]?)", str(v).strip().upper())
        if m:
            return f"SW{int(m.group(1)):02d}{m.group(2)}"
    return None


def _psa_name(props):
    for k, v in props.items():
        if "name" in k.lower() and v and not _psa_code({k: v}):
            return str(v)
    return None


def cmd_fetch_psa(a):
    import shapely.geometry as sg
    from shapely.ops import unary_union
    lon0, lon1, lat0, lat1 = SW_BBOX
    tried = []
    gj = None
    for kind, url in PSA_SOURCES:
        if kind == "query":
            q = urllib.parse.urlencode({"where": "1=1", "outFields": "*", "outSR": 4326, "f": "geojson",
                                        "geometry": f"{lon0},{lat0},{lon1},{lat1}", "geometryType": "esriGeometryEnvelope",
                                        "inSR": 4326, "spatialRel": "esriSpatialRelIntersects"})
            url = f"{url}?{q}"
        try:
            txt = _http(url, timeout=300)
            js = json.loads(txt)
            feats = js.get("features")
            if not feats:
                tried.append(f"{url[:100]} -> {txt[:300]}")
                continue
            if any(_psa_code(f.get("properties") or {}) for f in feats):
                gj = js
                print(f"PSA boundaries from {url[:100]}")
                break
            tried.append(f"{url[:100]} -> no SW PSA codes in fields {list((feats[0].get('properties') or {}).keys())}")
        except Exception as e:
            tried.append(f"{url[:100]} -> {e}")
    if gj is None:
        print("Could not get PSA boundaries. Responses:")
        for t in tried:
            print("  " + t)
        sys.exit(1)

    parts, names = {}, {}
    for f in gj["features"]:
        pr = f.get("properties") or {}
        code = _psa_code(pr)
        if not code or not f.get("geometry"):
            continue
        parts.setdefault(code, []).append(sg.shape(f["geometry"]).buffer(0))
        names.setdefault(code, _psa_name(pr))
    feats = []
    for code in sorted(parts):
        geom = unary_union(parts[code]).simplify(0.005, preserve_topology=True)
        pt = geom.representative_point()
        feats.append({"type": "Feature", "geometry": sg.mapping(geom),
                      "properties": {"psa": code, "name": names[code], "lx": round(pt.x, 3), "ly": round(pt.y, 3)}})
    missing = [p for p in SW_PSAS if p not in parts]
    if missing:
        print(f"Warning: no boundary found for {', '.join(missing)}")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump({"type": "FeatureCollection", "features": feats}, fh)
    print(f"Wrote {len(feats)} PSAs to {a.out}")


def load_psas(path):
    import shapely.geometry as sg
    gj = json.load(open(path))
    return {f["properties"]["psa"]: sg.shape(f["geometry"]) for f in gj["features"]}


# ------------------------------------------------------------------ models
def crop(da):
    lon0, lon1, lat0, lat1 = SW_BBOX
    if "latitude" in da.dims and "longitude" in da.dims:
        if float(da.longitude.max()) > 180:
            lon0, lon1 = lon0 + 360, lon1 + 360
        lat_slice = slice(lat1, lat0) if da.latitude[0] > da.latitude[-1] else slice(lat0, lat1)
        da = da.sel(latitude=lat_slice, longitude=slice(lon0, lon1))
    return da


def fetch_field(H, pattern):
    try:
        ds = H.xarray(pattern, remove_grib=True)
    except Exception:
        return None
    if isinstance(ds, list):
        ds = ds[0]
    return crop(ds[list(ds.data_vars)[0]].squeeze())


def rh_from_t_td(t_k, td_k):
    t, td = t_k - 273.15, td_k - 273.15
    return np.clip(100 * np.exp(17.67 * td / (td + 243.5)) / np.exp(17.67 * t / (t + 243.5)), 1, 100)


def vpd_hpa(t_k, rh):
    t = t_k - 273.15
    return 6.112 * np.exp(17.67 * t / (t + 243.5)) * (1 - rh / 100)


# Which cycles to use: (every N hours, first cycle hour, hours after cycle before it is usually posted).
# HRRR 00/06/12/18Z and RAP 03/09/15/21Z run long (48-51 h); NBM 01/07/13/19Z are the full runs;
# ECMWF open data 06/18Z stop at day 4, so IFS/AIFS use 00/12Z.
CYCLES = {"HRRR": (6, 0, 2), "RAP": (6, 3, 2), "NBM": (6, 1, 2), "NAMNEST": (6, 0, 4), "RRFS": (6, 0, 4),
          "GFS": (6, 0, 5), "GEFS": (6, 0, 6), "IFS": (12, 0, 6), "AIFS": (12, 0, 6)}


def latest_init(name, now):
    step, off, lag = CYCLES[name]
    t = (now - timedelta(hours=lag)).replace(minute=0, second=0, microsecond=0)
    return t - timedelta(hours=(t.hour - off) % step)


def pick_init(name, kw, hours, days, first):
    """Use the newest cycle whose last needed forecast hour is posted; else step back up to 2 cycles."""
    from herbie import Herbie
    step = CYCLES[name][0]
    for k in range(3):
        init = first - timedelta(hours=step * k)
        need = [f for f in hours if (init + timedelta(hours=f)).astimezone(MST).date() <= days[-1]]
        try:
            if need and Herbie(init.replace(tzinfo=None), fxx=need[-1], verbose=False, **kw).grib is not None:
                return init
        except Exception:
            pass
        print(f"[{name}] {init:%d %HZ} not complete yet, trying the cycle before", flush=True)
    return first


def cmd_model(a):
    from herbie import Herbie
    import shapely

    name = a.model.upper()
    kw, hours, search, _ = MODELS[name]
    now = datetime.now(timezone.utc)
    days = [now.astimezone(MST).date() + timedelta(days=i) for i in range(7)]
    init = (datetime.strptime(a.init, "%Y-%m-%dT%H").replace(tzinfo=timezone.utc) if a.init
            else pick_init(name, kw, hours, days, latest_init(name, now)))
    psas = load_psas(a.psa)
    print(f"[{name}] init {init:%Y-%m-%d %HZ}", flush=True)

    masks, recs, cum, last_total = None, [], {}, None
    save_grids = name in GRID_MODELS
    grid = {"box": None, "rh": {}, "gust": {}, "qpf": {}}
    for f in hours:
        valid = init + timedelta(hours=f)
        local = valid.astimezone(MST)
        if local.date() > days[-1]:
            break
        try:
            H = Herbie(init.replace(tzinfo=None), fxx=f, verbose=False, **kw)
            if H.grib is None:
                continue
        except Exception:
            continue
        fld = {k: fetch_field(H, p) for k, p in search.items()}
        base = next((v for v in fld.values() if v is not None), None)
        if base is None:
            continue
        if masks is None:
            lat, lon = base.latitude.values, base.longitude.values
            if lat.ndim == 1:
                lon, lat = np.meshgrid(lon, lat)
            lon = np.where(lon > 180, lon - 360, lon)
            masks = {k: shapely.contains_xy(g, lon, lat) for k, g in psas.items()}
            if save_grids:
                lo0, lo1, la0, la1 = SW_BBOX
                inb = (lon >= lo0) & (lon <= lo1) & (lat >= la0) & (lat <= la1)
                rr, cc = np.where(inb)
                step_px = max(1, int(round(max(np.ptp(rr), np.ptp(cc)) / 700)))     # keep maps under ~700 px across
                sl = (slice(rr.min(), rr.max() + 1, step_px), slice(cc.min(), cc.max() + 1, step_px))
                grid.update(box=sl, lat=lat[sl].astype("float32"), lon=lon[sl].astype("float32"))
        g = {k: (None if v is None else np.asarray(v.values, dtype=float)) for k, v in fld.items()}

        wspd = np.hypot(g["u10"], g["v10"]) if g.get("u10") is not None and g.get("v10") is not None else g.get("wspd")
        rh = g.get("rh2")
        if rh is None and g.get("t2") is not None and g.get("d2") is not None:
            rh = rh_from_t_td(g["t2"], g["d2"])
        lapse = None
        if g.get("t700") is not None and g.get("t500") is not None:
            dz = (g["z500"] - g["z700"]) / 1000.0 if g.get("z700") is not None and g.get("z500") is not None else 2.7
            lapse = (g["t700"] - g["t500"]) / dz
        hdwi = vpd_hpa(g["t2"], rh) * wspd if g.get("t2") is not None and rh is not None and wspd is not None else None

        step_precip = None
        if fld.get("apcp") is not None:
            val = g["apcp"] * (1000.0 if name in ("IFS", "AIFS") else 1.0)
            m = re.match(r"(\d+)-(\d+)", str(fld["apcp"].attrs.get("GRIB_stepRange", f"0-{f}")))
            s0, s1 = (int(m.group(1)), int(m.group(2))) if m else (0, f)
            total = val if s0 == 0 else (cum[s0] + val if s0 in cum else None)
            if total is not None:
                cum[s1] = total
                step_precip = (total - (last_total if last_total is not None else 0)) * MM_TO_IN
                last_total = total

        if save_grids:
            sl, dk = grid["box"], local.date().isoformat()
            if local.hour in AFTERNOON:
                if rh is not None:
                    grid["rh"][dk] = np.fmin(grid["rh"].get(dk, np.inf), rh[sl])
                gg = g.get("gust") if g.get("gust") is not None else wspd
                if gg is not None:
                    grid["gust"][dk] = np.fmax(grid["gust"].get(dk, -np.inf), gg[sl] * MS_TO_MPH)
            if step_precip is not None:
                grid["qpf"][dk] = grid["qpf"].get(dk, 0) + np.clip(step_precip[sl], 0, None)

        for psa, mk in masks.items():
            if not mk.any():
                continue
            q = lambda arr, p: None if arr is None else float(np.nanpercentile(arr[mk], p))
            thunder = None
            if g.get("cape") is not None:
                wet = (step_precip[mk] > 0.005) if step_precip is not None else True
                thunder = float(np.mean((g["cape"][mk] >= 250) & wet) * 100)
            recs.append(dict(
                psa=psa, date=local.date(), hour=local.hour,
                wind=None if wspd is None else q(wspd, 90) * MS_TO_MPH,
                gust=None if g.get("gust") is None else q(g["gust"], 90) * MS_TO_MPH,
                minrh=q(rh, 10),
                tmax=None if g.get("t2") is None else (q(g["t2"], 90) - 273.15) * 9 / 5 + 32,
                qpf=None if step_precip is None else float(np.nanmedian(step_precip[mk])),
                thunder=thunder, lapse=None if lapse is None else float(np.nanmean(lapse[mk])),
                hdwi=q(hdwi, 90)))
    if not recs:
        sys.exit(f"[{name}] no data retrieved")
    df = pd.DataFrame(recs)
    aft = df[df.hour.isin(AFTERNOON)]
    out = pd.concat([
        aft.groupby(["psa", "date"]).agg(wind=("wind", "max"), gust=("gust", "max"), minrh=("minrh", "min"),
                                         thunder=("thunder", "max"), lapse=("lapse", "max"), hdwi=("hdwi", "max")),
        df.groupby(["psa", "date"]).agg(tmax=("tmax", "max"), qpf=("qpf", "sum")),
    ], axis=1).reset_index()
    out["model"], out["init"] = name, init.strftime("%Y-%m-%dT%HZ")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    out.to_csv(a.out, index=False)
    if save_grids and grid["box"] is not None:
        dkeys = [d.isoformat() for d in days]
        stack = lambda f: np.stack([grid[f].get(k, np.full(grid["lat"].shape, np.nan)) for k in dkeys]).astype("float16")
        np.savez_compressed(a.out.replace(".csv", "_grids.npz"), lat=grid["lat"], lon=grid["lon"], days=np.array(dkeys),
                            rh=stack("rh"), gust=stack("gust"), qpf=stack("qpf"), init=init.strftime("%Y-%m-%dT%HZ"))
        print(f"[{name}] saved daily grids for maps", flush=True)
    print(f"[{name}] wrote {a.out}: {len(out)} PSA-days", flush=True)


# ------------------------------------------------------------------ classify
NAMES = {0: "nodata", 1: "green", 2: "yellow", 3: "brown"}


def erc_pct(cfg, psa, erc):
    bp = cfg["psas"].get(psa, {}).get("erc")
    if erc is None or not bp:
        return None
    pts = [(p, v) for p, v in zip(cfg["erc_percentiles"], bp) if v is not None]
    if erc < pts[0][1]:
        return max(0, round(pts[0][0] * erc / pts[0][1]))
    for (p0, v0), (p1, v1) in zip(pts, pts[1:]):
        if erc < v1:
            return round(p0 + (p1 - p0) * (erc - v0) / (v1 - v0))
    return min(100, round(97 + (erc - pts[-1][1]) * 0.5))


def classify(cfg, psa, rec, d):
    gv = lambda f, i: (rec.get(f) or [None] * 7)[i] if 0 <= i < 7 else None
    why = []
    erc = gv("erc", d)
    if erc is None:
        return {"category": "nodata", "trigger": None, "why": ["No ERC for this PSA-day."]}
    p = erc_pct(cfg, psa, erc)
    lvl = 1 if p < cfg["erc_low_pct"] else 2 if p < cfg["erc_mod_pct"] else 3
    sfx = "th" if 10 <= p % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(p % 10, "th")
    src = gv("source", d)
    why.append(f"ERC {erc:g}{' (' + src + ')' if src else ''} is about the {p}{sfx} percentile -> {NAMES[lvl]}.")
    f100, f1000, kb = gv("fm100", d), gv("fm1000", d), gv("kbdi", d)
    if f100 is not None or f1000 is not None:
        why.append(f"Fuel moisture: 100-hr {f100}%, 1000-hr {f1000}%; KBDI {kb}.")
    bi, hd, fi = gv("bi_pct", d), gv("hdwi", d), gv("fires", d)
    if lvl < 3 and ((bi is not None and bi >= cfg["bi_raise_pct"]) or (hd is not None and hd >= cfg["hdwi_raise"])
                    or (fi is not None and fi >= cfg["fires_raise"])):
        lvl += 1
        why.append(f"Raised one level (BI pct {bi}, HDWI {hd}, fires {fi}).")
    q, qp = gv("qpf", d), gv("qpf", d - 1)
    if (q or 0) >= cfg["wetting_rain_in"] or (qp or 0) >= cfg["wetting_rain_in"]:
        lvl = 1
        why.append(f'Wetting rain ({q or 0}" today, {qp or 0}" yesterday) -> green.')
    elif q is not None and q >= cfg["light_rain_in"] and lvl > 1:
        lvl -= 1
        why.append(f'Light rain {q}" -> down one level.')
    trig = []
    if lvl >= 2:
        def wind_ok(i):
            w, g, rh = gv("wind", i), gv("gust", i), gv("minrh", i)
            return rh is not None and rh <= cfg["wind_rh_max"] and (
                (w or 0) >= cfg["wind_sustained_mph"] or (g or 0) >= cfg["wind_gust_mph"])
        rh, tx, th, la = gv("minrh", d), gv("tmax", d), gv("thunder", d), gv("lapse", d)
        pc = cfg["psas"].get(psa, {})
        hot = pc.get("hot_day_f") or cfg["hot_day_default_f"]
        peak = pc.get("peak_tmax_f") or cfg["much_above_peak_default_f"]
        if th is not None and th >= cfg["lightning_coverage_pct"] and rh is not None and rh < cfg["lightning_rh_below"]:
            hot_n = sum(1 for k in range(d - 3, d) if k >= 0 and (gv("tmax", k) or -999) >= hot)
            n = d + 1 if d < 6 else d
            after = (gv("minrh", n) is not None and gv("minrh", n) < cfg["lightning_rh_below"]) or wind_ok(n)
            if hot_n >= min(cfg["lightning_hot_days"], d) and after:
                trig.append("L")
                why.append(f"L: thunder {th:.0f}% with min RH {rh:.0f}%, {hot_n} hot day(s) before, dry or windy after.")
            else:
                why.append(f"Thunder {th:.0f}% but L not met (hot days {hot_n}).")
        if wind_ok(d):
            trig.append("W")
            why.append(f"W: wind {gv('wind', d)} / gust {gv('gust', d)} mph with min RH {rh}%.")
        if tx is not None and tx >= peak and rh is not None and rh <= cfg["burn_rh_max"] and la is not None and la >= cfg["burn_lapse_c_km"]:
            trig.append("B")
            why.append(f"B: Tmax {tx}F >= {peak}F, min RH {rh}%, lapse {la} C/km.")
    else:
        why.append("Fuels too moist for High Risk triggers.")
    return {"category": NAMES[lvl], "trigger": trig[0] if trig else None, "all_triggers": trig, "why": why, "erc_pct": p}


TRIG_NAME = {"W": "Wind", "L": "Lightning", "B": "Burn environment"}
CAT_NAME = {"green": "little or no risk", "yellow": "low risk", "brown": "moderate risk", "nodata": "no data"}


def _plist(ps):
    ps = list(ps)
    return ps[0] if len(ps) == 1 else ", ".join(ps[:-1]) + " and " + ps[-1]


def _day(d):
    return f"{datetime.fromisoformat(d):%a %b %-d}"


# ---- plain-language areas instead of PSA codes
STATE_BOX = {"AZ": (-114.82, -109.05, 31.33, 37.0), "NM": (-109.05, -103.0, 31.33, 37.0)}
_REGION = {}


def load_regions(psa_path):
    """Give each PSA a description such as 'northern AZ' or 'east-central NM' from its boundary's center point."""
    _REGION.clear()
    if not psa_path or not os.path.exists(psa_path):
        return
    for f in json.load(open(psa_path))["features"]:
        pr = f["properties"]
        x, y = pr.get("lx"), pr.get("ly")
        if x is None:
            continue
        if x > -106.7 and y < 32.0:
            _REGION[pr["psa"]] = ("TX", "far west")
            continue
        st = "AZ" if x < -109.05 else "NM"
        x0, x1, y0, y1 = STATE_BOX[st]
        fx, fy = (x - x0) / (x1 - x0), (y - y0) / (y1 - y0)
        ns = "north" if fy > 0.62 else "south" if fy < 0.33 else "central"
        ew = "west" if fx < 0.33 else "east" if fx > 0.67 else "central"
        word = {("north", "west"): "northwest", ("north", "central"): "northern", ("north", "east"): "northeast",
                ("central", "west"): "west-central", ("central", "central"): "central", ("central", "east"): "east-central",
                ("south", "west"): "southwest", ("south", "central"): "southern", ("south", "east"): "southeast"}[(ns, ew)]
        _REGION[pr["psa"]] = (st, word)


def _area(psas):
    """'northern and central AZ', 'much of NM', 'across the Southwest' ..."""
    psas = [p for p in psas]
    if not psas:
        return "the area"
    if not _REGION:
        return "parts of the area"
    known = [p for p in SW_PSAS if p in _REGION]
    if len(psas) >= 0.8 * len(known):
        return "across the Southwest" if len(psas) == len(known) else "across most of the Southwest"
    out = []
    for st in ("AZ", "NM", "TX"):
        allst = [p for p in known if _REGION[p][0] == st]
        mine = [p for p in psas if _REGION.get(p, ("", ""))[0] == st]
        if not mine:
            continue
        if st == "TX":
            out.append("far west Texas")
        elif len(mine) == len(allst):
            out.append(f"all of {st}")
        elif len(mine) >= 0.7 * len(allst):
            out.append(f"much of {st}")
        else:
            words = []
            for p in mine:
                w = _REGION[p][1]
                if w not in words:
                    words.append(w)
            # fold e.g. northwest + northern + northeast into 'northern'
            for band, group in (("northern", {"northwest", "northern", "northeast"}),
                                ("southern", {"southwest", "southern", "southeast"}),
                                ("central", {"west-central", "central", "east-central"})):
                if len(group & set(words)) >= 2:
                    words = [w for w in words if w not in group]
                    words.insert(0, band)
            order = ["northwest", "northern", "northeast", "west-central", "central", "east-central",
                     "southwest", "southern", "southeast"]
            words.sort(key=order.index)
            out.append(f"{_plist(words)} {st}")
    return _plist(out)


def _in(psas):
    a = _area(psas)
    return a if a.startswith("across") else "in " + a


def _wd(days, i):
    return "Today" if i == 0 else f"{datetime.fromisoformat(days[i]):%A}"


def _days_text(days, idx):
    """'Wednesday', 'Wednesday and Thursday', or 'Wednesday through Friday'."""
    idx = sorted(set(idx))
    if len(idx) == 1:
        return _wd(days, idx[0])
    if idx == list(range(idx[0], idx[-1] + 1)):
        joiner = " and " if len(idx) == 2 else " through "
        return _wd(days, idx[0]) + joiner + _wd(days, idx[-1])
    return _plist([_wd(days, i) for i in idx])


def _paragraphs(sentences, max_words=110):
    """Join sentences into paragraphs, starting a new one when a paragraph runs long.
    A sentence tuple (text, True) marks a good place to break (start of a new day/period)."""
    paras, cur, words = [], [], 0
    for text, breakable in sentences:
        n = len(text.split())
        if cur and breakable and words + n > max_words:
            paras.append(" ".join(cur))
            cur, words = [], 0
        cur.append(text)
        words += n
    if cur:
        paras.append(" ".join(cur))
    return paras


def auto_discussion(days, psas, out, fires=None, cfg=None):
    """Weather discussion in day order. Quiet, similar days are grouped; High Risk trigger days are called out.
    Long paragraphs are split at the start of a new day."""
    cfg = cfg or {"erc_low_pct": 60, "erc_mod_pct": 90}
    val = lambda p, f, i: (psas[p].get(f) or [None] * 7)[i]
    S = []                                   # (sentence, ok to start a paragraph here)

    # ---------------- day by day
    if not any(val(p, "gust", 0) is not None or val(p, "minrh", 0) is not None for p in SW_PSAS):
        return ["Model weather data was not available for this run."]

    def info(i):
        g = {p: val(p, "gust", i) for p in SW_PSAS if val(p, "gust", i) is not None}
        rh = {p: val(p, "minrh", i) for p in SW_PSAS if val(p, "minrh", i) is not None}
        return {
            "g": g, "rh": rh,
            "windy": sorted([p for p, v in g.items() if v >= cfg.get("wind_gust_mph", 35)], key=SW_PSAS.index),
            "dry": sorted([p for p, v in rh.items() if v <= cfg.get("wind_rh_max", 15)], key=SW_PSAS.index),
            "wet": sorted([p for p in SW_PSAS if (val(p, "qpf", i) or 0) >= 0.10], key=SW_PSAS.index),
            "storms": sorted([p for p in SW_PSAS if (val(p, "thunder", i) or 0) >= cfg.get("lightning_coverage_pct", 25)],
                             key=SW_PSAS.index),
            "trig": {t: sorted([p for p in SW_PSAS if out[p][i]["trigger"] == t], key=SW_PSAS.index) for t in "WLB"},
            "brown": sorted([p for p in SW_PSAS if out[p][i]["category"] == "brown"], key=SW_PSAS.index),
        }

    D = [info(i) for i in range(7)]
    sig = lambda d: (bool(d["windy"]), bool(d["dry"]), tuple(d["wet"]), tuple(d["storms"]), any(d["trig"].values()))
    periods, cur = [], [0]
    for i in range(1, 7):
        if sig(D[i]) == sig(D[cur[-1]]) and not any(D[i]["trig"].values()):
            cur.append(i)
        else:
            periods.append(cur)
            cur = [i]
    periods.append(cur)

    for k, ids in enumerate(periods):
        when = _days_text(days, ids)
        g = {p: max(D[i]["g"].get(p, 0) for i in ids) for p in SW_PSAS}
        rh = {p: min(D[i]["rh"].get(p, 100) for i in ids) for p in SW_PSAS}
        windy = sorted({p for i in ids for p in D[i]["windy"]}, key=SW_PSAS.index)
        dry = sorted({p for i in ids for p in D[i]["dry"]}, key=SW_PSAS.index)
        wet = sorted({p for i in ids for p in D[i]["wet"]}, key=SW_PSAS.index)
        storms = sorted({p for i in ids for p in D[i]["storms"]}, key=SW_PSAS.index)
        bits = []
        if windy:
            pk = max(windy, key=lambda p: g[p])
            bits.append(f"gusts of 35 mph or more {_in(windy)}, peaking near {g[pk]:.0f} mph")
        elif g:
            cap = int(np.ceil((max(g.values()) + 0.1) / 5.0) * 5)
            bits.append(f"gusts under {cap} mph")
        if dry:
            bits.append(f"minimum humidity of 15% or lower {_in(dry)}")
        elif rh:
            bits.append(f"minimum humidity of {min(rh.values()):.0f}-{max(rh.values()):.0f}%")
        if storms:
            bits.append(f"scattered thunderstorms {_in(storms)}")
        if wet:
            bits.append(f"wetting rain {_in(wet)}")
        verb = "brings" if len(ids) == 1 else "bring"
        if not windy and not dry and not storms and not wet:
            lead = f"{when} {'looks' if len(ids) == 1 else 'look'} quiet, with " + " and ".join(bits)
        else:
            lead = f"{when} {verb} " + (", ".join(bits[:-1]) + (", and " if len(bits) > 2 or "," in bits[0] else " and ")
                                        + bits[-1] if len(bits) > 1 else bits[0])
        S.append((lead + ".", True))

        for t, name in (("L", "Lightning"), ("W", "Wind"), ("B", "Burn environment")):
            ps = D[ids[0]]["trig"][t]
            if not ps:
                continue
            i = ids[0]
            rm = min((val(p, "minrh", i) or 100) for p in ps)
            if t == "W":
                det = f"gusts to {max((val(p, 'gust', i) or 0) for p in ps):.0f} mph and humidity near {rm:.0f}%"
            elif t == "L":
                det = f"thunderstorms over about {max((val(p, 'thunder', i) or 0) for p in ps):.0f}% of the area and humidity near {rm:.0f}%"
            else:
                det = f"highs near {max((val(p, 'tmax', i) or 0) for p in ps):.0f}F, humidity near {rm:.0f}% and an unstable atmosphere"
            S.append((f"A {name} High Risk trigger is forecast {_in(ps)}, with {det}.", False))

    tx = [(val(p, "tmax", i), i) for p in SW_PSAS for i in range(7) if val(p, "tmax", i) is not None]
    if tx:
        t0 = np.mean([v for v, i in tx if i == 0])
        t6 = np.mean([v for v, i in tx if i == 6]) if any(i == 6 for _, i in tx) else t0
        if t6 - t0 >= 5:
            S.append(("Temperatures warm through the period.", False))
        elif t0 - t6 >= 5:
            S.append(("Temperatures cool through the period.", False))
    return _paragraphs(S)


# ------------------------------------------------------------------ FEMS
# Public, no-login endpoints (FEMS TechNote 2025-09 and Climatology API guide).
FEMS_GQL = "https://fems.fs2c.usda.gov/api/climatology/graphql"
FEMS_DL = "https://fems.fs2c.usda.gov/api/ext-climatology/"
FEMS_NFDR_PATHS = ["download-nfdr-daily-summary/", "download-nfdr-dailysummary/"]
FEMS_STATES = ["AZ", "NM", "TX"]          # TX covers the far west Texas edge of SW14S
# Output names, the header names FEMS uses (normalized), and the column position as a fallback.
FEMS_COLS = {
    "fm1": (["1hrfm"], 4), "fm10": (["10hrfm"], 6), "fm100": (["100hrfm"], 8), "fm1000": (["1000hrfm"], 10),
    "kbdi": (["kbdi"], 12), "gsi": (["gsi"], 13), "woody": (["woodylfi", "woodyfm", "woodylfm", "woody"], 14),
    "herb": (["herbaceouslfi", "herbaceousfm", "herblfi", "herbfm", "herbaceous", "herb"], 15),
    "ic": (["ic", "ignitioncomponent"], 16), "erc": (["erc", "energyreleasecomponent"], 18),
    "sc": (["sc", "spreadcomponent"], 20), "bi": (["bi", "burningindex"], 22),
}
FEMS_FIELDS = list(FEMS_COLS)


def _http(url, data=None, headers=None, timeout=120, tries=3):
    import time
    for k in range(tries):
        try:
            req = urllib.request.Request(url, data=data, headers=headers or {"User-Agent": "sw-psa-map"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read().decode("utf-8", "replace")
        except Exception as e:
            if k == tries - 1:
                raise
            print(f"  retry {url[:90]}… ({e})", flush=True)
            time.sleep(5 * (k + 1))


def fems_discover_stations(psa_path):
    """Find NFDRS stations in AZ/NM/TX through the FEMS GraphQL API and assign them to PSAs."""
    import shapely
    q = """query($s:String,$p:Int,$n:Int){stationMetaData(stateId:$s,page:$p,per_page:$n){
           _metadata{page_count total_count} data{station_id station_name latitude longitude station_status network_name last_observation}}}"""
    rows = []
    for st in FEMS_STATES:
        page = 0
        while True:
            body = json.dumps({"query": q, "variables": {"s": st, "p": page, "n": 500}}).encode()
            res = json.loads(_http(FEMS_GQL, body, {"Content-Type": "application/json", "User-Agent": "sw-psa-map"}))
            if res.get("errors"):
                raise RuntimeError(res["errors"][0].get("message"))
            blk = res["data"]["stationMetaData"]
            rows += blk["data"] or []
            page += 1
            if page >= (blk["_metadata"].get("page_count") or 1):
                break
    psas = load_psas(psa_path)
    out = {p: [] for p in SW_PSAS}
    for r in rows:
        try:
            x, y = float(r["longitude"]), float(r["latitude"])
        except (TypeError, ValueError):
            continue
        if str(r.get("station_status") or "A").upper()[:1] not in ("A", ""):   # skip inactive
            continue
        for p, g in psas.items():
            if p in out and shapely.contains_xy(g, x, y):
                out[p].append({"id": str(r["station_id"]), "name": r.get("station_name")})
                break
    return out


def fems_download(station_ids, start, end, dataset="all", fuel="Y"):
    """Daily NFDRS summary CSV for a list of stations -> DataFrame (one row per station-date-type)."""
    import io
    frames = []
    multi_year = (end - start).days > 366
    chunk = 1 if multi_year else (10 if (end - start).days > 14 else 40)
    for i in range(0, len(station_ids), chunk):
        ids = ",".join(station_ids[i:i + chunk])
        txt = None
        for path in FEMS_NFDR_PATHS:
            q = urllib.parse.urlencode({"dataset": dataset, "startDate": start.isoformat(), "endDate": end.isoformat(),
                                        "dataFormat": "csv", "stationIds": ids, "fuelModels": fuel})
            try:
                txt = _http(f"{FEMS_DL}{path}?{q}", timeout=300)
                if txt.lstrip().startswith('"') or "," in txt[:200]:
                    break
            except Exception as e:
                txt = None
                last = e
        if not txt or not txt.strip():
            print(f"  no FEMS data for {ids[:60]} ({'' if txt else last})", flush=True)
            continue
        raw = pd.read_csv(io.StringIO(txt))
        if raw.empty:
            continue
        norm = {re.sub(r"[^a-z0-9]", "", c.lower()): c for c in raw.columns}
        df = pd.DataFrame()
        df["station"] = raw[norm.get("stationid", raw.columns[-1])].astype(str)
        df["date"] = pd.to_datetime(raw[norm.get("observationtime", raw.columns[1])]).dt.date
        df["type"] = raw[norm.get("nfdrtype", raw.columns[2])].astype(str).str.upper().str[0]
        for out, (names, pos) in FEMS_COLS.items():
            col = next((norm[n] for n in names if n in norm), raw.columns[pos] if pos < len(raw.columns) else None)
            df[out] = pd.to_numeric(raw[col], errors="coerce") if col is not None else np.nan
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def nwcg_key_raws():
    """SWCC key RAWS per PSA from the NWCG Predictive Services map service (layer 2)."""
    lon0, lon1, lat0, lat1 = SW_BBOX
    q = urllib.parse.urlencode({"where": "1=1", "outFields": "*", "f": "json", "returnGeometry": "false",
                                "geometry": f"{lon0},{lat0},{lon1},{lat1}", "geometryType": "esriGeometryEnvelope",
                                "inSR": 4326, "spatialRel": "esriSpatialRelIntersects"})
    js = json.loads(_http(f"{NWCG_PSP}/2/query?{q}"))
    out = {}
    for f in js.get("features", []):
        at = f.get("attributes") or {}
        psa = _psa_code(at)
        sid = None
        for k, v in at.items():
            if any(t in k.lower() for t in ("wims", "station_id", "stationid", "nfdrs", "raws_id", "sta_id")):
                if re.fullmatch(r"\d{5,6}", str(v).strip().split(".")[0]):
                    sid = str(v).strip().split(".")[0]
                    break
        if psa and sid:
            out.setdefault(psa, []).append({"id": sid, "name": next((str(v) for k, v in at.items() if "name" in k.lower() and v), None)})
    return out


def load_station_map(a):
    """config/stations.json (your SIG lists) wins; then SWCC key RAWS from NWCG; then every station found in the PSA."""
    manual = json.load(open(a.stations)) if os.path.exists(a.stations) else {}
    manual = {k: v for k, v in manual.items() if not k.startswith("_") and v}
    auto = {}
    if len(manual) < len(SW_PSAS):
        if os.path.exists(a.discovered) and not getattr(a, "rediscover", False):
            auto = json.load(open(a.discovered))
        else:
            key = {}
            try:
                key = nwcg_key_raws()
                print(f"NWCG key RAWS: {sum(len(v) for v in key.values())} stations in {len(key)} PSAs")
            except Exception as e:
                print(f"NWCG key RAWS unavailable ({e})")
            found = {}
            if len(key) < len(SW_PSAS):
                try:
                    found = fems_discover_stations(a.psa)
                except Exception as e:
                    print(f"FEMS station search failed ({e})")
            auto = {p: key.get(p) or found.get(p, []) for p in SW_PSAS}
            json.dump(auto, open(a.discovered, "w"), indent=1)
            print(f"Station list: {sum(len(v) for v in auto.values())} stations -> {a.discovered}")
    smap = {}
    for p in SW_PSAS:
        if p in manual:
            smap[p] = [str(s) for s in manual[p]]
        else:
            smap[p] = [s["id"] for s in auto.get(p, [])]
    return smap


def psa_daily(df, smap):
    """Average stations into one value per PSA-day; observed beats forecast for the same day."""
    if df.empty:
        return pd.DataFrame()
    df = df.sort_values("type", key=lambda s: s.map({"O": 0, "F": 1}).fillna(2))
    df = df.drop_duplicates(["station", "date"], keep="first")
    rows = []
    for p, ids in smap.items():
        sub = df[df.station.isin(ids)]
        if sub.empty:
            continue
        g = sub.groupby("date")
        agg = g[FEMS_FIELDS].mean()
        agg["n_stations"] = g["erc"].count()
        agg["source"] = g["type"].agg(lambda t: "obs" if (t == "O").all() else "fcst" if (t == "F").all() else "mixed")
        agg["psa"] = p
        rows.append(agg.reset_index())
    return pd.concat(rows, ignore_index=True)


def bi_percentile(climo, psa, bi):
    c = (climo or {}).get(psa, {}).get("bi")
    if bi is None or pd.isna(bi) or not c:
        return None
    pts = sorted((float(k), v) for k, v in c.items() if v is not None)
    if bi < pts[0][1]:
        return round(pts[0][0] * bi / pts[0][1]) if pts[0][1] else 0
    for (p0, v0), (p1, v1) in zip(pts, pts[1:]):
        if bi < v1:
            return round(p0 + (p1 - p0) * (bi - v0) / (v1 - v0)) if v1 != v0 else round(p1)
    return 99


def cmd_fems(a):
    smap = load_station_map(a)
    ids = sorted({s for v in smap.values() for s in v})
    if not ids:
        sys.exit("No FEMS stations found. Fill in config/stations.json.")
    today = datetime.now(timezone.utc).astimezone(MST).date()
    start, end = today - timedelta(days=5), today + timedelta(days=7)
    print(f"FEMS: {len(ids)} stations, {start} to {end}", flush=True)
    df = fems_download(ids, start, end, "all", a.fuel_model)
    daily = psa_daily(df, smap)
    if daily.empty:
        sys.exit("FEMS returned no NFDRS rows.")
    daily = daily.sort_values(["psa", "date"])
    # SWCC's PSA ERC charts use a 2-day moving average; set erc_smoothing_days in thresholds.json
    cfg = json.load(open(a.config))
    n = int(cfg.get("erc_smoothing_days", 1))
    if n > 1:
        daily["erc_raw"] = daily["erc"]
        daily["erc"] = daily.groupby("psa")["erc"].transform(lambda s: s.rolling(n, min_periods=1).mean())
    climo = json.load(open(a.climo)) if os.path.exists(a.climo) else {}
    daily["bi_pct"] = [bi_percentile(climo, p, b) for p, b in zip(daily.psa, daily.bi)]
    daily = daily.round(2)
    daily.to_csv(a.out, index=False)
    print(f"Wrote {a.out}: {len(daily)} PSA-days")


def fems_tmax(station_ids, start, end):
    """PSA-average daily max temperature (F) from FEMS daily weather summaries."""
    import io
    series = []
    for sid in station_ids:
        q = urllib.parse.urlencode({"dataset": "observation", "startDate": start.isoformat(), "endDate": end.isoformat(),
                                    "dataFormat": "csv", "stationIds": sid})
        txt = None
        for path in ("download-wx-daily-summary/", "download-wx-dailysummary/"):
            try:
                txt = _http(f"{FEMS_DL}{path}?{q}", timeout=300)
                if txt and "," in txt[:200]:
                    break
            except Exception:
                txt = None
        if not txt:
            continue
        raw = pd.read_csv(io.StringIO(txt))
        norm = {re.sub(r"[^a-z0-9]", "", c.lower()): c for c in raw.columns}
        tcol = next((norm[k] for k in norm if "max" in k and ("temp" in k or "tmp" in k)), None)
        dcol = next((norm[k] for k in norm if k in ("observationtime", "date", "summarydate", "obsdate")), raw.columns[1])
        if tcol is None:
            continue
        s = pd.Series(pd.to_numeric(raw[tcol], errors="coerce").values, index=pd.to_datetime(raw[dcol]).dt.date)
        series.append(s[(s > -40) & (s < 130)])
    if not series:
        raise RuntimeError("no temperature data")
    return pd.concat(series, axis=1).mean(axis=1).dropna()


def cmd_fems_climo(a):
    """Build BI and ERC percentile breakpoints for each PSA from FEMS history (run once a year)."""
    smap = load_station_map(a)
    start, end = datetime(a.start_year, 1, 1).date(), datetime(a.end_year, 12, 31).date()
    months = set(range(a.start_month, a.end_month + 1))
    pcts = [50, 60, 70, 75, 80, 85, 90, 95, 97]
    out = {"_note": f"FEMS fuel model {a.fuel_model}, {a.start_year}-{a.end_year}, months {a.start_month}-{a.end_month}, "
                    "PSA mean of station daily values.", "percentiles": pcts}
    for p, ids in smap.items():
        if not ids:
            continue
        print(f"{p}: {len(ids)} stations", flush=True)
        df = fems_download(ids, start, end, "observation", a.fuel_model)
        if df.empty:
            continue
        daily = psa_daily(df, {p: ids})
        daily = daily[pd.to_datetime(daily.date).dt.month.isin(months)]
        out[p] = {k: {str(q): round(float(np.nanpercentile(daily[k].dropna(), q)), 1) for q in pcts}
                  for k in ("bi", "erc") if daily[k].notna().sum() > 30}
        out[p]["days"] = int(len(daily))
        try:
            t = fems_tmax(ids, start, end)
            t = t[pd.to_datetime(t.index).month.isin([5, 6, 7])]          # peak fire season May-Jul
            if len(t) > 60:
                out[p]["tmax"] = {"peak_normal": round(float(t.mean()), 1),
                                  "hot_day": round(float(np.percentile(t, 75)), 1),
                                  "much_above": round(float(np.percentile(t, 95)), 1)}
        except Exception as e:
            print(f"  {p} temperatures skipped: {e}")
    out["created"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"Wrote {a.out}")


# ------------------------------------------------------------------ fires on the ground (NIFC WFIGS)
WFIGS = ["https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/WFIGS_Incident_Locations_Current/FeatureServer/0/query",
         "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/Current_WildlandFire_Locations/FeatureServer/0/query"]


def _first(props, *names):
    low = {k.lower(): v for k, v in props.items()}
    for n in names:
        if low.get(n.lower()) not in (None, ""):
            return low[n.lower()]
    return None


def cmd_fires(a):
    """Active wildfires in each PSA from the NIFC WFIGS current incident service -> docs/fires.json."""
    import shapely
    cfg = json.load(open(a.config))
    min_ac = float(cfg.get("fires_min_acres", 100))
    lon0, lon1, lat0, lat1 = SW_BBOX
    q = urllib.parse.urlencode({"where": "1=1", "outFields": "*", "outSR": 4326, "f": "geojson",
                                "geometry": f"{lon0},{lat0},{lon1},{lat1}", "geometryType": "esriGeometryEnvelope",
                                "inSR": 4326, "spatialRel": "esriSpatialRelIntersects", "resultRecordCount": 2000})
    gj = None
    for url in WFIGS:
        try:
            gj = json.loads(_http(f"{url}?{q}"))
            if "features" in gj:
                break
        except Exception as e:
            print(f"  {url.split('/services/')[1].split('/')[0]}: {e}")
    if not gj or "features" not in gj:
        sys.exit("Could not reach the WFIGS incident service.")
    psas = load_psas(a.psa)
    fires = []
    for f in gj["features"]:
        pr, geo = f.get("properties") or {}, f.get("geometry")
        if not geo:
            continue
        cat = str(_first(pr, "IncidentTypeCategory") or "WF").upper()
        if cat not in ("WF", "WILDFIRE"):
            continue
        if _first(pr, "FireOutDateTime", "FireOutDate"):
            continue
        acres = _first(pr, "IncidentSize", "DailyAcres", "CalculatedAcres", "DiscoveryAcres")
        acres = float(acres) if acres is not None else 0.0
        cont = _first(pr, "PercentContained")
        cont = float(cont) if cont is not None else None
        if acres < min_ac or (cont is not None and cont >= 100):
            continue
        x, y = geo["coordinates"][:2]
        psa = next((p for p, g in psas.items() if shapely.contains_xy(g, x, y)), None)
        if not psa:
            continue
        fires.append({"name": str(_first(pr, "IncidentName") or "Unnamed").title(), "psa": psa, "acres": round(acres),
                      "contained": None if cont is None else round(cont), "lat": round(y, 4), "lon": round(x, 4)})
    counts = {p: sum(1 for f in fires if f["psa"] == p) for p in SW_PSAS}
    json.dump({"created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"), "min_acres": min_ac,
               "counts": counts, "fires": fires}, open(a.out, "w"), indent=1)
    print(f"Wrote {a.out}: {len(fires)} active fires >= {min_ac:g} ac")


# ------------------------------------------------------------------ discussion maps
MAP_FIELDS = {
    "rh": ("Minimum RH", "%", [0, 5, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 100],
           ["#5c0a0a", "#9e1b1b", "#d7301f", "#ef6548", "#fc8d59", "#fdbb84", "#fee8a0",
            "#d9f0a3", "#addd8e", "#78c679", "#41ab5d", "#2b8cbe"]),
    "gust": ("Peak wind gust", "mph", [0, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 80],
             ["#f7fbff", "#d6e9f5", "#a8d8b9", "#e6f598", "#fee08b", "#fdae61", "#f46d43",
              "#d73027", "#a50026", "#7a0177", "#49006a"]),
    "qpf": ("Precipitation", "in", [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0],
            ["#d9f0d3", "#a6dba0", "#5aae61", "#1b7837", "#a6cee3", "#1f78b4", "#6a51a3",
             "#cb181d", "#fd8d3c", "#fed976"]),
}
OUTLOOK_COLORS = {"nodata": "#9aa0a3", "green": "#3f9a3f", "yellow": "#e8cf2c", "brown": "#8a5a2b",
                  "W": "#f08c1e", "B": "#f08c1e", "L": "#d22b25"}


def _draw_psas(ax, gj, fill=None, letters=None):
    from matplotlib.patches import Polygon as MPoly
    for f in gj["features"]:
        geom, pr = f["geometry"], f["properties"]
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        for poly in polys:
            ring = np.array(poly[0])
            if fill:
                ax.add_patch(MPoly(ring, closed=True, facecolor=fill.get(pr["psa"], "#9aa0a3"),
                                   edgecolor="none", alpha=0.92, zorder=1))
            for r in poly:
                r = np.array(r)
                ax.plot(r[:, 0], r[:, 1], color="#111111", lw=1.0, zorder=3)
        lab = pr["psa"] + (f"  {letters[pr['psa']]}" if letters and letters.get(pr["psa"]) else "")
        ax.text(pr["lx"], pr["ly"], lab, ha="center", va="center", fontsize=8.5, fontweight="bold", zorder=4,
                color="#111111", bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.9))


def _frame(gj):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = [c[0] for f in gj["features"] for c in _ring_pts(f["geometry"])]
    ys = [c[1] for f in gj["features"] for c in _ring_pts(f["geometry"])]
    fig, ax = plt.subplots(figsize=(9, 6.2), dpi=100)
    ax.set_xlim(min(xs) - 0.3, max(xs) + 0.3)
    ax.set_ylim(min(ys) - 0.3, max(ys) + 0.3)
    ax.set_aspect(1 / np.cos(np.radians(34.2)))
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.11)
    return fig, ax, plt


def _ring_pts(geom):
    polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
    return [pt for poly in polys for pt in poly[0]]


def _titles(fig, left, right):
    fig.text(0.01, 0.965, left, fontsize=13, fontweight="bold", va="center", color="#111111")
    fig.text(0.99, 0.965, right, fontsize=9, ha="right", va="center", color="#444444")


def cmd_maps(a):
    """Draw gridded maps (min RH, gusts, QPF) and static outlook maps with PSA outlines; add them to outlook.json."""
    from matplotlib.colors import BoundaryNorm, ListedColormap
    gj = json.load(open(a.psa))
    O = json.load(open(a.outlook))
    days = O["days"]
    os.makedirs(a.out_dir, exist_ok=True)
    rel = os.path.relpath(a.out_dir, os.path.dirname(a.outlook))
    manifest = {"fields": {}, "source": None}
    dlab = lambda d: f"{datetime.fromisoformat(d):%a %b %-d}"

    # static outlook maps
    paths = []
    for i, d in enumerate(days):
        fill = {p: OUTLOOK_COLORS[(c[i]["trigger"] or c[i]["category"])] for p, c in O["outlook"].items()}
        letters = {p: c[i]["trigger"] for p, c in O["outlook"].items() if c[i]["trigger"]}
        fig, ax, plt = _frame(gj)
        _draw_psas(ax, gj, fill=fill, letters=letters)
        _titles(fig, f"7-Day Significant Fire Potential  |  Day {i + 1}  {dlab(d)}", f"Issued {O['run']['created']}")
        from matplotlib.patches import Patch
        hand = [Patch(color=OUTLOOK_COLORS[k], label=l) for k, l in
                [("nodata", "No data"), ("green", "Little/no risk"), ("yellow", "Low"), ("brown", "Moderate"),
                 ("W", "W / B trigger"), ("L", "L trigger")]]
        fig.legend(handles=hand, loc="lower center", ncol=6, frameon=False, fontsize=9)
        fn = f"outlook_d{i + 1}.png"
        fig.savefig(os.path.join(a.out_dir, fn)); plt.close(fig)
        paths.append(f"{rel}/{fn}")
    manifest["fields"]["outlook"] = {"label": "PSA outlook", "days": paths}

    # gridded fields
    gfile = next((f for f in [os.path.join(a.partials, "NBM_grids.npz"), os.path.join(a.partials, "GFS_grids.npz")]
                  if os.path.exists(f)), None)
    if gfile:
        G = np.load(gfile, allow_pickle=False)
        model = os.path.basename(gfile).split("_")[0]
        manifest["source"] = f"{model} {str(G['init'])}"
        lat, lon = G["lat"].astype(float), G["lon"].astype(float)
        lon = np.where(lon > 180, lon - 360, lon)
        gdays = [str(x) for x in G["days"]]
        for f, (label, unit, bins, colors) in MAP_FIELDS.items():
            cmap = ListedColormap(colors)
            if f == "qpf":
                cmap.set_under((1, 1, 1, 0))
            norm = BoundaryNorm(bins, cmap.N, clip=f != "qpf")
            data = G[f].astype(float)
            paths = []
            series = [(i, data[gdays.index(d)] if d in gdays else None) for i, d in enumerate(days)]
            if f == "qpf":
                tot = np.nansum(np.stack([x for _, x in series if x is not None]), axis=0)
                series.append(("total", tot))
            for i, arr in series:
                if arr is None or np.all(np.isnan(arr)):
                    paths.append(None)
                    continue
                fig, ax, plt = _frame(gj)
                m = ax.pcolormesh(lon, lat, np.ma.masked_invalid(arr), cmap=cmap, norm=norm, shading="auto", zorder=0)
                _draw_psas(ax, gj)
                when = f"7-day total {dlab(days[0])} - {dlab(days[-1])}" if i == "total" else f"Day {i + 1}  {dlab(days[i])}"
                _titles(fig, f"{model} {label} ({unit})  |  {when}", f"Init {str(G['init'])}")
                cax = fig.add_axes([0.15, 0.055, 0.7, 0.025])
                cb = fig.colorbar(m, cax=cax, orientation="horizontal", ticks=bins,
                                  extend="max" if f == "qpf" else "neither")
                cb.ax.tick_params(labelsize=8)
                if f == "qpf":
                    cb.ax.set_xticklabels([f"{b:g}" for b in bins])
                fn = f"{f}_{'total' if i == 'total' else 'd' + str(i + 1)}.png"
                fig.savefig(os.path.join(a.out_dir, fn)); plt.close(fig)
                paths.append(f"{rel}/{fn}")
            entry = {"label": label, "unit": unit, "days": paths[:7]}
            if f == "qpf":
                entry["total"] = paths[7]
            manifest["fields"][f] = entry

    # featured map: the thing the discussion is about
    feat = ("outlook", 0, "PSA outlook for today")
    trig = [(i, p, c[i]["trigger"]) for p, c in O["outlook"].items() for i in range(7) if c[i]["trigger"]]
    brown = [i for p, c in O["outlook"].items() for i in range(7) if c[i]["category"] == "brown"]
    if trig and gfile:
        i, p, t = sorted(trig)[0]
        feat = ("gust", i, "Peak gusts on the first wind trigger day") if t == "W" else ("rh", i, "Minimum RH on the first trigger day")
    elif trig or brown:
        feat = ("outlook", sorted(trig)[0][0] if trig else min(brown), "PSA outlook on the first day of concern")
    elif gfile and manifest["fields"].get("qpf", {}).get("total"):
        tot = np.nansum(np.stack([G["qpf"].astype(float)[k] for k in range(len(gdays))]), axis=0)
        if np.nanpercentile(tot, 75) >= 0.10:
            feat = ("qpf", "total", "7-day precipitation")
        else:
            lows = [np.nanpercentile(G["rh"].astype(float)[k], 10) for k in range(len(gdays))]
            feat = ("rh", int(np.nanargmin(lows)), "Minimum RH on the driest day")
    manifest["featured"] = {"field": feat[0], "day": feat[1], "caption": feat[2]}
    O["maps"] = manifest
    json.dump(O, open(a.outlook, "w"), indent=1, default=str)
    print(f"Maps written to {a.out_dir}; featured: {feat}")


def cmd_blend(a):
    cfg = json.load(open(a.config))
    # Temperatures from the FEMS climatology fill any PSA you have not set by hand
    climo = json.load(open(a.climo)) if os.path.exists(a.climo) else {}
    for p in SW_PSAS:
        t = (climo.get(p) or {}).get("tmax") or {}
        pc = cfg["psas"].setdefault(p, {})
        if pc.get("hot_day_f") is None and t.get("hot_day"):
            pc["hot_day_f"] = t["hot_day"]
        if pc.get("peak_tmax_f") is None and t.get("much_above"):
            pc["peak_tmax_f"] = t["much_above"]
    parts = [pd.read_csv(f) for f in sorted(glob.glob(os.path.join(a.partials, "*.csv")))]
    now = datetime.now(timezone.utc)
    if parts:
        allm = pd.concat(parts)
        allm["date"] = pd.to_datetime(allm["date"]).dt.date
        days = [now.astimezone(MST).date() + timedelta(days=i) for i in range(7)]
        used = allm.groupby("model")["init"].first().reset_index().to_dict("records")
        r = lambda x, n=1: None if x is None or pd.isna(x) else round(float(x), n)
        psas = {}
        for psa in SW_PSAS:
            rec = {f: [] for f in FIELDS}
            rec["spread"] = {f: [] for f in FIELDS}
            rec["members"] = {}
            sub = allm[allm.psa == psa]
            for i, d in enumerate(days):
                day = sub[sub.date == d]
                for f in FIELDS:
                    v = day[f].dropna()
                    nd = 2 if f == "qpf" else 1
                    rec[f].append(r(v.median(), nd) if len(v) else None)
                    rec["spread"][f].append([r(v.min(), nd), r(v.max(), nd)] if len(v) else None)
                for _, row in day.iterrows():
                    m = rec["members"].setdefault(row.model, {f: [None] * 7 for f in FIELDS})
                    for f in FIELDS:
                        m[f][i] = r(row[f], 2)
            psas[psa] = rec
        days = [d.isoformat() for d in days]
    elif os.path.exists(a.reuse):
        prev = json.load(open(a.reuse))
        days, psas, used = prev["days"], prev["psas"], prev["run"]["models"]
        print("No partials; reusing previous model blend.")
    else:
        days = [(now.astimezone(MST).date() + timedelta(days=i)).isoformat() for i in range(7)]
        psas, used = {p: {} for p in SW_PSAS}, []

    # fuels + fire on the ground
    fuel_fields = ["erc", "bi", "bi_pct", "fires"] + [f for f in FEMS_FIELDS if f not in ("erc", "bi")] + ["n_stations", "source"]
    for p in SW_PSAS:
        for f in fuel_fields:
            psas[p][f] = [None] * 7
    # 1) FEMS NFDRS (observed + forecast), 2) manual data/fuels.csv overrides it
    if os.path.exists(a.fems) and os.path.getsize(a.fems) > 0:
        fe = pd.read_csv(a.fems)
        for _, row in fe.iterrows():
            p, d = str(row["psa"]).upper(), str(row["date"])[:10]
            if p in psas and d in days:
                for f in fuel_fields:
                    if f in row and not pd.isna(row[f]):
                        psas[p][f][days.index(d)] = row[f] if f == "source" else round(float(row[f]), 1)
        print(f"Merged FEMS NFDRS from {a.fems}")
    if os.path.exists(a.fuels) and os.path.getsize(a.fuels) > 0:
        fu = pd.read_csv(a.fuels)
        for _, row in fu.iterrows():
            p, d = str(row["psa"]).upper(), str(row["date"])[:10]
            if p in psas and d in days:
                for f in ("erc", "bi_pct", "fires"):
                    if f in row and not pd.isna(row[f]):
                        psas[p][f][days.index(d)] = float(row[f])
                        if f == "erc":
                            psas[p]["source"][days.index(d)] = "manual"

    # Fires on the ground (WFIGS), same count every day unless data/fuels.csv sets one
    fires = json.load(open(a.fires)) if os.path.exists(a.fires) else None
    if fires:
        for p in SW_PSAS:
            for i in range(7):
                if psas[p]["fires"][i] is None:
                    psas[p]["fires"][i] = fires["counts"].get(p, 0)

    out = {p: [classify(cfg, p, psas[p], i) for i in range(7)] for p in SW_PSAS}

    # forecaster overrides win
    if os.path.exists(a.overrides) and os.path.getsize(a.overrides) > 0:
        ov = pd.read_csv(a.overrides, dtype=str).fillna("")
        for _, row in ov.iterrows():
            d = row["date"][:10]
            if d not in days:
                continue
            targets = SW_PSAS if row["psa"].upper() == "ALL" else [row["psa"].upper()]
            for p in targets:
                if p in out:
                    c = out[p][days.index(d)]
                    c["model_call"] = {"category": c["category"], "trigger": c["trigger"]}
                    c["category"] = row["category"] or c["category"]
                    c["trigger"] = row["trigger"] or None
                    c["why"].insert(0, f"Forecaster override: {row.get('note') or 'manual call'}.")
                    c["override"] = True

    disc_md = open(a.discussion).read() if os.path.exists(a.discussion) else ""
    meta = {"created": now.strftime("%Y-%m-%dT%H:%MZ"), "models": used,
            "method": "Median of available models per PSA-day; spread = model min/max."}
    os.makedirs(a.out_dir, exist_ok=True)
    json.dump({"run": meta, "days": days, "psas": psas}, open(os.path.join(a.out_dir, "forecast.json"), "w"), default=str)
    json.dump({"run": meta, "days": days, "outlook": out, "discussion": disc_md,
               "auto_discussion": (load_regions(a.psa), auto_discussion(days, psas, out, fires, cfg))[1],
               "fires": (fires or {}).get("fires", []),
               "thresholds": {k: v for k, v in cfg.items() if k not in ("psas", "_note")}},
              open(os.path.join(a.out_dir, "outlook.json"), "w"), default=str, indent=1)
    print(f"Wrote {a.out_dir}/forecast.json and outlook.json ({len(used)} models)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("fetch-psa"); s.add_argument("--out", default="docs/psa.geojson")
    s = sub.add_parser("model")
    s.add_argument("--model", required=True, choices=list(MODELS) + [m.lower() for m in MODELS])
    s.add_argument("--psa", default="docs/psa.geojson"); s.add_argument("--init")
    s.add_argument("--out")
    s = sub.add_parser("blend")
    s.add_argument("--partials", default="partials"); s.add_argument("--config", default="config/thresholds.json")
    s.add_argument("--fuels", default="data/fuels.csv"); s.add_argument("--overrides", default="data/overrides.csv")
    s.add_argument("--discussion", default="data/discussion.md"); s.add_argument("--reuse", default="docs/forecast.json")
    s.add_argument("--out-dir", default="docs"); s.add_argument("--fems", default="docs/fems.csv")
    s.add_argument("--fires", default="docs/fires.json"); s.add_argument("--climo", default="config/fems_climo.json")
    s.add_argument("--psa", default="docs/psa.geojson")
    s = sub.add_parser("maps")
    s.add_argument("--psa", default="docs/psa.geojson"); s.add_argument("--outlook", default="docs/outlook.json")
    s.add_argument("--partials", default="partials"); s.add_argument("--out-dir", default="docs/maps")
    s = sub.add_parser("fires")
    s.add_argument("--psa", default="docs/psa.geojson"); s.add_argument("--config", default="config/thresholds.json")
    s.add_argument("--out", default="docs/fires.json")
    for name in ("fems", "fems-climo"):
        s = sub.add_parser(name)
        s.add_argument("--psa", default="docs/psa.geojson")
        s.add_argument("--stations", default="config/stations.json", help="your PSA -> station id lists")
        s.add_argument("--discovered", default="docs/fems_stations.json")
        s.add_argument("--rediscover", action="store_true")
        s.add_argument("--fuel-model", default="Y")
        s.add_argument("--config", default="config/thresholds.json")
        s.add_argument("--climo", default="config/fems_climo.json")
    sub.choices["fems"].add_argument("--out", default="docs/fems.csv")
    c = sub.choices["fems-climo"]
    c.add_argument("--out", default="config/fems_climo.json")
    c.add_argument("--start-year", type=int, default=2015); c.add_argument("--end-year", type=int, default=2024)
    c.add_argument("--start-month", type=int, default=1); c.add_argument("--end-month", type=int, default=12)
    a = ap.parse_args()
    if a.cmd == "model" and not a.out:
        a.out = f"partials/{a.model.upper()}.csv"
    {"fetch-psa": cmd_fetch_psa, "model": cmd_model, "blend": cmd_blend,
     "fems": cmd_fems, "fems-climo": cmd_fems_climo, "fires": cmd_fires, "maps": cmd_maps}[a.cmd](a)


if __name__ == "__main__":
    main()
