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
PSA_SERVICE = ("https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/"
               "National_PSA_Current/FeatureServer/0/query")

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


# ------------------------------------------------------------------ PSAs
def cmd_fetch_psa(a):
    q = urllib.parse.urlencode({"where": "PSANationalCode LIKE 'SW%'", "outFields": "PSANationalCode,PSAName",
                                "outSR": 4326, "f": "geojson"})
    with urllib.request.urlopen(f"{PSA_SERVICE}?{q}", timeout=120) as r:
        gj = json.load(r)
    import shapely.geometry as sg
    feats = []
    for f in gj["features"]:
        p = f["properties"]
        code = str(p.get("PSANationalCode", "")).upper().replace(" ", "")
        geom = sg.shape(f["geometry"]).simplify(0.005, preserve_topology=True)
        pt = geom.representative_point()
        feats.append({"type": "Feature", "geometry": sg.mapping(geom),
                      "properties": {"psa": code, "name": p.get("PSAName"), "lx": round(pt.x, 3), "ly": round(pt.y, 3)}})
    if not feats:
        sys.exit("No SW PSAs returned by the NIFC service.")
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


def latest_init(name, now):
    kw, _, _, step = MODELS[name]
    lag = 2 if step == 1 else 6
    t = now - timedelta(hours=lag)
    return t.replace(minute=0, second=0, microsecond=0, hour=(t.hour // step) * step)


def cmd_model(a):
    from herbie import Herbie
    import shapely

    name = a.model.upper()
    kw, hours, search, _ = MODELS[name]
    now = datetime.now(timezone.utc)
    days = [now.astimezone(MST).date() + timedelta(days=i) for i in range(7)]
    init = (datetime.strptime(a.init, "%Y-%m-%dT%H").replace(tzinfo=timezone.utc) if a.init
            else latest_init(name, now))
    psas = load_psas(a.psa)
    print(f"[{name}] init {init:%Y-%m-%d %HZ}", flush=True)

    masks, recs, cum, last_total = None, [], {}, None
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
    return {"category": NAMES[lvl], "trigger": trig[0] if trig else None, "all_triggers": trig, "why": why}


def auto_discussion(days, psas, out):
    lines = []
    for i, d in enumerate(days):
        cats = [out[p][i] for p in SW_PSAS]
        trig = {}
        for p, c in zip(SW_PSAS, cats):
            if c["trigger"]:
                trig.setdefault(c["trigger"], []).append(p)
        cnt = {k: sum(1 for c in cats if c["category"] == k) for k in ("green", "yellow", "brown", "nodata")}
        wx = lambda f, fn: [psas[p][f][i] for p in SW_PSAS if psas[p].get(f) and psas[p][f][i] is not None]
        g, rh, q = wx("gust", max), wx("minrh", min), wx("qpf", max)
        parts = [f"{datetime.fromisoformat(d):%a %b %-d}: "
                 + ", ".join(f"{v} {k}" for k, v in cnt.items() if v)]
        if g:
            gp = SW_PSAS[[psas[p]["gust"][i] if psas[p].get("gust") and psas[p]["gust"][i] is not None else -1 for p in SW_PSAS].index(max(g))]
            parts.append(f"peak gusts {max(g):.0f} mph ({gp})")
        if rh:
            parts.append(f"lowest min RH {min(rh):.0f}%")
        if q and max(q) >= 0.05:
            parts.append(f"wetting rain up to {max(q):.2f} in")
        for t, ps in trig.items():
            parts.append(f"{ {'W': 'Wind', 'L': 'Lightning', 'B': 'Burn environment'}[t]} trigger: {', '.join(ps)}")
        lines.append("; ".join(parts) + ".")
    return lines


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


def load_station_map(a):
    """config/stations.json (your SIG lists) wins; otherwise the discovered list in docs/fems_stations.json."""
    manual = json.load(open(a.stations)) if os.path.exists(a.stations) else {}
    manual = {k: v for k, v in manual.items() if not k.startswith("_") and v}
    auto = {}
    if len(manual) < len(SW_PSAS):
        if os.path.exists(a.discovered) and not getattr(a, "rediscover", False):
            auto = json.load(open(a.discovered))
        else:
            auto = fems_discover_stations(a.psa)
            json.dump(auto, open(a.discovered, "w"), indent=1)
            print(f"Discovered {sum(len(v) for v in auto.values())} FEMS stations -> {a.discovered}")
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
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"Wrote {a.out}")


def cmd_blend(a):
    cfg = json.load(open(a.config))
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
               "auto_discussion": auto_discussion(days, psas, out) if used else [],
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
     "fems": cmd_fems, "fems-climo": cmd_fems_climo}[a.cmd](a)


if __name__ == "__main__":
    main()
