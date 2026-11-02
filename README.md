# Southwest 7-Day Significant Fire Potential Map

A GitHub Pages map of the 15 current Southwest Predictive Service Areas (SW01–SW14N), colored for each of the next 7 days with the NWCG 7-Day categories:

| Color | Meaning |
|---|---|
| Gray | No data (no ERC for the PSA-day) |
| Green | Little or no risk for large fire |
| Yellow | Low risk of large fire in the absence of a High Risk event |
| Brown | Moderate risk of large fire in the absence of a High Risk event |
| Orange **W** | Wind: sustained ≥ 20 mph or gusts ≥ 35 mph with RH ≤ 15% |
| Red **L** | Lightning: scattered storms, RH < 20%, hot days before, dry or windy after |
| Orange **B** | Burn environment: temps much above peak-season max, RH ≤ 15%, unstable |

**It runs itself. You do not need to edit any files.** Twice a day (5:45 am and 11:30 pm MST; 6:45 am and 12:30 am in daylight time) GitHub Actions:

1. Runs every model Herbie can reach (HRRR, RAP, NAM nest, NBM, GFS, GEFS mean, ECMWF IFS and AIFS) and blends them by median for each PSA and day.
2. Pulls NFDRS fuels from FEMS: ERC, BI, IC, SC, 1/10/100/1000-hr fuel moisture, KBDI, herbaceous and woody fuel moisture, GSI.
3. Counts active wildfires (100+ acres, not fully contained) in each PSA from the NIFC WFIGS incident service.
4. Colors every PSA-day, writes a one- or two-paragraph discussion, draws maps (PSA outlook, minimum RH, peak gusts, precipitation) with PSA outlines, and publishes the site.

Each Forecast run starts by checking the FEMS climatology, before FEMS is pulled and the map is drawn. It rebuilds (about 5 minutes) only when it is missing, over a year old, or built for different PSA boundaries, using the last 10 years of FEMS data for each PSA: BI and ERC percentiles, day-of-year normals for every FEMS value (shown beside the current values on the map), and peak-season temperature references. To force a rebuild, run **Actions → Climatology**, which starts a Forecast run that rebuilds it.

## Set up (one time)

1. Create a new GitHub repository and upload everything in this folder (keep the `.github` folder).
2. **Settings → Pages**: Source = **GitHub Actions**. (The workflow publishes the site itself, so the twice-daily map images never pile up in the repository.)
3. **Settings → Actions → General → Workflow permissions**: choose *Read and write permissions*. Save.
4. **Actions → Forecast → Run workflow** to get the first map right away (20–40 minutes). Everything after that is automatic.
5. Open `https://<your-user>.github.io/<repo>/`.

The first Forecast run also builds the climatology, so it takes about 5 minutes longer than later runs.

## Data sources, all automatic

| Data | Source | Notes |
|---|---|---|
| Wind, gusts, RH, temperature, QPF, thunder, instability, HDWI | Herbie: 9 models | Median blend; the map shows the spread and each model |
| ERC, BI, IC, SC, fuel moistures, KBDI, GSI | FEMS public download service | Past 5 days observed + 7-day NFDRS forecast. Observed beats forecast for the same station and day. No account needed. |
| FEMS stations in each PSA | FEMS station API | Every active NFDRS station inside each PSA, saved to `docs/fems_stations.json` and rebuilt when boundaries change |
| Fires on the ground | NIFC WFIGS current incidents | Wildfires of 100+ acres not fully contained; shown as red dots on the map |
| BI percentiles, FEMS normals, peak-season temperatures | FEMS history, checked at the start of every Forecast run | Rebuilt yearly or when boundaries change |
| ERC percentiles | SWCC 2015–2024 breakpoints | In `config/thresholds.json` |
| PSA boundaries | NWCG 7-Day Significant Fire Potential map service (current 15 PSAs) | Refreshed every run; the last good copy is kept if NWCG is down |

## Dry lightning potential

For each afternoon model step (HRRR, RAP, NAM nest, GFS, IFS and AIFS), every grid point where CAPE is at least 150 J/kg gets a 0-7 score from four ingredients:

| Ingredient | 2 points | 1 point |
|---|---|---|
| Precipitable water | 0.60 in or less | 0.80 in or less |
| Cloud base (surface LCL, ft AGL) | 10,000 ft or higher | 8,000 ft or higher |
| DCAPE (parcel from the driest 700-500 mb level, wet-bulb cooled, moist descent to the surface) | 1,200 J/kg or more | 800 J/kg or more |
| Storm motion (mean 700/500/300 mb wind) | 20 kt or more = 1 point | 15 kt or more = 0.5 |

A score of 3 or more is moderate and 5 or more is high. A PSA is **High** when at least 15% of it scores high, **Moderate** when 15% scores moderate, and **Low** at 5%. High dry lightning potential sets the **L** trigger when fuels are at least low risk. The map has a Dry lightning tab (GFS), the right-hand panel lists each ingredient, and the discussion calls out high-based, low-precipitation storms. All thresholds are the `dlp_*` settings in `config/thresholds.json`.

## Optional adjustments

Nothing here is required. Edit on github.com with the pencil icon; committing starts a fresh run and the site updates about 5-10 minutes later.

| File | Use it to |
|---|---|
| `data/overrides.csv` | Make a final call that replaces the automatic one: `psa,date,category,trigger,note`. `psa` can be `ALL`; category `green`/`yellow`/`brown`/`nodata`; trigger `W`/`L`/`B` or blank. Old dates are ignored. |
| `data/discussion.md` | Add forecaster notes above the automatic discussion. Leave it empty to show only the automatic one. |
| `data/fuels.csv` | Override FEMS ERC or BI percentile, or the fire count, for a PSA and day. |
| `config/stations.json` | List your SIG station ids per PSA so ERC averages match SWCC's charts exactly. |
| `config/thresholds.json` | Change any threshold. `erc_smoothing_days: 2` uses a 2-day moving average like SWCC's charts. |

## How a PSA-day is colored

1. ERC → percentile from the PSA's SWCC breakpoints. Below 60th = green, 60th–89th = yellow, 90th and up = brown.
2. Up one level if BI ≥ 90th percentile, HDWI ≥ threshold, or ≥ 3 active wildfires of 100+ acres.
3. ≥ 0.25" QPF that day or the day before → green. ≥ 0.10" → down one level.
4. If yellow or brown, check the W, L and B triggers with the blended model weather. L shows first, then W, then B.
5. Rows in `overrides.csv`, if any, replace the result.

Weather per PSA-day (local MST, afternoon 11 am–6 pm): 90th percentile of grid points for wind, gust, max temp and HDWI; 10th percentile for min RH (the driest part of the PSA); median for QPF; % of the PSA with CAPE ≥ 250 J/kg and rain for thunder coverage; mean 700–500 mb lapse rate for instability. HDWI is a near-surface proxy (VPD × wind), not the full lowest-500 m calculation.

## Run locally

```
pip install -r requirements.txt
python scripts/psa_pipeline.py fetch-psa
python scripts/psa_pipeline.py model --model GFS
python scripts/psa_pipeline.py fems
python scripts/psa_pipeline.py fires
python scripts/psa_pipeline.py blend
python -m http.server -d docs
```

## Sources

- ERC breakpoints: SWCC, 2015–2024, Fuel Model Y (CY2025, current PSAs).
- PSA boundaries: NIFC Open Data, National Predictive Service Areas (PSA) Boundaries.
- Fires: NIFC WFIGS Incident Locations (Current).
- Fuels: FEMS (Fire Environment Mapping System) public download and climatology services.
- Official product: https://fsapps.nwcg.gov/psp/npsg/forecast/#/outlooks?state=map&gaccId=10
