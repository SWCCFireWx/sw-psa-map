# Southwest 7-Day Significant Fire Potential Map

A GitHub Pages map of the 16 Southwest Predictive Service Areas (SW01–SW14S), colored for each of the next 7 days with the NWCG 7-Day categories:

| Color | Meaning |
|---|---|
| Gray | No data (no ERC for the PSA-day) |
| Green | Little or no risk for large fire |
| Yellow | Low risk of large fire in the absence of a High Risk event |
| Brown | Moderate risk of large fire in the absence of a High Risk event |
| Orange **W** | Wind: sustained ≥ 20 mph or gusts ≥ 35 mph with RH ≤ 15% |
| Red **L** | Lightning: scattered storms, RH < 20%, hot days before, dry or windy after |
| Orange **B** | Burn environment: temps much above peak-season max, RH ≤ 15%, unstable |

GitHub Actions runs every model Herbie can reach (HRRR, RAP, NAM nest, RRFS, NBM, GFS, GEFS mean, ECMWF IFS and AIFS) twice a day, blends them by median for each PSA and day, applies the criteria in `config/thresholds.json`, and commits the result to `docs/`, which GitHub Pages serves.

## Set up (one time)

1. Create a new GitHub repository and upload everything in this folder (keep the `.github` folder).
2. **Settings → Pages**: Source = *Deploy from a branch*, Branch = `main`, folder = `/docs`. Save.
3. **Settings → Actions → General → Workflow permissions**: choose *Read and write permissions*. Save.
4. **Actions → Forecast → Run workflow.** The first run downloads the PSA boundaries from NIFC and runs all models (about 20–40 minutes).
5. Open `https://<your-user>.github.io/<repo>/`.

After that it runs on its own at 6:40 am and 12:40 pm MST.

## FEMS fuels (automatic)

Each forecast run also pulls NFDRS from FEMS (fuel model Y): ERC, BI, IC, SC, 1/10/100/1000-hr fuel moisture, KBDI, herbaceous and woody fuel moisture, and GSI. It uses FEMS's public download service, so no account or API key is needed.

- It pulls the past 5 days of observations and the 7-day NFDRS forecast. For the same station and day, observed beats forecast.
- Station values are averaged into one value per PSA. The map shows whether a PSA-day came from observations, forecasts, or both, and how many stations went in.
- **Stations:** the first run finds every active NFDRS station in AZ, NM and far west TX through the FEMS station API, assigns each to the PSA it sits in, and saves the list to `docs/fems_stations.json`. To match SWCC's ERC charts exactly, put your SIG station ids for each PSA in `config/stations.json`; those win over the automatic list.
- **BI percentile:** run **Actions → Climatology → Run workflow** once. It builds BI and ERC percentile breakpoints for each PSA from FEMS 2015–2024 history and saves them to `config/fems_climo.json`. Until then BI shows as a raw value and cannot raise a category.
- ERC is ranked against SWCC's published breakpoints in `config/thresholds.json`. Set `erc_smoothing_days` to 2 to use a 2-day moving average like SWCC's charts.
- A number you type into `data/fuels.csv` overrides FEMS for that PSA and day.

## Daily forecaster inputs

Edit these files right on github.com (pencil icon). Each commit re-colors the map in about a minute without rerunning the models.

| File | What goes in it |
|---|---|
| `data/fuels.csv` | `psa,date,erc,bi_pct,fires` — active fires on the ground (IMSR/IRWIN), plus any ERC or BI percentile you want to override FEMS with. |
| `data/overrides.csv` | `psa,date,category,trigger,note` — your final call. `psa` can be `ALL`. Category: `green`, `yellow`, `brown`, `nodata`. Trigger: `W`, `L`, `B` or blank. Overrides win over the model-based call; old dates are ignored automatically. |
| `data/discussion.md` | The written discussion shown under the map. `**bold**` and blank-line paragraphs work. |
| `config/thresholds.json` | Every threshold, the SWCC ERC percentile breakpoints for each PSA, and each PSA's hot-day and peak-season max temperature (fill these in for the L and B triggers). |

The workflow also writes a model summary for each day (peak gusts, lowest RH, rain, triggers) that you can use as a starting point for the discussion.

## How a PSA-day is colored

1. ERC → percentile from the PSA's SWCC breakpoints. Below 60th = green, 60th–89th = yellow, 90th and up = brown.
2. Up one level if BI ≥ 90th percentile, HDWI ≥ threshold, or ≥ 3 active fires.
3. ≥ 0.25" QPF that day or the day before → green. ≥ 0.10" → down one level.
4. If yellow or brown, check the W, L and B triggers with the blended model weather. L shows first, then W, then B.
5. Rows in `overrides.csv` replace the result.

Weather per PSA-day (local MST, afternoon 11 am–6 pm): 90th percentile of grid points for wind, gust, max temp and HDWI; 10th percentile for min RH (the driest part of the PSA); median for QPF; % of the PSA with CAPE ≥ 250 J/kg and rain for thunder coverage; mean 700–500 mb lapse rate for instability. HDWI is a near-surface proxy (VPD × wind), not the full lowest-500 m calculation.

## Run locally

```
pip install -r requirements.txt
python scripts/psa_pipeline.py fetch-psa
python scripts/psa_pipeline.py model --model GFS
python scripts/psa_pipeline.py blend
python -m http.server -d docs
```

## Sources

- ERC breakpoints: SWCC, 2015–2024, Fuel Model Y. SW14S uses the older WFAS chart (50th/90th/97th only); verify.
- PSA boundaries: NIFC Open Data, National Predictive Service Areas (PSA) Boundaries.
- Official product: https://fsapps.nwcg.gov/psp/npsg/forecast/#/outlooks?state=map&gaccId=10
