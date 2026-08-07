# SolarMap

Detects rooftop solar panels in aerial imagery and outputs them as
georeferenced polygons with area and estimated capacity.

```
  imagery source ──▶ georeferenced tiles ──U-Net──▶ probability maps
   (tiles or GE Pro)                                       │
        Leaflet UI ◀── detections.geojson ◀── polygonise + dedupe
```

Every captured image is paired with a world file, so detections come out as
real-world polygons with m² and kW — GIS data, not boxes on a picture.

## Two capture backends

The capture stage is the only part that touches an imagery source; everything
downstream reads `manifest.json` and doesn't care where the pixels came from.

| | **XYZ tiles** (default) | **Google Earth Pro** |
|---|---|---|
| Setup | none | install + configure the desktop app |
| Georeferencing | exact (computed tile bounds) | approximate (measured from a GUI) |
| Speed | parallel, ~8 tiles/s | ~4 s per view, serial |
| Resumable | yes | no |
| Machine usable meanwhile | yes | **no** — it drives the live window |
| Commercial licensing | yes, with Mapbox/NAIP | no |
| Longevity | fine | **discontinued: no downloads after 25 June 2027** |

**Use tiles unless you have a specific reason not to.** The Earth Pro backend
is kept because its imagery is sometimes newer or higher-resolution in
particular areas, but it depends on an undocumented COM API in an app Google
is winding down.

## Setup

```powershell
pip install -r requirements.txt
```

Torch is only needed for training and detection — capturing imagery and
browsing results work without it.

## Use

### Capture (tiles — recommended)

```powershell
python scripts/capture_tiles.py --bbox 36.801,-119.805,36.8045,-119.800 `
    --name fresno --source esri
```

Add `--dry-run` to see the resolution and tile count before committing. Zoom
is chosen automatically from `--gsd` (default 0.15 m/px), capped at each
source's maximum.

Sources are configured in `config.yaml`:

| Source | Max zoom | Notes |
|---|---|---|
| `esri` | 19 | No key needed. ~0.24 m/px at mid-latitudes. Good for evaluation. |
| `mapbox` | 22 | Free token from <https://account.mapbox.com/>. Finest resolution, licensed for commercial use. |
| `naip` | 18 | US only, public domain, cleanest licensing. |

Re-running skips blocks already on disk, so an interrupted capture resumes.

**Check coverage before capturing a new region:**

```powershell
python scripts/check_coverage.py --at 33.5384,35.4318
```

A source's advertised max zoom is a best case, not a promise — coverage varies
enormously by region, and can differ between towns 15 km apart. Services return
a *"Map data not yet available"* placeholder with **HTTP 200**, not a 404, so
the only reliable test is to fetch a tile and look at it. The capture code
detects these placeholders, drops them, and warns; it never records them as
imagery. Silently keeping them would make a survey report zero panels across an
area it never actually saw.

### Capture (Google Earth Pro — optional)

Requires the desktop app from <https://www.google.com/earth/about/versions/>
(free, no account or licence key). Launch it once to register its COM server,
then verify the automation API is present on your build:

```powershell
python scripts/check_earth.py
```

Configure the window once — `SaveScreenShot` grabs the render window at its
on-screen size, so what you see is what you get:

- Maximise the window (bigger window → finer ground sample distance).
- `View ▸ Sidebar` off, `View ▸ Status Bar` off, `View ▸ Navigation ▸ Never`.
- `Tools ▸ Options ▸ 3D View`: **Terrain off** — a flat globe keeps the view
  planar, which is what the world-file georeferencing assumes.

Then calibrate and capture:

```powershell
python scripts/calibrate.py --at 37.7749,-122.4194 --alt 150
python scripts/capture_aoi.py --bbox 37.770,-122.430,37.780,-122.415 --name mission
```

`calibrate.py` reports your actual ground sample distance and how many pixels
a PV module spans. **Below ~6 px/module, only large arrays will be found.** It
also writes a sample frame — if any Earth UI is still visible in it, raise the
matching `capture.crop` fraction in `config.yaml`. Extents shrink to match the
crop automatically.

Don't touch the mouse or keyboard while a capture runs.

### Train

There is no pretrained checkpoint in this repo. Bootstrap from public data:

| Dataset | Notes |
|---|---|
| [**BDAPPV**](https://zenodo.org/records/7358126) (Kasmi et al. 2023, *Sci. Data*) | ~28k images with segmentation masks, from Google imagery. Best domain match — same source as your captures. CC-BY. |
| [**Duke / Bradbury et al. 2016**](https://doi.org/10.6084/m9.figshare.3385780) (*Sci. Data*) | ~19k PV arrays over 4 California cities, USGS 0.3 m aerial. Public domain. |

**Easiest path — `notebooks/train_solarmap.ipynb`.** Upload it to
[Colab](https://colab.research.google.com/), pick a T4 GPU, run top to bottom.
It downloads BDAPPV, builds the split, trains, reports per-array precision and
recall, and hands back `solar_unet.pt`. Drop that into `models/` and restart
the server. Roughly 40–60 minutes.

Locally (needs a GPU to be practical):

```powershell
python scripts/prepare_dataset.py --src data/raw/bdappv/google --name bdappv --keep-negatives
python scripts/train.py --dataset data/datasets/bdappv
```

`--keep-negatives` includes unannotated tiles as empty masks, which is what
suppresses false positives on skylights, dark flat roofs and swimming pools.

**This machine has no NVIDIA GPU**, so training here will take many hours.
Run `train.py` on Colab or a GPU box and copy `models/solar_unet.pt` back —
inference on captured tiles is fine on CPU.

### Detect

```powershell
python scripts/detect.py --capture mission
```

Writes `data/captures/mission/detections.geojson` — one polygon per array with
`area_m2`, `capacity_kw` and `confidence`, plus totals in the collection
properties. Loads directly into QGIS, ArcGIS or any web map.

### UI

```powershell
python scripts/serve.py
```

<http://127.0.0.1:8000> — pan to an area, *Use current map view* to set the
AOI, run capture and detection with live progress, and browse the results as
clickable polygons over the captured imagery.

---

## How the pieces fit

| Path | Role |
|---|---|
| `src/solarmap/capture/webmercator.py` | Slippy-map tile arithmetic, in Mercator metres |
| `src/solarmap/capture/tiles.py` | XYZ tile backend: parallel fetch, block stitching, resume |
| `src/solarmap/capture/earth_com.py` | COM wrapper: camera, streaming wait, view extents, screenshot |
| `src/solarmap/capture/grid.py` | AOI → camera grid. Spacing is **probed**, not assumed: it measures one real view's footprint and lays the grid out from that |
| `src/solarmap/capture/capture.py` | The Earth Pro capture loop |
| `src/solarmap/geo.py` | World files, crop-aware extents, UTM zone selection, GSD |
| `src/solarmap/model/` | U-Net (ResNet-34 encoder), Dice+BCE loss, dataset, training loop |
| `src/solarmap/infer/predict.py` | Overlapping sliding-window inference, averaged |
| `src/solarmap/infer/vectorize.py` | Mask → polygons, area in UTM, cross-tile dedupe |
| `src/solarmap/api/` + `web/` | FastAPI backend and Leaflet UI |

Design decisions worth knowing:

- **Overlapping capture tiles and overlapping inference windows.** Both exist
  so an array sitting on a boundary isn't split in two. The cost is that
  boundary arrays get detected twice, which `dedupe_across_tiles` merges by
  IoU — without it, installed capacity is over-counted. (Tile captures are
  seamless rather than overlapping, so only the inference windows overlap.)
- **Areas are computed in UTM, never in degrees.** A square degree is not an
  area, and its size varies with latitude.
- **Tile captures are georeferenced in Web Mercator, not lat/lon.** Slippy
  tiles are square in Mercator metres; their latitude span shrinks toward the
  poles. The manifest carries both `bounds_proj` (EPSG:3857, used for
  georeferencing) and WGS84 bounds (used for map display), so nothing is
  stretched. Earth Pro captures carry no `bounds_proj` and stay in lat/lon.
- **Capacity is an estimate.** `capacity.kw_per_m2: 0.19` derates mainstream
  crystalline-silicon module output (~200–220 W/m² of module) to account for
  the inter-module gaps included in a detected footprint. Tune it against
  known installations in your region before quoting the numbers.

## Measured results at Jbeil

Validated against 244 hand-verified arrays at Notre Dame des Secours,
re-derivable with `python scripts/evaluate.py --capture jbeil-nds`:

```
154 panels · 21,928 m2 · ~4.2 MW
precision 67.8%   ·   73.1% of arrays located
```

Areas are traced outlines, so they are measurements rather than bounding-box
upper bounds.

### Four labelling rounds, and why they stopped helping

| round | labels added | precision | arrays | F1 |
|---|---|---|---|---|
| 1 | — | 63.1% | 76.9% | 0.694 |
| **2** | +16 / +13 | **67.8%** | **73.1%** | **0.703** |
| 3 | +11 / −27 | 98.3% | 40.2% | 0.570 |
| 4 | +10 / +69 | 89.6% | 41.4% | 0.566 |

Rounds 3 and 4 added far more rejections than acceptances, and the classifier
learned selectivity: precision rose, recall collapsed. Round 2 is shipped.
Higher-precision variants are in `models/archive/` if a confirmed subset
matters more than coverage.

### Nine approaches that did not beat it

YOLO fine-tuning (3 variants, best 15.6%/9.6%); a patch CNN (24% precision);
U-Net on the Jbeil labels (IoU 0.39, overfit); U-Net on BDAPPV — 22,615
European rooftops, val IoU 0.757, 91% recall on its own data, **1.7% at
Jbeil**; glare handling (precision 68% → 44%); threshold sweeps (F1 falls
monotonically); grid-periodicity features (cross-validated F1 up, real F1 down
0.703 → 0.612); size-conditional thresholds (all below baseline).

The binding constraint is resolution. At 24.7 cm/px a 30 m² array is 31×16
pixels and measures the same as a rooftop water tank. Sub-15 cm imagery over
Lebanon means a drone survey — Nearmap and Vexcel do not fly the region, and
Pléiades Neo is 30 cm native.

## Imagery licensing

Each source carries its own terms, and the manifest records the required
attribution, which the UI displays.

- **Mapbox** — commercial use permitted under your account's plan. The right
  choice for anything you intend to ship.
- **NAIP** — US federal imagery, public domain. The cleanest licensing.
- **Esri World Imagery** — the public endpoint is fine for evaluation and
  research; production use wants an ArcGIS licence.
- **Google Earth Pro** — personal and research use with attribution.
  Systematic bulk capture feeding a commercial product is *not* covered, and
  desktop downloads end 25 June 2027.
