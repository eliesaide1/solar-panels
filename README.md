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

**Use `notebooks/train_multiscale.ipynb`.** Upload it to
[Colab](https://colab.research.google.com/), pick a T4 GPU, run top to bottom.
It pretrains on BDAPPV with wide scale augmentation, fine-tunes on local
labels, and hands back `solar_unet.pt`. Drop that into `models/` and restart
the server. Roughly 45–70 minutes.

Two things it does that the older notebook did not, both of which matter more
than any hyperparameter here:

- **Scale augmentation spanning ~5–20 cm** rather than ±16%. Without it a
  checkpoint works at one gsd and collapses either side of it, which is the
  40× swing tabulated under [Measured results](#measured-results-at-jbeil).
- **Checkpoint selection on array F1**, using the same match rule as
  `score_arrays.py`, so the notebook's numbers predict the ones you get on a
  real capture. Validation IoU rewards tracing known arrays more tightly,
  which is not the goal.

Build the local fine-tuning set first:

```powershell
python scripts/prepare_masks.py --capture jbeil-mb --name jbeil06_seg
```

Prefer the native-resolution set (806 crops) over a resampled one (189 at
10.4 cm) — scale augmentation covers the operating point anyway, and 189 crops
overfits within a few epochs.

`notebooks/train_solarmap.ipynb` is kept for reference. It produced every
checkpoint currently in `models/`, and its narrow scale augmentation is why
they are resolution-brittle.

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

### Label

Each region needs its own labelling pass — and, on the evidence at Jbeil, it
needs to happen on imagery fine enough to decide on. Seed a round from
whatever already exists, then sweep:

```powershell
python scripts/build_label_seed.py --from jbeil-nds --to jbeil-mb
python scripts/serve.py
```

then open `/label.html?capture=jbeil-mb` — the capture name is required, there
is no picker. Click a box to cycle unreviewed → accepted → rejected, drag on
empty space to draw one, shift-click to delete. `U` jumps to the next tile
holding an undecided box, `S` saves. Saving is per tile and atomic, so a crash
costs at most the tile in progress.

`build_label_seed.py` carries confirmed arrays across as accepted, re-opens
previous *rejections* as undecided — a rejection made at 7 px is a coin flip
and deserves re-asking at 27 px — and drops destination boxes that already
contain a confirmed array, which is what stops flooded proposals becoming
ground truth.

**Sweep every tile, not just the seeded ones.** Recall measured against a truth
that was itself seeded by a detector is circular: arrays the detector never
proposed are absent from both sides and cannot show up as misses. Tiles with
no boxes still need opening and saving — an empty saved tile and an unvisited
one are indistinguishable on disk.

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
| `src/solarmap/infer/predict.py` | Overlapping sliding-window inference, averaged. **No rescaling** — windows are cropped at native gsd, which is why imagery must match the checkpoint's training scale |
| `src/solarmap/infer/cvfilter.py` | Classical proposals + learned box classifier. A 25 cm technique; see the proposal-stage note above |
| `src/solarmap/infer/vectorize.py` | Mask → polygons, area in UTM, cross-tile dedupe |
| `src/solarmap/api/` + `web/` | FastAPI backend, Leaflet map and labelling UI |

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
- **Imagery must match the checkpoint's training scale.** Inference crops
  windows at native resolution and never resamples, so a capture finer or
  coarser than the model was trained on degrades sharply — see the resolution
  table. Resample with `scripts/resample_capture.py` rather than recapturing.
- **A model must never be scored against the labels it was trained on.** Doing
  exactly that is what made the cv+filter detector look 43 points more precise
  than it is. Hold out ground truth, or label independently.
- **A tile swept and found empty is ground truth, not a gap.** `score_arrays.py`
  originally skipped tiles holding no verified array, which silently excluded 34
  of 81 tiles at Jbeil and every detection on them — worth 8.6 points of
  precision, and it hid the greenhouse problem entirely. Scoring now covers the
  whole capture; `--labelled-tiles-only` restores the old behaviour for captures
  that genuinely were not swept end to end.

## Measured results at Jbeil

> **Every number in this section was restated in August 2026.** The earlier
> figures — 67.8% precision, 73.1% of arrays located — were scored against a
> ground truth drawn on 24.7 cm/px imagery, and that ground truth was wrong.
> Re-labelling the same rooftops on 6.2 cm Mapbox imagery overturned 26% of
> its confirmed arrays as false positives and found 155 arrays it had missed
> entirely: just under half the real total. The old numbers were not an upper
> or a lower bound, they were wrong in both directions at once. See
> [Why the old numbers were wrong](#why-the-old-numbers-were-wrong).

Validated against **312 hand-verified arrays** at Notre Dame des Secours,
labelled on 6.2 cm imagery where a panel is ~27 px across. Every tile in the
AOI was swept exhaustively, so an unlabelled array is a genuine absence rather
than something the proposal stage never suggested.

```
python scripts/score_arrays.py --capture jbeil-mb-104 --labels labels_clean.json
```

| detector | array recall | detection precision | F1 |
|---|---|---|---|
| **fine-tuned pair, agreeing** | **56.5%** | **77.3%** | **0.653** |
| fine-tuned pair, unioned | 58.5% | 66.0% | 0.611 |
| multi-scale model alone @ 0.70 | 50.3% | 59.6% | 0.546 |
| single fine-tuned model @ 0.85 | 50.7% | 71.1% | 0.592 |
| U-Net fine-tuned (single-scale base) @ 0.85 | 50.7% | 78.0% | 0.614 |
| U-Net fine-tuned (multi-scale base) @ 0.50 | 60.1% | 58.2% | 0.591 |
| U-Net BDAPPV, no fine-tune @ 10.4 cm | 23.2% | 96.5% | 0.374 |
| cv+filter @ 24.7 cm | 27.6% | 24.5% | 0.260 |
| cv+filter @ 6.2 cm | 11.2% | 16.7% | 0.134 |
| U-Net BDAPPV, no fine-tune @ 6.2 cm | 0.6% | 86.7% | 0.013 |

The shipped detector runs **two U-Nets and keeps only what both of them find**
(`scripts/merge_detections.py --min-sources 2`). They differ only in how long
their BDAPPV stage ran and whether it used scale augmentation, which is enough
to make them fail on different roofs.

**Agreement beats union, on both axes at once.** Unioning the same two models
gives 58.5%/66.0%; requiring them to agree gives 56.5%/**77.3%**. A false
positive has to fool two independently pretrained models rather than one, and
that is a much harder thing to do. Agreement also nearly halves the detection
count, which matters whenever a human reviews the output.

It is also the best defence found against **greenhouses**, which are the
signature false positive here — long parallel polytunnels read very much like
panel rows. The ensemble puts 15 detections on the 34 array-free tiles against
27–37 for every single model, and it does so without having been designed for
it.

**Fine-tuning on local labels is what moved this**, from 23.2% to 56.5% recall.
Nothing else in three months of work came close, which is the argument for
labelling each new region rather than hoping a model transfers into it.

**Two metrics, and they answer different questions.** `evaluate.py` reports
*area* precision and coverage, which is what a capacity estimate needs.
`score_arrays.py` reports *arrays located*, which is what a survey needs. A
detector that traces 40% of every array scores 40% coverage while having found
every installation. Quote the one that matches the claim.

### The operating curve, and how to choose a point

Threshold is not a detail here — it moves recall by 40 points. Measured on the
multi-scale checkpoint against the same 306 arrays:

| threshold | array recall | precision |
|---|---|---|
| 0.15 | 74.8% | 28.1% |
| 0.30 | 66.7% | 47.9% |
| 0.50 | 60.1% | 58.2% |
| 0.70 | 50.3% | 69.6% |
| 0.85 | 36.6% | 86.0% |
| 0.95 | 9.2% | 92.1% |

Recall saturates near **75%**: going 0.30 → 0.15 buys three points while
precision falls from 47.9% to 28.1% and the detection count nearly doubles.
Past that the map is being flooded, not searched. `inference.threshold` in
`config.yaml` is a per-checkpoint decision — the default of 0.5 gives 27%
precision on some of these models — so re-derive it whenever the checkpoint
changes.

### Ensembling resolutions is worth less than ensembling models

A scale-robust checkpoint can be run at several ground sample distances and the
results unioned, on the theory that an array missed at 10 cm is caught at 8.
Measured, that theory is only slightly true:

| | recall | precision | F1 |
|---|---|---|---|
| 8.1 cm alone @0.70 | 55.8% | 65.1% | 0.601 |
| 10.4 cm alone @0.70 | 50.3% | 69.6% | 0.584 |
| 12.5 cm alone @0.70 | 42.8% | 72.1% | 0.584 |
| union of all three @0.70 | 60.1% | 59.4% | 0.598 |

The union beats every single resolution on recall, but its F1 is flat, and
60.1%/59.4% is the same trade as running one resolution at threshold 0.50. It
costs three times the inference for 3–4 points of recall at matched precision.

Combining two *different models* at the same resolution gains far more (0.653
against 0.592 and 0.546). Diversity of error is what an ensemble monetises, and
two separately pretrained models provide plenty of it; the same model shown the
same roof at 8 and 12 cm provides very little, because it fails the same way
both times.

**How the layers are combined matters as much as what is in them.** Union
maximises recall and lets any single model's mistakes through; agreement
(`--min-sources 2`) keeps only what both found and is better on both axes:

| combination | recall | precision | F1 |
|---|---|---|---|
| union @0.85 | 58.5% | 66.0% | 0.611 |
| **agree, 0.85 + 0.70** | **56.5%** | **77.3%** | **0.653** |
| agree, three models, 3 of 3 | 53.3% | 83.2% | 0.650 |
| agree, three models, 2 of 3 | 59.5% | 69.3% | 0.640 |

### Resolution is a tuning parameter, and it has a peak

Same model, same rooftops, same labels; only the imagery resampled with
`scripts/resample_capture.py`:

| gsd | array recall | precision |
|---|---|---|
| 6.2 cm | 0.6% | 86.7% |
| 8.1 cm | 14.7% | 96.9% |
| **10.4 cm** | **23.2%** | **96.5%** |
| 12.5 cm | 11.4% | 93.1% |
| 15.0 cm | 6.2% | 90.6% |
| 20.0 cm | 5.6% | 82.4% |

A 40× swing in recall from resampling alone, peaking at 10.4 cm with fall-off
proven on both sides. **Sharper imagery is not better imagery.** The peak sits
where BDAPPV was captured, because `RandomResizedCrop(scale=(0.7, 1.0))` varies
linear scale by only ±16% — the model only ever saw rooftops at one apparent
size. This is a property of the training pipeline, not of rooftops, and
`notebooks/train_multiscale.ipynb` widens it to roughly 5–20 cm.

Until a scale-robust checkpoint exists, resample captures to ~10 cm before
detection.

### Why the old numbers were wrong

The cv+filter detector was trained on human accept/reject decisions made at
24.7 cm/px, then scored against those same decisions. That is circular, and it
inflated precision from a true **24.5%** to an apparent 67.5%. Against
independent ground truth, three of every four things it flags are not panels.

Re-reviewing the 247 transferred arrays on 6.2 cm imagery:

| verdict at 6.2 cm | count | share |
|---|---|---|
| still a panel | 139 | 56% |
| **not a panel after all** | **63** | **26%** |
| deleted or redrawn | 45 | 18% |

plus **155 arrays the 25 cm pass never found**. At 7 px across, accept/reject
is close to a coin flip; at 27 px it is obvious.

This also explains the four labelling rounds that "stopped helping". Rounds 3
and 4 added mostly rejections and appeared to collapse recall, which read as a
data ceiling. It was not: the imagery could not support the decisions being
made. Re-labelling on sharp imagery moved the ground truth, not the model.

### Approaches that did not beat it

YOLO fine-tuning (3 variants, best 15.6%/9.6%); a patch CNN (24% precision);
U-Net on the Jbeil labels (IoU 0.39, overfit); glare handling (precision
68% → 44%); threshold sweeps on the box classifier (F1 falls monotonically);
grid-periodicity features (cross-validated F1 up, real F1 down 0.703 → 0.612);
size-conditional thresholds (all below baseline).

**Withdrawn: "U-Net on BDAPPV does not transfer — 1.7% at Jbeil."** That was an
artefact of scoring at 24.7 cm against the flawed labels. Measured properly the
same architecture, fine-tuned, is the best detector in the project by a wide
margin. Cross-region transfer was never the problem; scale mismatch and bad
ground truth were, and both were errors in how it was evaluated rather than
anything the model did.

**Training explicitly against the look-alikes did not work.** Greenhouses are
the signature false positive: long parallel polytunnels read very much like
panel rows. `prepare_masks.py` had been building crops only from tiles
containing a panel, so the 34 array-free tiles — nearly all the greenhouse
hillsides — never entered training at all. Adding them back
(`--include-empty-tiles`, 806 → 1,056 crops) and fine-tuning for 16 epochs
changed nothing: 37 false positives on those tiles at threshold 0.70, against
36 for the model that had never seen a greenhouse, and worse overall
(F1 0.568 against 0.653). Adding it to the ensemble as a third voter did not
help either — 3-of-3 gives 0.650, 2-of-3 gives 0.640.

The flag is kept because omitting negative tiles is still wrong in principle,
but on this evidence 312 positive examples is too little contrast to teach the
distinction. Ensembling suppresses greenhouses better than training against
them does, which was not the expected result.

**Withdrawn: "each region needs its own labelling pass, and the tooling makes
that roughly an hour."** The first half holds and is now the central finding —
fine-tuning on local labels is the only thing that has ever moved recall
materially. The second half does not. A real pass over 1.3 km² was 1,521
decisions across 81 tiles, and the proposal stage that made it fast at 24.7 cm
does not work at 6 cm. Budget a day per region, seeded by a previous model
rather than by classical proposals.

**The classical proposal stage does not survive high resolution.** A sweep of
108 threshold and morphology combinations at 6.2 cm
(`scripts/tune_proposals.py`) found no setting that keeps both coverage and
shape: the best covers 87.5% of arrays with 0.6% of proposals panel-shaped,
and where shape becomes usable coverage falls below 30%. At 6 cm the module
grid, the inter-module gaps and the glare are all resolved, so an array is no
longer a uniform drab blob — the mask either merges it into the rooftop or
shatters it. cv+filter is therefore a 25 cm technique, and the path forward is
segmentation.

### Reproducing any of this

| command | answers |
|---|---|
| `scripts/score_arrays.py` | how many arrays did it find? |
| `scripts/evaluate.py` | how much panel *area* did it find? |
| `scripts/resample_capture.py` | rebuild a capture at another gsd |
| `scripts/tune_proposals.py` | re-derive proposal thresholds for a resolution |
| `scripts/unet_ceiling.py` | threshold-independent recall ceiling of a checkpoint |
| `scripts/proposal_recall.py` | ceiling imposed by the proposal stage |
| `scripts/build_label_seed.py` | seed a labelling round on sharper imagery |
| `scripts/finetune_unet.py` | fine-tune a checkpoint on local labels (CPU, ~1 h) |
| `scripts/merge_detections.py` | dissolve several detection layers into one |

### What would get this past 60%

Every cheap lever has been pulled and measured: threshold tuning, resolution
ensembling, model ensembling, and how the ensemble combines. They are all in
the tables above, and together they took the detector from 23.2% of arrays at
96.5% precision to **56.5% at 77.3%**.

What remains is data. Jbeil holds **242 arrays/km²**, so the 1.3 km² labelled
here yielded 312 arrays; BDAPPV needed 22,615 rooftops to reach 91% recall on
its own imagery. Roughly **6 km²** would give ~1,500 local arrays, which is the
next honest step toward a survey-grade number.

The next pass is cheaper than this one was. A 77%-precision model can seed it,
which is a different proposition from the classical proposer that does not
function at 6 cm — the labeller confirms a good model's output instead of
adjudicating a bad heuristic's.

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
