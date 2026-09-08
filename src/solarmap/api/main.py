"""FastAPI backend for the SolarMap UI.

Capture and detection are long-running, so they are dispatched to background
threads and the UI polls ``/api/jobs/{id}``.

Only one capture may run at a time: Google Earth Pro is a single GUI instance
and two threads driving the same camera would interleave into garbage.
"""

from __future__ import annotations

import json
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..config import Config
from ..capture.capture import CaptureSession
from ..capture.grid import BBox
from ..capture.tiles import TileCapture, source_from_config

# NOTE: solarmap.infer pulls in torch, which is a large dependency and is not
# needed to capture imagery or browse results. It is imported inside the
# detect handler so the UI runs on a torch-free install.

cfg = Config.load()
app = FastAPI(title="SolarMap", version="0.1.0")

WEB_DIR = Path(__file__).resolve().parents[1] / "web"
# src/solarmap/api/main.py -> repo root -> scripts/
REPO_SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
CAPTURES = cfg.path("captures")
CAPTURES.mkdir(parents=True, exist_ok=True)

_earth_lock = threading.Lock()


# --------------------------------------------------------------------------
# jobs
# --------------------------------------------------------------------------


@dataclass
class Job:
    id: str
    kind: Literal["capture", "detect"]
    status: Literal["queued", "running", "done", "error"] = "queued"
    current: int = 0
    total: int = 0
    message: str = ""
    result: dict[str, Any] = field(default_factory=dict)


JOBS: dict[str, Job] = {}
_jobs_lock = threading.Lock()


def _set(job: Job, **kw) -> None:
    with _jobs_lock:
        for k, v in kw.items():
            setattr(job, k, v)


def _progress(job: Job):
    def fn(i: int, total: int, label: str) -> None:
        _set(job, current=i, total=total, message=label)
    return fn


def _spawn(kind: str, target) -> Job:
    job = Job(id=uuid.uuid4().hex[:12], kind=kind)
    JOBS[job.id] = job

    def wrapper():
        _set(job, status="running")
        try:
            result = target(job)
            _set(job, status="done", result=result or {}, message="complete")
        except Exception as exc:
            traceback.print_exc()
            _set(job, status="error", message=f"{type(exc).__name__}: {exc}")

    threading.Thread(target=wrapper, daemon=True).start()
    return job


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------


class CaptureRequest(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    south: float
    west: float
    north: float
    east: float
    # "earth" drives Google Earth Pro over COM; anything else is a key from
    # tile_sources in config.yaml.
    backend: str = "esri"
    eye_altitude_m: float | None = None
    zoom: int | None = None
    target_gsd_m: float = 0.15


class DetectRequest(BaseModel):
    name: str
    checkpoint: str | None = None


# --------------------------------------------------------------------------
# routes
# --------------------------------------------------------------------------


@app.get("/api/config")
def get_config():
    sources = cfg.get("tile_sources", {})
    backends = [
        {"id": k, "label": k, "max_zoom": v.get("max_zoom"),
         "needs_token": "{token}" in v.get("url", "") and not v.get("token")}
        for k, v in sources.items()
    ]
    backends.append({"id": "earth", "label": "Google Earth Pro (COM)", "max_zoom": None,
                     "needs_token": False})
    return {
        "eye_altitude_m": cfg["capture"]["eye_altitude_m"],
        "overlap": cfg["capture"]["overlap"],
        "kw_per_m2": cfg["capacity"]["kw_per_m2"],
        "threshold": cfg["inference"]["threshold"],
        "backends": backends,
        # .pkl is the classical-proposal + box-classifier detector.
        "checkpoints": sorted(
            p.name for p in cfg.path("checkpoints").iterdir()
            if p.suffix in (".pt", ".pkl")
        ),
    }


@app.get("/api/captures")
def list_captures():
    out = []
    for d in sorted(CAPTURES.iterdir()) if CAPTURES.exists() else []:
        mf = d / "manifest.json"
        if not mf.is_file():
            continue
        m = json.loads(mf.read_text(encoding="utf-8"))
        out.append(
            {
                "name": d.name,
                "aoi": m.get("aoi"),
                "tile_count": m.get("tile_count", 0),
                "captured_at": m.get("captured_at"),
                "has_detections": (d / "detections.geojson").is_file(),
                # Surface the panel count so the picker shows results, not just
                # how much imagery was captured.
                "detections": _detection_count(d),
            }
        )
    return out


def _detection_count(d: Path) -> int | None:
    f = d / "detections.geojson"
    if not f.is_file():
        return None
    try:
        return int(json.loads(f.read_text(encoding="utf-8"))["properties"]["detections"])
    except Exception:
        return None


@app.get("/api/captures/{name}/manifest")
def get_manifest(name: str):
    path = _safe_capture_dir(name) / "manifest.json"
    if not path.is_file():
        raise HTTPException(404, f"No capture named {name!r}")
    return JSONResponse(json.loads(path.read_text(encoding="utf-8")))


@app.get("/api/captures/{name}/detections")
def get_detections(name: str):
    path = _safe_capture_dir(name) / "detections.geojson"
    if not path.is_file():
        raise HTTPException(404, f"No detections for {name!r}; run detection first.")
    return JSONResponse(json.loads(path.read_text(encoding="utf-8")))


@app.get("/api/captures/{name}/tiles/{tile}")
def get_tile(name: str, tile: str):
    if "/" in tile or "\\" in tile or ".." in tile:
        raise HTTPException(400, "Bad tile name")
    path = _safe_capture_dir(name) / "tiles" / tile
    if not path.is_file():
        raise HTTPException(404, "No such tile")
    return FileResponse(path, media_type="image/jpeg")


@app.post("/api/capture")
def start_capture(req: CaptureRequest):
    try:
        aoi = BBox(south=req.south, west=req.west, north=req.north, east=req.east)
    except ValueError as exc:
        raise HTTPException(422, str(exc))

    name = _safe_name(req.name)
    local = Config.load()

    if req.backend == "earth":
        # Google Earth Pro is a single GUI instance; two threads driving the
        # same camera would interleave into garbage.
        if _earth_lock.locked():
            raise HTTPException(409, "A Google Earth capture is already running.")
        if req.eye_altitude_m:
            local.raw["capture"]["eye_altitude_m"] = req.eye_altitude_m

        def run(job: Job):
            with _earth_lock:
                out = CaptureSession(local).run(aoi, name, progress=_progress(job))
            return _summarise(out, name)
    else:
        try:
            source = source_from_config(local, req.backend)
        except SystemExit as exc:
            raise HTTPException(422, str(exc))

        def run(job: Job):
            out = TileCapture(local, source).run(
                aoi, name, zoom=req.zoom, target_gsd_m=req.target_gsd_m,
                progress=_progress(job),
            )
            return _summarise(out, name)

    return asdict(_spawn("capture", run))


def _summarise(out_dir: Path, name: str) -> dict:
    m = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    return {
        "name": name,
        "tile_count": m["tile_count"],
        "failed": len(m.get("failed", [])),
        "gsd_m": m.get("gsd_m"),
    }


@app.post("/api/detect")
def start_detect(req: DetectRequest):
    capture_dir = _safe_capture_dir(req.name)
    if not (capture_dir / "manifest.json").is_file():
        raise HTTPException(404, f"No capture named {req.name!r}")

    ckpt_dir = cfg.path("checkpoints")
    ckpt = ckpt_dir / (req.checkpoint or "solar_unet.pt")
    if not ckpt.is_file():
        raise HTTPException(
            404,
            f"Checkpoint {ckpt.name!r} not found in {ckpt_dir}. Train a model "
            "first (python scripts/train.py) or drop a .pt file there.",
        )

    try:
        from ..infer.pipeline import detect_capture
    except ImportError as exc:
        raise HTTPException(
            503,
            f"Inference dependencies are not installed ({exc}). "
            "Run: pip install -r requirements.txt",
        )

    def run(job: Job):
        out = detect_capture(cfg, capture_dir, ckpt, progress=_progress(job))
        gj = json.loads(out.read_text(encoding="utf-8"))
        return {"name": req.name, **gj["properties"]}

    return asdict(_spawn("detect", run))


class LabelBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    score: float = 0.0
    # None = not yet reviewed, True = accepted, False = rejected.
    verified: bool | None = None


class LabelSave(BaseModel):
    tile_id: str
    boxes: list[LabelBox]


@app.get("/api/captures/{name}/labels")
def get_labels(name: str):
    path = _safe_capture_dir(name) / "labels.json"
    if not path.is_file():
        raise HTTPException(
            404,
            f"No labels for {name!r}. Generate proposals first: "
            f"python scripts/propose_labels.py --capture {name}",
        )
    return JSONResponse(json.loads(path.read_text(encoding="utf-8")))


@app.post("/api/captures/{name}/labels")
def save_labels(name: str, payload: LabelSave):
    """Persist one tile's reviewed boxes.

    Saved per tile rather than in bulk so a browser crash costs at most the
    tile in progress.
    """
    path = _safe_capture_dir(name) / "labels.json"
    if not path.is_file():
        raise HTTPException(404, f"No labels file for {name!r}")

    data = json.loads(path.read_text(encoding="utf-8"))
    data.setdefault("tiles", {})[payload.tile_id] = [b.model_dump() for b in payload.boxes]

    # Write via a temporary file so an interrupted save cannot truncate the
    # labels accumulated so far.
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    tmp.replace(path)

    reviewed = sum(
        1 for boxes in data["tiles"].values() for b in boxes if b.get("verified") is not None
    )
    accepted = sum(
        1 for boxes in data["tiles"].values() for b in boxes if b.get("verified") is True
    )
    return {"saved": payload.tile_id, "reviewed": reviewed, "accepted": accepted}


class Correction(BaseModel):
    """One hand-drawn polygon, or one deletion, made on the map.

    Corrections are kept apart from ``detections.geojson`` on purpose: that file
    is rewritten in place by detect.py, merge_detections.py,
    curate_detections.py and build_survey_layer.py, so an edit stored there
    survives only until the next pipeline run. This file is never machine-
    written, so a correction is permanent and can be folded into the labels
    that actually train the model.
    """
    # [[lon, lat], ...] closed ring, WGS84.
    ring: list[list[float]] = Field(default_factory=list)
    # "add" = this is a real array the layer missed or traced wrongly.
    # "remove" = the detection at this location is not an array.
    kind: Literal["add", "remove"] = "add"
    note: str = ""
    # Set once the correction has been folded into the labels and the layer
    # rebuilt. Round-tripped through the client so a later save does not
    # present already-applied edits as pending work again.
    applied: str | None = None


class CorrectionSave(BaseModel):
    corrections: list[Correction]


@app.get("/api/captures/{name}/corrections")
def get_corrections(name: str):
    path = _safe_capture_dir(name) / "corrections.geojson"
    if not path.is_file():
        return JSONResponse({"type": "FeatureCollection", "features": []})
    return JSONResponse(json.loads(path.read_text(encoding="utf-8")))


@app.post("/api/captures/{name}/corrections")
def save_corrections(name: str, payload: CorrectionSave):
    """Replace the correction set for a capture.

    The whole set is sent each time rather than a delta: the editor holds the
    authoritative list, and a delta protocol would need conflict handling for a
    single-user local tool that has none.
    """
    cap = _safe_capture_dir(name)
    if not (cap / "manifest.json").is_file():
        raise HTTPException(404, f"No capture named {name!r}")

    feats = []
    for c in payload.corrections:
        if len(c.ring) < 4:
            raise HTTPException(422, "A polygon needs at least 3 distinct points.")
        ring = [list(map(float, p)) for p in c.ring]
        if ring[0] != ring[-1]:
            ring.append(ring[0])
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [ring]},
            "properties": {"kind": c.kind, "note": c.note, "source": "hand-drawn",
                           **({"applied": c.applied} if c.applied else {})},
        })

    doc = {
        "type": "FeatureCollection",
        "properties": {
            "capture": name,
            "count": len(feats),
            "note": "Hand corrections made in the map UI. Fold into labels with "
                    "scripts/apply_corrections.py; nothing reads this file "
                    "automatically.",
        },
        "features": feats,
    }
    path = cap / "corrections.geojson"
    # Same atomic write as the label editor: an interrupted save must not
    # truncate the corrections accumulated so far.
    tmp = path.with_suffix(".geojson.tmp")
    tmp.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    tmp.replace(path)
    return {"saved": len(feats),
            "added": sum(1 for f in feats if f["properties"]["kind"] == "add"),
            "removed": sum(1 for f in feats if f["properties"]["kind"] == "remove")}


class ApplyRequest(BaseModel):
    labels: str | None = None
    detections: str | None = None
    checkpoint: str = "solar_unet_ms.pt"


def _first_existing(cap: Path, *names: str) -> str | None:
    for n in names:
        if (cap / n).is_file():
            return n
    return None


@app.post("/api/captures/{name}/apply-corrections")
def apply_corrections(name: str, req: ApplyRequest):
    """Fold saved corrections into the labels and rebuild the layer.

    Runs the same scripts the CLI does, as subprocesses, rather than
    reimplementing them here: their guards matter. apply_corrections.py refuses
    a removal polygon that would reject more than a handful of verified arrays,
    and that refusal is the thing standing between a loose rectangle and 11 real
    arrays silently entering the training masks as background.
    """
    import subprocess
    import sys

    cap = _safe_capture_dir(name)
    if not (cap / "corrections.geojson").is_file():
        raise HTTPException(404, "No saved corrections for this capture.")

    labels = req.labels or _first_existing(
        cap, "labels_corrected.json", "labels_reviewed.json",
        "labels_clean.json", "labels.json")
    if not labels:
        raise HTTPException(404, "No label file to start from.")
    dets = req.detections or _first_existing(
        cap, "detections_clipped.geojson", "detections_curated.geojson",
        "detections_raw.geojson")
    if not dets:
        raise HTTPException(404, "No detection layer to build the survey from.")

    scripts = REPO_SCRIPTS
    steps = [
        ("folding corrections into the labels", [
            str(scripts / "apply_corrections.py"), "--capture", name,
            "--labels", labels, "--out", "labels_corrected.json"]),
        ("rebuilding the survey layer", [
            str(scripts / "build_survey_layer.py"), "--capture", name,
            "--labels", "labels_corrected.json", "--detections", dets,
            "--out", "survey.geojson"]),
        ("tracing the arrays it missed", [
            str(scripts / "trace_known_arrays.py"), "--capture", name,
            "--labels", "labels_corrected.json", "--survey", "survey.geojson",
            "--checkpoint", req.checkpoint, "--out", "survey_traced.geojson"]),
    ]

    def run(job: Job):
        log: list[str] = []
        for i, (label, argv) in enumerate(steps, start=1):
            _set(job, current=i, total=len(steps), message=label)
            p = subprocess.run([sys.executable, *argv], capture_output=True,
                               text=True, cwd=str(REPO_SCRIPTS.parent))
            log.append(f"$ {' '.join(argv[1:])}\n{p.stdout}{p.stderr}")
            if p.returncode != 0:
                raise RuntimeError(f"{label} failed:\n{p.stdout}{p.stderr}")
        # Publish: the map reads detections.geojson, so the finished survey has
        # to land there. The traced layer is kept alongside it.
        traced = cap / "survey_traced.geojson"
        if traced.is_file():
            (cap / "detections.geojson").write_text(
                traced.read_text(encoding="utf-8"), encoding="utf-8")

        # Mark the corrections as applied. They are NOT deleted: the survey
        # builder needs the removals to keep suppressing their shapes, and
        # apply_corrections.py rebuilds the drawn labels from this file every
        # run, so clearing it would undo the very edits just applied. The flag
        # only stops the UI redrawing them as pending work -- an applied
        # correction is already the blue shape underneath it.
        cpath = cap / "corrections.geojson"
        doc = json.loads(cpath.read_text(encoding="utf-8"))
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        for f in doc.get("features", []):
            f.setdefault("properties", {})["applied"] = stamp
        doc.setdefault("properties", {})["applied_at"] = stamp
        tmp = cpath.with_suffix(".geojson.tmp")
        tmp.write_text(json.dumps(doc, indent=1), encoding="utf-8")
        tmp.replace(cpath)
        gj = json.loads((cap / "detections.geojson").read_text(encoding="utf-8"))
        return {"name": name, "features": len(gj["features"]),
                "log": "\n".join(log)[-4000:], **gj.get("properties", {})}

    return asdict(_spawn("detect", run))


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "No such job")
    with _jobs_lock:
        return asdict(job)


# --------------------------------------------------------------------------


def _safe_name(name: str) -> str:
    cleaned = "".join(c for c in name if c.isalnum() or c in "-_").strip("-_")
    if not cleaned:
        raise HTTPException(422, "Name must contain alphanumeric characters.")
    return cleaned


def _safe_capture_dir(name: str) -> Path:
    """Resolve a capture directory, refusing anything outside CAPTURES."""
    path = (CAPTURES / _safe_name(name)).resolve()
    if not path.is_relative_to(CAPTURES.resolve()):
        raise HTTPException(400, "Bad capture name")
    return path


class NoCacheStatic(StaticFiles):
    """Serve the UI with caching off.

    index.html carries the whole client: the editor, the correction logic, the
    styling rules. A cached copy means a fix lands on disk, the server serves
    it, and the browser keeps running yesterday's code -- which looks exactly
    like the fix not working, and cost several rounds of debugging something
    that was already correct.
    """

    async def get_response(self, path: str, scope):
        resp = await super().get_response(path, scope)
        if path.endswith((".html", ".js", ".css")) or path in ("", "."):
            resp.headers["Cache-Control"] = "no-store, must-revalidate"
            resp.headers["Pragma"] = "no-cache"
            resp.headers["Expires"] = "0"
        return resp


app.mount("/", NoCacheStatic(directory=WEB_DIR, html=True), name="web")
