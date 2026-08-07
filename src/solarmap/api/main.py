"""FastAPI backend for the SolarMap UI.

Capture and detection are long-running, so they are dispatched to background
threads and the UI polls ``/api/jobs/{id}``.

Only one capture may run at a time: Google Earth Pro is a single GUI instance
and two threads driving the same camera would interleave into garbage.
"""

from __future__ import annotations

import json
import threading
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


app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
