"""Thin wrapper over the Google Earth Pro COM automation API (Windows only).

Google Earth Pro registers the ProgID ``GoogleEarth.ApplicationGE``. The three
calls that matter for imagery capture are:

    SetCameraParams(lat, lon, alt, altMode, range, tilt, azimuth, speed)
    GetViewExtents()  -> object with .North .South .East .West
    SaveScreenShot(absolute_path_to_jpg, quality)

``GetViewExtents`` is what makes the captures useful as GIS data rather than
just pictures: it reports the geographic bounds of whatever is currently on
screen, which we turn into a world file.

Note that ``SaveScreenShot`` grabs the 3D render window at its on-screen
resolution -- it is not the high-resolution "File > Save Image" export. Ground
sample distance is therefore set by your window size and the eye altitude.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

# AltitudeModeGE enum from the Earth COM API.
ALT_RELATIVE_TO_GROUND = 1
ALT_ABSOLUTE = 2

# SetCameraParams speed: 5.0 means "teleport", i.e. jump without animating.
# Animating between grid cells wastes many seconds per tile and buys nothing.
SPEED_TELEPORT = 5.0

PROGID = "GoogleEarth.ApplicationGE"


class EarthNotRunning(RuntimeError):
    pass


class StreamingTimeout(RuntimeError):
    pass


@dataclass(frozen=True)
class ViewExtents:
    """Geographic bounds of the current view, in WGS84 degrees."""

    north: float
    south: float
    east: float
    west: float

    @property
    def height_deg(self) -> float:
        return self.north - self.south

    @property
    def width_deg(self) -> float:
        return self.east - self.west


class GoogleEarth:
    """Drives a running Google Earth Pro instance."""

    def __init__(self, stream_timeout: float = 45.0, settle: float = 1.5):
        self.stream_timeout = stream_timeout
        self.settle = settle
        self._app = None

    # -- lifecycle ---------------------------------------------------------

    def connect(self) -> "GoogleEarth":
        try:
            import pythoncom  # noqa: F401
            import win32com.client
        except ImportError as exc:  # pragma: no cover - platform dependent
            raise EarthNotRunning(
                "pywin32 is required to drive Google Earth Pro. "
                "Install it with: pip install pywin32"
            ) from exc

        try:
            # Dispatch attaches to the running instance if there is one, and
            # launches Earth if there is not.
            self._app = win32com.client.Dispatch(PROGID)
        except Exception as exc:  # pragma: no cover - platform dependent
            raise EarthNotRunning(
                "Could not reach Google Earth Pro over COM. Check that it is "
                "installed (earth.google.com/versions) and has been launched "
                "at least once so it registers its COM server."
            ) from exc

        self._wait_for_init()
        return self

    def _wait_for_init(self) -> None:
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            try:
                if self._app.IsInitialized():
                    return
            except Exception:
                # Earth is still starting up and not yet answering COM calls.
                pass
            time.sleep(1.0)
        raise EarthNotRunning("Google Earth Pro did not finish initialising.")

    @property
    def app(self):
        if self._app is None:
            raise EarthNotRunning("Call connect() first.")
        return self._app

    # -- layers ------------------------------------------------------------

    def hide_layers(self, names: list[str]) -> list[str]:
        """Hide the named layer groups. Returns the names actually matched.

        Roads, borders, labels and Photos icons are drawn into the render
        window, so if they are left on they end up baked into the captured
        imagery and the model learns to react to them.
        """
        hidden: list[str] = []
        wanted = {n.strip().lower() for n in names}
        try:
            collection = self.app.GetLayersDatabases()
            for i in range(collection.Count):
                db = collection.Item(i + 1)
                for feature in _iter_features(db):
                    if feature.Name.strip().lower() in wanted:
                        feature.Visibility = False
                        hidden.append(feature.Name)
        except Exception:
            # Layer tree layout varies between Earth builds; failing to hide
            # layers degrades imagery quality but should not abort a capture.
            pass
        return hidden

    # -- camera ------------------------------------------------------------

    def goto(self, lat: float, lon: float, eye_altitude_m: float) -> None:
        """Move to a nadir (straight-down) view centred on lat/lon.

        ``range`` is the camera-to-target distance; with tilt=0 that is the
        eye altitude. ``alt``/``altMode`` place the *target* on the ground.
        """
        self.app.SetCameraParams(
            lat,
            lon,
            0.0,                      # target altitude
            ALT_RELATIVE_TO_GROUND,   # ...measured from the terrain
            float(eye_altitude_m),    # range
            0.0,                      # tilt: 0 = straight down
            0.0,                      # azimuth: 0 = north up
            SPEED_TELEPORT,
        )

    def wait_for_imagery(self) -> None:
        """Block until Earth has finished streaming tiles for this view."""
        deadline = time.monotonic() + self.stream_timeout
        while time.monotonic() < deadline:
            try:
                if self.app.GetStreamingProgressPercentage() >= 100:
                    # Earth reports 100% a beat before the highest-resolution
                    # texture is actually resolved on screen.
                    time.sleep(self.settle)
                    return
            except Exception:
                pass
            time.sleep(0.25)
        raise StreamingTimeout(
            f"Imagery did not finish streaming within {self.stream_timeout:.0f}s."
        )

    def view_extents(self) -> ViewExtents:
        e = self.app.GetViewExtents()
        return ViewExtents(
            north=float(e.North),
            south=float(e.South),
            east=float(e.East),
            west=float(e.West),
        )

    def screenshot(self, path: str | Path, quality: int = 95) -> Path:
        path = Path(path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        # The COM API writes JPEG and expects a native absolute path.
        self.app.SaveScreenShot(str(path), int(quality))
        if not path.exists():
            raise RuntimeError(f"Google Earth did not write a screenshot to {path}")
        return path


def _iter_features(node):
    """Depth-first walk of Earth's layer tree."""
    try:
        children = node.GetChildren()
    except Exception:
        return
    for i in range(children.Count):
        child = children.Item(i + 1)
        yield child
        yield from _iter_features(child)
