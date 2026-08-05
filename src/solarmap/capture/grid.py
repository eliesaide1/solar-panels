"""Turn an area of interest into a grid of camera positions.

Rather than hard-coding Google Earth's field of view (it varies with window
aspect ratio and Earth build), we *probe*: fly to the centre of the AOI at the
target altitude, ask for the resulting view extents, and use the measured
footprint to lay out the grid. That makes the spacing self-calibrating.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BBox:
    """Area of interest in WGS84 degrees."""

    south: float
    west: float
    north: float
    east: float

    def __post_init__(self) -> None:
        if self.north <= self.south or self.east <= self.west:
            raise ValueError(
                f"Degenerate bbox: south={self.south} west={self.west} "
                f"north={self.north} east={self.east}. Expected "
                "south < north and west < east."
            )

    @property
    def center(self) -> tuple[float, float]:
        return ((self.south + self.north) / 2.0, (self.west + self.east) / 2.0)

    @classmethod
    def parse(cls, text: str) -> "BBox":
        """Parse ``south,west,north,east``."""
        parts = [float(p) for p in text.split(",")]
        if len(parts) != 4:
            raise ValueError("Expected 4 comma-separated values: south,west,north,east")
        return cls(*parts)


@dataclass(frozen=True)
class CameraPoint:
    row: int
    col: int
    lat: float
    lon: float

    @property
    def tile_id(self) -> str:
        return f"r{self.row:04d}c{self.col:04d}"


def build_grid(
    aoi: BBox,
    footprint_deg: tuple[float, float],
    overlap: float,
) -> list[CameraPoint]:
    """Lay out camera centres covering ``aoi``.

    ``footprint_deg`` is the measured (height, width) of a single view, in
    degrees. ``overlap`` is the fraction of each view shared with its
    neighbour, which absorbs edge distortion and stops a panel that straddles
    a boundary from being split.
    """
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")

    fh, fw = footprint_deg
    if fh <= 0 or fw <= 0:
        raise ValueError(f"Non-positive view footprint: {footprint_deg}")

    step_lat = fh * (1.0 - overlap)
    step_lon = fw * (1.0 - overlap)

    points: list[CameraPoint] = []

    # Start half a footprint inside the AOI so the first view's *edge*, not its
    # centre, sits on the boundary -- otherwise the outer half-tile is missed.
    lat = aoi.south + fh / 2.0
    row = 0
    while True:
        lon = aoi.west + fw / 2.0
        col = 0
        while True:
            points.append(CameraPoint(row=row, col=col, lat=lat, lon=lon))
            if lon + fw / 2.0 >= aoi.east:
                break
            lon += step_lon
            col += 1
        if lat + fh / 2.0 >= aoi.north:
            break
        lat += step_lat
        row += 1

    return points
