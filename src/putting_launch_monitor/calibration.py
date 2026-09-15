"""The one file that makes a camera a putting monitor.

:class:`Calibration` holds everything measured once for a given camera
placement: the hitting mat's corners in the image and its real size (the
ground-plane homography), the target direction, the ball colour, the
region of interest and the stream settings. It is a versioned, fail-closed
JSON document (``putting_monitor.calibration/1``): a reader either
understands exactly this format or refuses, following the repository's
wire-record idiom.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from shared.python.contracts import require

from .detect import COLOUR_PROFILES, HsvBallDetector, HsvRange, RegionOfInterest
from .geometry import GroundPlane
from .track import TrackerSettings

SCHEMA = "putting_monitor.calibration/1"
APP_DIR_NAME = "putting_launch_monitor"
ORG_NAME = "D-sorganization"


def default_calibration_path() -> Path:
    """Where the calibration lives unless told otherwise (per user, per OS)."""
    import platformdirs

    return (
        Path(platformdirs.user_config_dir(APP_DIR_NAME, ORG_NAME)) / "calibration.json"
    )


CORNER_ORDER = ("near-left", "near-right", "far-right", "far-left")
ROI_MARGIN = 0.25  # of the mat's bounding box, each side


@dataclass(frozen=True)
class Calibration:
    """Camera placement, mat geometry and detection settings for one setup.

    ``mat_corners_px`` follow :data:`CORNER_ORDER`, judged from where the
    player stands looking at the target. ``target_deg`` rotates the target
    line from the mat's long axis (positive = toward the player's right),
    for a hole that is not straight down the mat.
    Invariants: four corners; positive mat size; ``fps > 0``.
    """

    camera_instance_id: str
    mat_corners_px: tuple[tuple[float, float], ...]
    mat_width_mm: float
    mat_length_mm: float
    target_deg: float = 0.0
    colour: str = "white"
    custom_colour: HsvRange | None = None
    roi: RegionOfInterest | None = None
    fps: int = 60
    capture_width: int = 1920
    capture_height: int = 1200
    tracker: TrackerSettings = field(default_factory=TrackerSettings)
    notes: str = ""

    def __post_init__(self) -> None:
        require(bool(self.camera_instance_id), "camera instance id")
        require(
            len(self.mat_corners_px) == 4, "four mat corners", len(self.mat_corners_px)
        )
        require(self.mat_width_mm > 0 and self.mat_length_mm > 0, "mat size", self)
        require(self.fps > 0, "fps", self.fps)
        require(abs(self.target_deg) < 90, "target_deg", self.target_deg)
        require(
            self.colour in COLOUR_PROFILES or self.custom_colour is not None,
            "unknown colour profile",
            self.colour,
        )

    # -- derived objects ------------------------------------------------------------
    def ground_plane(self) -> GroundPlane:
        return GroundPlane.from_rectangle(
            np.asarray(self.mat_corners_px, dtype=np.float64),
            self.mat_width_mm,
            self.mat_length_mm,
        )

    def target_vector(self) -> tuple[float, float]:
        """Unit target direction in the world frame (x right, y toward target)."""
        a = np.radians(self.target_deg)
        return (float(np.sin(a)), float(np.cos(a)))

    def hsv(self) -> HsvRange:
        return self.custom_colour or COLOUR_PROFILES[self.colour]

    def search_region(self) -> RegionOfInterest:
        """Where the ball is looked for: ``roi`` if set, else the mat plus a margin.

        On the lab rig a whole-frame search returned 59 ball-like blobs — book
        spines, bright fittings — of which three were balls. Putts happen on
        the mat, so by default the search is the mat's bounding box grown by
        :data:`ROI_MARGIN` on every side (a putt may start just off the edge).
        """
        if self.roi is not None:
            return self.roi
        pts = np.asarray(self.mat_corners_px, dtype=np.float64)
        (x0, y0), (x1, y1) = pts.min(axis=0), pts.max(axis=0)
        mx, my = (x1 - x0) * ROI_MARGIN, (y1 - y0) * ROI_MARGIN
        return RegionOfInterest(
            max(0, int(x0 - mx)),
            max(0, int(y0 - my)),
            min(self.capture_width, int(x1 + mx) + 1),
            min(self.capture_height, int(y1 + my) + 1),
        )

    def detector(self) -> HsvBallDetector:
        """A detector whose radius bounds follow the mat's scale in the image
        and whose search is confined to :meth:`search_region`."""
        plane = self.ground_plane()
        centre = np.mean(np.asarray(self.mat_corners_px, dtype=np.float64), axis=0)
        mm_per_px = plane.mm_per_px_at(centre)
        ball_px = 42.67 / 2.0 / max(mm_per_px, 1e-6)
        return HsvBallDetector(
            colour=self.hsv(),
            min_radius_px=max(2.0, ball_px * 0.4),
            max_radius_px=ball_px * 2.5,
            roi=self.search_region(),
        )

    def tracker_settings(self) -> TrackerSettings:
        base = asdict(self.tracker)
        base["target"] = self.target_vector()
        return TrackerSettings(**base)

    # -- persistence ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "schema": SCHEMA,
            "camera_instance_id": self.camera_instance_id,
            "mat_corners_px": [list(map(float, c)) for c in self.mat_corners_px],
            "mat_width_mm": self.mat_width_mm,
            "mat_length_mm": self.mat_length_mm,
            "target_deg": self.target_deg,
            "colour": self.colour,
            "custom_colour": asdict(self.custom_colour) if self.custom_colour else None,
            "roi": asdict(self.roi) if self.roi else None,
            "fps": self.fps,
            "capture_width": self.capture_width,
            "capture_height": self.capture_height,
            "tracker": {k: v for k, v in asdict(self.tracker).items() if k != "target"},
            "notes": self.notes,
        }
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Calibration:
        """Precondition: the document declares :data:`SCHEMA` and no unknown keys."""
        require(isinstance(data, dict), "calibration must be an object")
        require(
            data.get("schema") == SCHEMA,
            "unsupported calibration schema",
            data.get("schema"),
        )
        known = set(cls.__dataclass_fields__) | {"schema"}
        unknown = set(data) - known
        require(not unknown, "unknown calibration fields", sorted(unknown))
        custom = data.get("custom_colour")
        roi = data.get("roi")
        tracker = data.get("tracker") or {}
        return cls(
            camera_instance_id=str(data["camera_instance_id"]),
            mat_corners_px=tuple(
                (float(c[0]), float(c[1])) for c in data["mat_corners_px"]
            ),
            mat_width_mm=float(data["mat_width_mm"]),
            mat_length_mm=float(data["mat_length_mm"]),
            target_deg=float(data.get("target_deg", 0.0)),
            colour=str(data.get("colour", "white")),
            custom_colour=HsvRange(**custom) if custom else None,
            roi=RegionOfInterest(**roi) if roi else None,
            fps=int(data.get("fps", 60)),
            capture_width=int(data.get("capture_width", 1920)),
            capture_height=int(data.get("capture_height", 1200)),
            tracker=TrackerSettings(**tracker),
            notes=str(data.get("notes", "")),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> Calibration:
        require(path.is_file(), "calibration file must exist", str(path))
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
