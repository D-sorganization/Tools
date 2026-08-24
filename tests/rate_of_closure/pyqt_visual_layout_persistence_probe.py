"""DPI-isolated restart evidence for persisted PyQt visual layout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
from PyQt6.QtCore import QSettings
from PyQt6.QtGui import QFont, QFontDatabase, QFontMetrics, QImage
from PyQt6.QtWidgets import QApplication

from rate_of_closure.club_camera import ClubCamera
from rate_of_closure.ui.pyqt6.main_window import RateOfClosureMainWindow


def _install_evidence_font(application: QApplication) -> dict[str, object]:
    font_path = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
    font_id = QFontDatabase.addApplicationFont(str(font_path))
    families = QFontDatabase.applicationFontFamilies(font_id)
    if font_id < 0 or not families:
        raise RuntimeError("could not load the bundled layout evidence font")
    font = QFont(families[0])
    metrics = QFontMetrics(font)
    if not all(metrics.inFontUcs4(ord(char)) for char in "Layout 0123 px"):
        raise RuntimeError("layout evidence font lacks required ASCII glyphs")
    application.setFont(font)
    return {
        "font_family": families[0],
        "font_ascii_supported": True,
    }


def _capture(
    output: Path,
    scale: float,
    name: str,
    window: RateOfClosureMainWindow,
) -> tuple[dict[str, object], np.ndarray]:
    app = QApplication.instance()
    assert app is not None
    app.processEvents()
    window_image = window.grab()
    window_path = output / f"layout-{name}-scale-{scale:g}.png"
    window_image.save(str(window_path))
    canvas_image = window._club_view._canvas.grab()
    canvas_path = output / f"layout-canvas-{name}-scale-{scale:g}.png"
    canvas_image.save(str(canvas_path))
    raw_image = canvas_image.toImage().convertToFormat(QImage.Format.Format_RGBA8888)
    pixel_bytes = raw_image.bits().asstring(raw_image.sizeInBytes())
    sizes = window._shell_splitter.sizes()
    rect = window._club_view._canvas.geometry()
    camera = window._club_view.camera()
    pixels = np.frombuffer(pixel_bytes, dtype=np.uint8).reshape(
        raw_image.height(), raw_image.width(), 4
    )
    return {
        "artifact_policy": "diagnostic-only; not an approved golden",
        "state": name,
        "bytes": window_path.stat().st_size,
        "canvas_bytes": canvas_path.stat().st_size,
        "canvas_sha256": hashlib.sha256(canvas_path.read_bytes()).hexdigest(),
        "canvas_pixel_sha256": hashlib.sha256(pixel_bytes).hexdigest(),
        "canvas": [rect.x(), rect.y(), rect.width(), rect.height()],
        "camera": [camera.azimuth_deg, camera.elevation_deg, camera.zoom],
        "status": window._club_view._status.text(),
        "splitter_sizes": sizes,
        "sidebar_fraction": sizes[0] / sum(sizes),
        "tab_width": window._tabs.width(),
        "device_pixel_ratio": window.devicePixelRatioF(),
        "window": [window.width(), window.height()],
    }, pixels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scale", type=float, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    settings_path = args.output / "layout.ini"
    if settings_path.exists():
        settings_path.unlink()
    app = QApplication.instance() or QApplication([])
    font = _install_evidence_font(app)
    settings = QSettings(str(settings_path), QSettings.Format.IniFormat)

    first = RateOfClosureMainWindow(navigation_settings=settings)
    first.resize(1440, 900)
    first.show()
    app.processEvents()
    first._club_view.stop()
    first._club_view.set_camera(ClubCamera(-40.0, 35.0, 2.25))
    first._shell_splitter.moveSplitter(390, 1)
    app.processEvents()
    before, before_pixels = _capture(args.output, args.scale, "before-restart", first)
    settings.sync()
    first.close()
    first.deleteLater()
    app.processEvents()

    second = RateOfClosureMainWindow(
        navigation_settings=QSettings(str(settings_path), QSettings.Format.IniFormat)
    )
    second.resize(1440, 900)
    second.show()
    app.processEvents()
    second._club_view.stop()
    app.processEvents()
    restored, restored_pixels = _capture(args.output, args.scale, "restored", second)
    difference = np.abs(
        before_pixels.astype(np.int16) - restored_pixels.astype(np.int16)
    )
    comparison = {
        "mean_absolute_channel_delta": float(difference.mean() / 255.0),
        "changed_pixel_fraction": float(np.mean(np.any(difference != 0, axis=2))),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(
            {
                "requested_scale": args.scale,
                "font": font,
                "states": [before, restored],
                "pixel_comparison": comparison,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    second.close()


if __name__ == "__main__":
    main()
