from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from rate_of_closure.mesh import write_binary_stl
from rate_of_closure.scripts.generate_example_head import build_example_head
from rate_of_closure.ui.pyqt6.main_window import RateOfClosureMainWindow


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scale", type=float, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance() or QApplication([])
    window = RateOfClosureMainWindow()
    window.resize(1440, 900)
    window.show()
    club_index = next(
        index
        for index in range(window._tabs.count())
        if window._tabs.tabText(index) == "3D Clubhead"
    )
    window._tabs.setCurrentIndex(club_index)
    view = window._club_view
    view.stop()
    view.clear_mesh()
    app.processEvents()
    states: list[dict[str, object]] = []

    def capture(name: str) -> None:
        app.processEvents()
        canvas_image = view._canvas.grab()
        canvas_path = args.output / f"club-canvas-{name}-scale-{args.scale:g}.png"
        canvas_image.save(str(canvas_path))
        canvas_digest = hashlib.sha256(canvas_path.read_bytes()).hexdigest()
        image = window.grab()
        path = args.output / f"club-{name}-scale-{args.scale:g}.png"
        image.save(str(path))
        rect = view._canvas.geometry()
        states.append(
            {
                "artifact_policy": "diagnostic-only; not an approved golden",
                "state": name,
                "bytes": path.stat().st_size,
                "canvas_sha256": canvas_digest,
                "canvas": [rect.x(), rect.y(), rect.width(), rect.height()],
                "status": view._status.text(),
                "error": view._error.text(),
                "camera": [
                    view._camera.azimuth_deg,
                    view._camera.elevation_deg,
                    view._camera.zoom,
                ],
                "focus": view._canvas.hasFocus(),
                "source": {
                    "kind": view._source.kind,
                    "generation": view._source.generation,
                    "sha256": view._source.sha256,
                },
                "device_pixel_ratio": window.devicePixelRatioF(),
                "window": [window.width(), window.height()],
            }
        )

    view._canvas.setFocus()
    capture("procedural")
    QTest.keyClick(view._canvas, Qt.Key.Key_Left)
    mesh_path = args.output / "bounded-head.stl"
    mesh_path.write_bytes(write_binary_stl(build_example_head()))
    view.load_mesh(str(mesh_path))
    capture("imported-selected-camera")
    imported_canvas_digest = states[-1]["canvas_sha256"]
    malformed = args.output / "malformed.stl"
    malformed.write_text("solid broken\nfacet nope\nendsolid broken", encoding="utf-8")
    assert not view.try_load_mesh(str(malformed))
    capture("error-prior")
    assert states[-1]["canvas_sha256"] == imported_canvas_digest
    (args.output / "manifest.json").write_text(
        json.dumps({"requested_scale": args.scale, "states": states}, indent=2),
        encoding="utf-8",
    )
    window.close()


if __name__ == "__main__":
    main()
