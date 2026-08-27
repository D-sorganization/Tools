#!/usr/bin/env python3
"""Build the deterministic renderer-pathway figure for the Tools manual."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BACKGROUND = "#F7F9FC"
BLUE = "#2E74B5"
BLUE_DARK = "#1F4D78"
BODY = "#20252B"
MUTED = "#5B6573"


def _font(
    size: int, *, bold: bool = False
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = [
        "arialbd.ttf" if bold else "arial.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _box(
    draw: ImageDraw.ImageDraw, bounds: tuple[int, int, int, int], text: str
) -> None:
    draw.rounded_rectangle(bounds, radius=14, fill="white", outline=BLUE, width=3)
    left, top, right, bottom = bounds
    box = draw.textbbox((0, 0), text, font=_font(27, bold=True))
    width = box[2] - box[0]
    height = box[3] - box[1]
    draw.text(
        ((left + right - width) / 2, (top + bottom - height) / 2 - 4),
        text,
        fill=BLUE_DARK,
        font=_font(27, bold=True),
    )


def build_figure(path: Path) -> None:
    """Draw a readable fixed-size workflow diagram without external assets."""
    image = Image.new("RGB", (1600, 780), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.text(
        (80, 52),
        "One source, four representations, one semantic contract",
        fill=BODY,
        font=_font(42, bold=True),
    )
    draw.text(
        (80, 112),
        "Every byte is digest-bound; approval remains separate.",
        fill=MUTED,
        font=_font(26),
    )
    _box(draw, (560, 205, 1040, 335), "Canonical QMD")
    outputs = [
        (90, 500, 410, 625, "HTML"),
        (455, 500, 775, 625, "LaTeX"),
        (820, 500, 1140, 625, "PDF"),
        (1185, 500, 1505, 625, "DOCX"),
    ]
    for left, top, right, bottom, label in outputs:
        center = (left + right) // 2
        draw.line((800, 335, center, top), fill=BLUE, width=5)
        _box(draw, (left, top, right, bottom), label)
    draw.line((250, 680, 1350, 680), fill=BLUE_DARK, width=5)
    for left, _, right, bottom, _ in outputs:
        center = (left + right) // 2
        draw.line((center, bottom, center, 680), fill=BLUE_DARK, width=4)
    label = "Shared semantic SHA-256"
    bounds = draw.textbbox((0, 0), label, font=_font(29, bold=True))
    draw.rectangle((585, 654, 1015, 715), fill=BACKGROUND)
    draw.text(
        (800 - (bounds[2] - bounds[0]) / 2, 665),
        label,
        fill=BLUE_DARK,
        font=_font(29, bold=True),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=False, compress_level=9)


def main(argv: Sequence[str] | None = None) -> int:
    """Build the default figure path or a caller-selected output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    output = (
        args.output or root / "manuals" / "tools" / "figures" / "render-pipeline.png"
    )
    build_figure(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
