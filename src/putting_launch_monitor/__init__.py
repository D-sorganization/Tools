"""Camera-based putting launch monitor for GSPro (epic #5218).

An overhead camera watches the putting surface; a putt's launch speed and
horizontal launch angle are measured on the ground plane and sent to GSPro
over its Open Connect v1 API, so putting happens inside the same round as
full shots from a launch monitor that cannot see putts.

Modules, in pipeline order: :mod:`geometry` (mat-corner homography, launch
fit, HLA), :mod:`detect` (the ball in a frame), :mod:`track` (resting ball
to finished putt), :mod:`gspro` (the wire protocol and client),
:mod:`calibration` (the one document that describes a setup),
:mod:`monitor` (frames in, putts out), :mod:`cli`.
"""

from __future__ import annotations

__version__ = "0.1.0"
