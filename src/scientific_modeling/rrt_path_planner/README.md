# RRT Asteroid Navigator

A cinematic-but-educational RRT path planning simulator for asteroid-field navigation and pursuit scenarios.

## Overview

This project currently includes:

- MATLAB material for related experiments and notes
- A Python reference implementation in `python/src/star_wars_rrt.py`

The Python app focuses on:

- readable 3D asteroid-field navigation
- RRT path planning plus path smoothing
- route metrics such as efficiency, clearance, and turn angle

## Project Layout

```text
rrt_path_planner/
├── matlab/
├── python/
│   ├── src/star_wars_rrt.py
│   ├── tests/test_star_wars_rrt.py
│   ├── requirements.txt
│   └── setup.py
├── launch_rrt_asteroid_navigator.py
└── README.md
```

## Features

- RRT path planning in 3D asteroid fields
- Path smoothing for cleaner routes
- Science-style route metrics: path length, efficiency, clearance, turn angle
- Pursuit mode with evasive target behavior
- Cinematic, chase, and top-down camera modes
- Optional OpenGL renderer with waypoint markers and starfield

## Quick Start

```bash
pip install -r src/scientific_modeling/rrt_path_planner/python/requirements.txt
python src/scientific_modeling/rrt_path_planner/launch_rrt_asteroid_navigator.py
```

You can also run the core module directly:

```bash
python src/scientific_modeling/rrt_path_planner/python/src/star_wars_rrt.py
```

## Testing

```bash
pytest src/scientific_modeling/rrt_path_planner/python/tests/test_star_wars_rrt.py
```

## Notes

- The planning core is importable without graphics dependencies, so tests can run in headless environments.
- The renderer activates only when `pygame` and `PyOpenGL` are installed.
- The standalone launcher is the easiest entry point for end users.
