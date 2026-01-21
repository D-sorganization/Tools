# Scientific Modeling

Advanced simulations and modeling tools for scientific computing, path planning, and 3D visualization.

## Overview

This directory contains scientific modeling applications implemented in both MATLAB and Python, providing comprehensive capabilities for research, simulation, and visualization.

## Components

### [RRT Path Planner](rrt_path_planner/README.md)

A Star Wars-themed path planning simulator featuring:

- **RRT Algorithm**: Rapidly-exploring Random Trees for path planning
- **3D Environment**: Dynamic obstacle avoidance in 3D space
- **Dual Implementation**: MATLAB (research-focused) and Python (performance-optimized)
- **Cinematic Visualization**: Star Wars-inspired graphics and camera views
- **AI Pursuit System**: Intelligent pursuit scenarios with dynamic replanning

### [Solar System Model](solar_system_model/)

Interactive 3D visualization of the solar system:

- **Planet Simulation**: Accurate orbital mechanics
- **3D Rendering**: Visual representation of planetary bodies
- **Python Implementation**: Built with Python visualization libraries

## Quick Start

### RRT Path Planner

**MATLAB:**
```matlab
cd scientific_modeling/rrt_path_planner/matlab/src
main_improved
```

**Python:**
```bash
cd scientific_modeling/rrt_path_planner/python/src
pip install -r requirements.txt
python star_wars_rrt.py
```

### Solar System Model

```bash
cd scientific_modeling/solar_system_model
python run_solar_system.py
```

## Requirements

- **MATLAB**: R2020a or later (for MATLAB implementations)
- **Python**: 3.11+ with NumPy, SciPy, and visualization libraries
- **GPU**: Optional CUDA/OpenCL support for Python RRT planner

## Integration

This suite integrates with:
- **Data Processing** (`data_processing/`) - Data analysis pipelines
- **MATLAB Core** (`matlab/`) - Shared MATLAB infrastructure
- **Web Applications** (`web_applications/`) - Shared math utilities

## License

Part of the Tools repository. See main repository license for details.
