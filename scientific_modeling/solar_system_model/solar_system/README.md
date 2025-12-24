# Solar System Simulation

A professional-grade, scientifically accurate 3D simulation of our solar system built in Python. This educational tool provides real-time visualization of planetary orbits, interplanetary trajectory planning, and multiple camera perspectives for exploring the cosmos.

## Features

### Scientific Accuracy
- **Keplerian Orbital Mechanics**: Accurate planetary positions calculated using orbital elements from NASA JPL
- **Real Ephemeris Data**: Orbital elements valid for 1800-2050 AD with secular variations
- **Physical Properties**: Accurate masses, radii, densities, and other physical characteristics
- **Time Simulation**: Real-time or accelerated simulation with proper orbital mechanics

### Visualization
- **3D OpenGL Rendering**: Smooth, hardware-accelerated graphics
- **Star Field Background**: Accurate sky dome built from bright-star catalogs for constellation fidelity
- **Planetary Rings**: Saturn and other gas giants rendered with ring systems
- **Orbital Paths**: Visualize complete orbital trajectories
- **Color-Coded Bodies**: Each planet has accurate representative colors

### Camera System
- **Free Camera**: Full control over position and orientation
- **Heliocentric View**: Fixed at the Sun, looking outward
- **Planet-Centric**: Follow any planet through its orbit
- **Top-Down View**: Bird's eye view of the solar system
- **Spacecraft Following**: Track spacecraft along trajectories

### Trajectory Planning
- **Hohmann Transfers**: Calculate optimal two-impulse transfers between planets
- **Delta-V Calculations**: Accurate fuel requirements for missions
- **Launch Windows**: Find optimal departure dates
- **Transfer Visualization**: See spacecraft trajectories in real-time

### Educational Features
- **Information Panels**: Detailed data about selected celestial bodies
- **Real-Time Data**: Current distance, orbital speed, and position
- **Time Control**: Speed up, slow down, or reverse time
- **Interactive Selection**: Click to select and learn about any body
- **Immersion Checklist**: Guided set of goals to explore missions, data overlays, and time travel

## Installation

### Prerequisites
- Python 3.8 or higher
- OpenGL-capable graphics hardware

### Install Dependencies

```bash
cd solar_system
pip install -r requirements.txt
```

Or install manually:

```bash
pip install numpy pygame PyOpenGL PyOpenGL_accelerate
```

## Usage

### Basic Usage

```bash
# From the Playground directory
python run_solar_system.py
```

For a one-click start with built-in dependency checks, double-click the
top-level `launch_solar_system.py` script (or run
`python launch_solar_system.py`). It launches a windowed view at 1280x720 and
prints a clear message if PyGame or PyOpenGL still need to be installed.

Or as a module:

```bash
python -m solar_system.main
```

### Command Line Options

```bash
python run_solar_system.py --help

Options:
  --fullscreen      Start in fullscreen mode
  --width W         Window width (default: 1600)
  --height H        Window height (default: 900)
  --no-vsync        Disable vertical sync
  --start-date      Start date in YYYY-MM-DD format
  --no-antialiasing Disable antialiasing
```

### Examples

```bash
# Start in fullscreen
python run_solar_system.py --fullscreen

# Start at a specific date
python run_solar_system.py --start-date 2025-07-04

# Custom window size
python run_solar_system.py --width 1920 --height 1080
```

## Controls

### Keyboard

| Key | Action |
|-----|--------|
| `SPACE` | Pause/Resume simulation |
| `+` / `-` | Speed up / slow down time |
| `R` | Reverse time flow |
| `0-9` | Select celestial body (0=Sun, 1=Mercury, ..., 9=Pluto) |
| `F` | Focus camera on selected body |
| `C` | Cycle through camera modes |
| `O` | Toggle orbital path display |
| `L` | Toggle labels |
| `I` | Toggle information panel |
| `G` | Toggle reference grid |
| `H` | Toggle help overlay |
| `M` | Toggle immersion checklist |
| `V` | Toggle stereo/VR rendering |
| `T` | Plan trajectory to Mars |
| `HOME` | Reset camera view |
| `ESC` | Quit |

### Mouse

| Action | Effect |
|--------|--------|
| Left Drag | Orbit camera around target |
| Right Drag | Pan camera |
| Scroll Wheel | Zoom in/out |

## Architecture

```
solar_system/
├── __init__.py          # Package initialization
├── main.py              # Application entry point
├── requirements.txt     # Dependencies
│
├── core/                # Core simulation components
│   ├── constants.py     # Astronomical constants and data
│   ├── celestial_body.py # Body classes (Star, Planet, Moon)
│   └── time_manager.py  # Simulation time management
│
├── physics/             # Orbital mechanics
│   ├── orbital_mechanics.py  # Kepler's laws, vis-viva
│   └── trajectory_planner.py # Transfer orbit calculations
│
├── visualization/       # Rendering system
│   ├── renderer.py      # OpenGL renderer
│   ├── camera.py        # Camera system
│   └── scene.py         # Scene management
│
├── ui/                  # User interface
│   ├── controls.py      # Input handling
│   └── widgets.py       # UI components
│
├── data/                # Additional data files
└── assets/              # Textures and shaders
    ├── textures/
    └── shaders/
```

## Scientific Data Sources

- **Orbital Elements**: NASA JPL Keplerian Elements for Approximate Positions
- **Physical Properties**: NASA Planetary Fact Sheets
- **Gravitational Parameters**: IAU 2015 Resolutions
- **Time Standards**: Julian Date system, J2000.0 epoch

## Orbital Mechanics

The simulation uses Keplerian orbital mechanics with the following elements:
- Semi-major axis (a)
- Eccentricity (e)
- Inclination (i)
- Longitude of ascending node (Ω)
- Longitude of perihelion (ϖ)
- Mean longitude (L)

Positions are calculated by:
1. Computing mean anomaly from mean longitude
2. Solving Kepler's equation for eccentric anomaly
3. Converting to true anomaly
4. Transforming to heliocentric ecliptic coordinates

## API Usage

You can also use the library programmatically:

```python
from solar_system.core import Planet, Star, TimeManager
from solar_system.physics import TrajectoryPlanner, OrbitalMechanics

# Create solar system
sun = Star("Sun")
earth = Planet("Earth", parent=sun)
mars = Planet("Mars", parent=sun)

# Get planet positions
time_mgr = TimeManager()
earth_state = earth.get_state_at_time(time_mgr.julian_date)
print(f"Earth position: {earth_state.position_au} AU")

# Plan a trajectory
planner = TrajectoryPlanner()
transfer = planner.calculate_transfer(earth, mars, time_mgr.julian_date)
print(f"Transfer time: {transfer.time_of_flight:.1f} days")
print(f"Delta-V required: {transfer.total_delta_v:.1f} m/s")
```

## Performance Notes

- The simulation targets 60 FPS on modern hardware
- Orbital calculations are cached to improve performance
- Display lists are used for efficient OpenGL rendering
- Star field is pre-generated for quick rendering

## Known Limitations

- Moon orbits are simplified (no perturbations)
- No atmospheric effects or surface details
- Spacecraft trajectories use patched conics approximation
- Planet sizes are exaggerated for visibility

## Future Enhancements

- [x] Planet textures from NASA imagery
- [x] Shader-based rendering for better visuals
- [x] More detailed moon systems
- [x] Asteroid belt visualization
- [x] Comet trajectories
- [x] Gravity assists in trajectory planning
- [x] VR support

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- NASA JPL for orbital data
- The pygame and PyOpenGL communities
- All contributors to open-source astronomy software
