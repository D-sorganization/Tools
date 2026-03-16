# Solar System Model

A professional-grade, scientifically accurate 3D interactive solar system simulation built with Python, PyGame, and PyOpenGL. Features real-time orbital mechanics, Keplerian orbit calculations, trajectory planning, and educational information overlays.

## Purpose

The Solar System Model provides an immersive educational and scientific visualization tool for:

- Visualizing planetary orbits and positions in real-time
- Understanding orbital mechanics principles
- Planning interplanetary trajectories
- Exploring historical and future celestial events
- Learning about celestial body properties

## Key Features

- **Keplerian Orbital Mechanics**: Accurate positions based on NASA JPL orbital elements
- **3D OpenGL Rendering**: Hardware-accelerated visualization with starfield background
- **Multiple Camera Modes**: Free camera, heliocentric, planet-centric, top-down views
- **Trajectory Planning**: Hohmann transfer calculations with delta-V requirements
- **Time Navigation**: Jump to any date (1800-2200), forward/reverse time flow
- **Historical Events**: View space exploration milestones at their actual dates
- **Educational Overlays**: Fun facts, physical properties, orbital data for each body
- **Interactive Selection**: Click or key-select any celestial body for detailed info

## Installation

### Prerequisites

- Python 3.8 or higher
- OpenGL-capable graphics hardware
- PyGame
- PyOpenGL
- NumPy

### Install Dependencies

```bash
cd Tools/src/scientific_modeling/solar_system_model/solar_system
pip install -r requirements.txt
```

Or install manually:

```bash
pip install numpy pygame PyOpenGL PyOpenGL_accelerate
```

### From Repository

```bash
cd Tools/src/scientific_modeling/solar_system_model
python launch_solar_system.py
```

## Usage Instructions

### Launching the Application

```bash
# Using launcher (recommended - includes dependency checks)
python launch_solar_system.py

# Using run script
python run_solar_system.py

# As module
python -m solar_system.main
```

### Command Line Options

```bash
python run_solar_system.py --help

Options:
  --fullscreen        Start in fullscreen mode
  --width W           Window width (default: 1600)
  --height H          Window height (default: 900)
  --no-vsync          Disable vertical sync
  --start-date DATE   Start date in YYYY-MM-DD format
  --no-antialiasing   Disable antialiasing
  --no-shaders        Disable shaders
```

### Examples

```bash
# Start in fullscreen
python run_solar_system.py --fullscreen

# Start at Apollo 11 launch date
python run_solar_system.py --start-date 1969-07-16

# Custom window size
python run_solar_system.py --width 1920 --height 1080
```

## Input Parameters

### Keyboard Controls

| Key   | Action                                         |
| ----- | ---------------------------------------------- |
| SPACE | Pause/Resume simulation                        |
| +/-   | Speed up/slow down time                        |
| R     | Reverse time flow                              |
| D     | Toggle date picker                             |
| N     | Toggle time navigation panel                   |
| E     | Toggle historical events panel                 |
| [ / ] | Jump backward/forward 1 day                    |
| { / } | Jump backward/forward 1 month                  |
| 0-9   | Select celestial body (0=Sun, 3=Earth, 4=Mars) |
| F     | Focus on selected body                         |
| C     | Cycle camera mode                              |
| O     | Toggle orbital paths                           |
| L     | Toggle labels                                  |
| I     | Toggle info panel                              |
| G     | Toggle grid                                    |
| M     | Toggle immersion checklist                     |
| H     | Toggle help overlay                            |
| T     | Plan trajectory to Mars                        |
| .     | Cycle fun facts (planet selected)              |
| HOME  | Reset view                                     |
| ESC   | Quit                                           |

### Mouse Controls

| Action       | Effect                     |
| ------------ | -------------------------- |
| Left Drag    | Orbit camera around target |
| Right Drag   | Pan camera                 |
| Scroll Wheel | Zoom in/out                |

## Output Format

### Information Panel Display

```
EARTH
Distance from Sun: 1.00 AU
Orbital Period: 365.25 days
Current Speed: 29.78 km/s
Diameter: 12,742 km
Mass: 5.97 x 10^24 kg
Moons: 1 (Moon)
```

### Trajectory Planning Output

```
EARTH -> MARS TRANSFER
Launch Date: 2026-08-15
Arrival Date: 2027-02-28
Time of Flight: 197 days
Delta-V: 3,600 m/s
Phase Angle: 44.3 degrees
```

## Example Usage

### Basic Exploration

```bash
# Launch simulation
python launch_solar_system.py

# Controls:
# - Scroll wheel to zoom out and see all planets
# - Press H to see help overlay
# - Press 4 to select Mars
# - Press I to see Mars information
```

### Time Travel to Historical Event

```bash
# Launch at Apollo 11 date
python run_solar_system.py --start-date 1969-07-16

# In simulation:
# - Press E to see historical events
# - Press 3 to select Earth
# - See Moon position during Apollo 11
```

### Trajectory Planning

```bash
# Launch simulation
python launch_solar_system.py

# In simulation:
# - Press 3 to select Earth
# - Press T to plan trajectory to Mars
# - View transfer orbit and delta-V requirements
# - Adjust time to see optimal launch windows
```

### Programmatic Usage

```python
from solar_system.core import Planet, Star, TimeManager
from solar_system.physics import TrajectoryPlanner, OrbitalMechanics

# Create celestial bodies
sun = Star("Sun")
earth = Planet("Earth", parent=sun)
mars = Planet("Mars", parent=sun)

# Get planet position at specific time
time_mgr = TimeManager()
earth_state = earth.get_state_at_time(time_mgr.julian_date)
print(f"Earth position: {earth_state.position_au} AU")

# Plan interplanetary transfer
planner = TrajectoryPlanner()
transfer = planner.calculate_transfer(earth, mars, time_mgr.julian_date)
print(f"Transfer time: {transfer.time_of_flight:.1f} days")
print(f"Delta-V required: {transfer.total_delta_v:.1f} m/s")
```

## Troubleshooting

### Application Won't Start

```
ERROR: Failed to initialize. Make sure PyGame and PyOpenGL are installed.
```

**Solution**: Install dependencies:

```bash
pip install pygame PyOpenGL PyOpenGL_accelerate numpy
```

### Black Screen or No Rendering

**Causes**:

- Graphics driver issues
- OpenGL not supported

**Solutions**:

- Update graphics drivers
- Try `--no-shaders` flag
- Try `--no-antialiasing` flag

### Low Frame Rate

**Solutions**:

- Reduce window size: `--width 1280 --height 720`
- Disable antialiasing: `--no-antialiasing`
- Enable VSync: remove `--no-vsync` if present

### Date Picker Not Working

**Issue**: Date navigation unresponsive

**Solution**: Click in the simulation window first to ensure it has focus.

### Planets Not Visible

**Issue**: Only sun visible, planets too small

**Solution**: Use scroll wheel to zoom out significantly. Planets are far from sun at realistic scale.

## Related Tools

- **Multi-Parameter Analysis**: For analyzing orbital parameter sensitivities
- **Optimizer GUI**: For optimizing trajectory parameters
- **Data Processor**: For processing observational data

## Technical Notes

### Orbital Mechanics Implementation

The simulation uses Keplerian orbital elements:

- **Semi-major axis (a)**: Orbit size
- **Eccentricity (e)**: Orbit shape (0=circle, 1=parabola)
- **Inclination (i)**: Tilt relative to ecliptic
- **Longitude of ascending node**: Where orbit crosses ecliptic
- **Argument of perihelion**: Orientation of ellipse
- **Mean anomaly**: Position along orbit

### Position Calculation Process

1. Compute mean anomaly from mean longitude and time
2. Solve Kepler's equation for eccentric anomaly (iterative)
3. Convert eccentric to true anomaly
4. Transform to heliocentric ecliptic coordinates
5. Apply rotation matrices for 3D rendering

### Key Equations

**Vis-viva equation** (orbital velocity):

```
v^2 = GM(2/r - 1/a)
```

**Kepler's equation**:

```
M = E - e*sin(E)
```

**Orbital period**:

```
T = 2*pi*sqrt(a^3 / GM)
```

### Data Sources

- **Orbital Elements**: NASA JPL Keplerian Elements
- **Physical Properties**: NASA Planetary Fact Sheets
- **Gravitational Parameters**: IAU 2015 Resolutions
- **Time Standards**: Julian Date, J2000.0 epoch

### Directory Structure

```
solar_system_model/
├── README.md                    # This file
├── launch_solar_system.py       # Quick launcher with checks
├── run_solar_system.py          # Main runner script
└── solar_system/
    ├── __init__.py
    ├── main.py                  # Entry point
    ├── requirements.txt         # Dependencies
    ├── core/                    # Core components
    │   ├── celestial_body.py   # Body classes
    │   ├── constants.py        # Astronomical constants
    │   └── time_manager.py     # Time management
    ├── physics/                 # Orbital mechanics
    │   ├── orbital_mechanics.py
    │   └── trajectory_planner.py
    ├── visualization/           # Rendering
    │   ├── renderer.py
    │   ├── camera.py
    │   ├── scene.py
    │   └── starfield.py
    ├── ui/                      # User interface
    │   ├── controls.py
    │   └── widgets.py
    └── data/                    # Celestial data
        ├── asteroids.py
        ├── comets.py
        ├── moon_systems.py
        └── planet_info.py
```

## Version History

- **1.0.0**: Initial release with basic planetary visualization
- **1.1.0**: Added trajectory planning
- **1.2.0**: Historical events and time navigation
- **1.3.0**: Starfield background and improved rendering
- **1.4.0**: Moon systems and asteroid belt
- **1.5.0**: Educational overlays and immersion checklist
- **1.6.0**: Date picker and manual time navigation
