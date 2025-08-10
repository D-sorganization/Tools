# copilot-instructions.md — GITHUB COPILOT STRICT RULES
**I am GitHub Copilot. I generate COMPLETE, TESTED, DETERMINISTIC code. NO PLACEHOLDERS. NO APPROXIMATIONS.**

--------------------------------------------------------------------------------
## MANDATORY REQUIREMENTS FOR EVERY SUGGESTION
1. **SHOW EXACT COMMANDS AND THEIR FULL OUTPUT** (never summaries)
2. **INCLUDE NEGATIVE TESTS** (what happens with bad input) 
3. **EXPLAIN EVERY NUMBER WITH SOURCE** (no magic numbers ever)
4. **IF YOU WRITE "approximately" or use placeholder, START OVER**

--------------------------------------------------------------------------------
## HARD STOPS (NON-NEGOTIABLE)
- If on `main` branch → STOP, create feature branch first
- If value unknown → STOP, never guess or approximate  
- If file >100 lines → NEVER truncate, use targeted edits
- If scientific calculation → REQUIRE source + units for every constant
- BANNED: `TODO`, `FIXME`, `...`, `pass`, `NotImplementedError`, `<placeholder>`, "approximately"

--------------------------------------------------------------------------------
## FILE PRESERVATION
- **NEVER DELETE** working code unless explicit "remove X"
- **NEVER TRUNCATE** with "rest remains the same" 
- **ALWAYS PRESERVE** all imports, functions, comments
- For large files: Make surgical edits, show exact line ranges

--------------------------------------------------------------------------------
## SCIENTIFIC COMPUTING REQUIREMENTS
```python
# EVERY constant needs citation + units (NO EXCEPTIONS):
GRAVITY_M_S2: Final[float] = 9.80665  # [m/s²] ISO 80000-3:2006 standard gravity
GOLF_BALL_MASS_KG: Final[float] = 0.04593  # [kg] USGA Rule 5-1 (1.620 oz)
GOLF_BALL_DIAMETER_M: Final[float] = 0.04267  # [m] USGA Rule 5-2 (1.680 in minimum)
AIR_DENSITY_KG_M3: Final[float] = 1.225  # [kg/m³] ISA sea level, 15°C

# NEVER write this:
velocity = distance * 3.6  # NO! What is 3.6?

# ALWAYS write this:
MPS_TO_KPH: Final[float] = 3.6  # [m/s → km/h] = (1000m/1km)/(3600s/1h)
velocity_kph = velocity_mps * MPS_TO_KPH

# EVERY calculation needs validation + error handling:
def calculate_velocity(energy_j: float, mass_kg: float) -> float:
    """Calculate velocity from kinetic energy.
    
    Formula: v = sqrt(2E/m) from E = (1/2)mv²
    
    Args:
        energy_j: Kinetic energy [J]
        mass_kg: Mass [kg]
        
    Returns:
        Velocity [m/s]
        
    Raises:
        ValueError: If energy negative or mass non-positive
    """
    if energy_j < 0:
        raise ValueError(f"Energy cannot be negative: {energy_j} J")
    if mass_kg <= 0:
        raise ValueError(f"Mass must be positive: {mass_kg} kg")
    
    return math.sqrt(2 * energy_j / mass_kg)

# TEST immediately after function:
def test_calculate_velocity():
    """Test with known values."""
    # 1 J energy, 2 kg mass → v = 1 m/s
    assert calculate_velocity(1.0, 2.0) == pytest.approx(1.0)
    
def test_calculate_velocity_invalid():
    """Test error handling."""
    with pytest.raises(ValueError, match="Energy cannot be negative"):
        calculate_velocity(-1.0, 1.0)
    with pytest.raises(ValueError, match="Mass must be positive"):
        calculate_velocity(1.0, 0.0)

# DETERMINISM IS MANDATORY (set all seeds):
np.random.seed(42)
random.seed(42)
if 'torch' in sys.modules:
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
```

--------------------------------------------------------------------------------
## COMPLETENESS CHECKLIST (ALL REQUIRED)
Before ANY completion, verify and show output:
- [X] Runs: `python -m py_compile <file>` → no errors
- [X] Types: All functions have type hints + returns
- [X] Docs: Docstrings with units for all functions
- [X] Errors: Try/except for all I/O and calculations  
- [X] Constants: All numbers defined with source citation
- [X] Tests: Positive + negative + edge cases exist
- [X] Quality: `python scripts/quality-check.py` → PASSED

Show this output:
```bash
$ python scripts/quality-check.py
✅ Quality check PASSED
Checked 25 Python files

$ pytest -xvs <test_file>
test_function PASSED
test_function_invalid PASSED
================== 2 passed ====================
```

--------------------------------------------------------------------------------
## PYTHON PATTERNS (REQUIRED)
```python
# Logging (no print in libraries)
logger = logging.getLogger(__name__)
logger.info(f"Processing {len(data)} samples")

# Path handling with validation
from pathlib import Path
config_path = Path("config") / "settings.yaml"
if not config_path.exists():
    raise FileNotFoundError(f"Config not found: {config_path.absolute()}")

# Type aliases for clarity
from typing import Final, TypeAlias
Vector3D: TypeAlias = tuple[float, float, float]  # [x, y, z] in meters
ForceVector: TypeAlias = tuple[float, float, float]  # [Fx, Fy, Fz] in Newtons

# Specific exceptions with context
try:
    data = pd.read_csv(filepath)
except FileNotFoundError as e:
    logger.error(f"Data file missing: {filepath}")
    raise FileNotFoundError(f"Required data file not found: {filepath}") from e
except pd.errors.EmptyDataError as e:
    logger.error(f"Data file empty: {filepath}")
    return pd.DataFrame()  # Return empty df as valid response
except Exception as e:
    logger.error(f"Unexpected error reading {filepath}: {e}")
    raise

# Constants at module level with citations
CLUB_LENGTHS_M: Final[dict[str, float]] = {
    "driver": 1.1684,      # [m] 46 inches max per USGA Rule 1-1c
    "3_wood": 1.0922,      # [m] 43 inches typical
    "5_iron": 0.9652,      # [m] 38 inches typical
    "pitching_wedge": 0.8890,  # [m] 35 inches typical
}
```

--------------------------------------------------------------------------------
## MATLAB PATTERNS (REQUIRED)
```matlab
function results = analyze_swing(data_file, options)
    % ANALYZE_SWING Process golf swing data with full validation
    %
    % Every constant must be documented:
    GRAVITY_M_S2 = 9.80665;  % [m/s^2] ISO 80000-3:2006
    SAMPLE_RATE_HZ = 1000;   % [Hz] Motion capture frequency
    
    arguments
        data_file (1,1) string {mustBeFile}
        options.seed (1,1) double {mustBePositive} = 42
        options.tolerance (1,1) double {mustBePositive} = 1e-6
    end
    
    % Set reproducibility
    rng(options.seed, 'twister');
    
    % Validate with clear errors
    assert(exist(data_file, 'file') == 2, ...
           'ANALYZE_SWING:FileNotFound', ...
           'Data file not found: %s', data_file);
    
    % Load with error handling
    try
        data = readmatrix(data_file);
        validateattributes(data, {'numeric'}, {'2d', 'nonempty'});
    catch ME
        fprintf('Error loading %s: %s\n', data_file, ME.message);
        rethrow(ME);
    end
    
    % Check for NaN/Inf
    if any(~isfinite(data(:)))
        error('ANALYZE_SWING:InvalidData', ...
              'Data contains NaN or Inf values');
    end
    
    % Process with metadata
    results.timestamp = datetime('now', 'TimeZone', 'UTC');
    results.matlab_version = version;
    results.data_file = char(data_file);
    
    % ... actual processing ...
end
```

--------------------------------------------------------------------------------
## INLINE COMPLETION RULES
When generating completions:
1. **Complete the entire logical unit** (full function + tests)
2. **Include all necessary imports** at file top
3. **Add type hints** automatically
4. **Generate tests** immediately after function
5. **Add constants** with citations for any numbers
6. **Never leave stubs** - implement fully or don't suggest

Example of COMPLETE suggestion:
```python
from typing import Final
import numpy as np

# Physical constants with sources
DRAG_COEFFICIENT_SPHERE: Final[float] = 0.47  # Smooth sphere, Reynolds number > 10^4
AIR_DENSITY_KG_M3: Final[float] = 1.225  # [kg/m³] ISA sea level, 15°C

def calculate_drag_force(velocity_mps: float, diameter_m: float) -> float:
    """Calculate aerodynamic drag on sphere.
    
    Formula: F_d = 0.5 * ρ * v² * C_d * A
    Source: Fluid Mechanics by White, 7th Ed, Ch 7.3
    
    Args:
        velocity_mps: Velocity relative to air [m/s]
        diameter_m: Sphere diameter [m]
        
    Returns:
        Drag force [N]
        
    Raises:
        ValueError: If velocity < 0 or diameter <= 0
    """
    if velocity_mps < 0:
        raise ValueError(f"Velocity must be non-negative, got {velocity_mps} m/s")
    if diameter_m <= 0:
        raise ValueError(f"Diameter must be positive, got {diameter_m} m")
    
    area_m2 = np.pi * (diameter_m / 2) ** 2
    return 0.5 * AIR_DENSITY_KG_M3 * velocity_mps**2 * DRAG_COEFFICIENT_SPHERE * area_m2

def test_calculate_drag_force():
    """Test drag calculation with known values."""
    # Golf ball at 70 m/s
    force = calculate_drag_force(70.0, 0.04267)
    assert force > 0
    assert force == pytest.approx(2.03, rel=0.01)  # Expected ~2 N
    
def test_calculate_drag_force_invalid():
    """Test error handling."""
    with pytest.raises(ValueError, match="Velocity must be non-negative"):
        calculate_drag_force(-1.0, 0.04267)
    with pytest.raises(ValueError, match="Diameter must be positive"):
        calculate_drag_force(70.0, 0.0)
```

--------------------------------------------------------------------------------
## ANTI-LAZY EXAMPLES

### BAD (INSTANTLY REJECT):
```python
# ALL OF THESE ARE UNACCEPTABLE:
def process_data(df):
    # TODO: implement processing
    pass

def calculate_swing_speed(data):
    return 45.0  # approximate average

angle = 30  # degrees probably
gravity = 9.8  # close enough
conversion_factor = 3.6  # for something

def helper_function():
    ...  # Details omitted

# Approximately 1/3 for moment of inertia
moi = mass * length * 0.33
```

### GOOD (ACCEPT):
```python
# Constants with full documentation
TYPICAL_DRIVER_SPEED_MPS: Final[float] = 48.8  # [m/s] PGA Tour average 2023 per TrackMan
GRAVITY_M_S2: Final[float] = 9.80665  # [m/s²] ISO 80000-3:2006
MPS_TO_MPH: Final[float] = 2.23694  # [m/s → mph] exact conversion

def calculate_club_head_speed(
    shaft_length_m: float,
    angular_velocity_rad_s: float
) -> float:
    """Calculate club head speed from rotation.
    
    Linear velocity at radius: v = ω × r
    
    Args:
        shaft_length_m: Club shaft length [m]
        angular_velocity_rad_s: Angular velocity [rad/s]
        
    Returns:
        Club head speed [m/s]
        
    Raises:
        ValueError: If shaft_length <= 0
    """
    if shaft_length_m <= 0:
        raise ValueError(f"Shaft length must be positive, got {shaft_length_m} m")
    
    return abs(angular_velocity_rad_s * shaft_length_m)

def test_calculate_club_head_speed():
    """Test with typical driver swing."""
    # 1.1684 m shaft, 250°/s = 4.363 rad/s
    speed = calculate_club_head_speed(1.1684, 4.363)
    assert speed == pytest.approx(5.098, rel=1e-3)
    
def test_calculate_club_head_speed_invalid():
    """Test error cases."""
    with pytest.raises(ValueError, match="Shaft length must be positive"):
        calculate_club_head_speed(0.0, 4.363)
    with pytest.raises(ValueError, match="Shaft length must be positive"):
        calculate_club_head_speed(-1.0, 4.363)
```

--------------------------------------------------------------------------------
## BRANCH AND COMMIT PATTERNS
Always suggest:
- Branch names: `feat/drag-calculation`, `fix/nan-handling`, `refactor/add-type-hints`
- Commit messages with WHY: 
  ```
  fix(physics): correct Magnus force sign convention
  
  Was using wrong sign for spin direction, causing upward force
  for backspin to be negative. Fixed per Resnick/Halliday Ch 14.
  
  All constants now cite sources.
  ```

--------------------------------------------------------------------------------
## FINAL RULE - PROOF OF COMPLETENESS
**Before claiming done, ALWAYS show:**
```bash
$ python scripts/quality-check.py
✅ Quality check PASSED

$ ruff check <file>
All checks passed!

$ mypy <file> --strict
Success: no issues found

$ pytest <test_file> -xvs
test_valid PASSED
test_invalid PASSED
test_edge_cases PASSED
================== 3 passed ====================
```

**If you cannot show this, output:**
```
INCOMPLETE: Cannot verify - missing quality-check.py
DONE: Function implemented with validation
NEEDED: Run quality-check.py to verify no placeholders
```

Never pretend it's done without proof.

# END COPILOT RULES — NO APPROXIMATIONS, NO PLACEHOLDERS, ALWAYS COMPLETE
    if shaft_length_m <= 0:
        raise ValueError(f"Shaft length must be positive, got {shaft_length_m} m")
    
    return abs(angular_velocity_rad_s * shaft_length_m)

def test_calculate_club_head_speed():
    """Test with typical driver swing."""
    # 1.1684 m shaft, 250°/s = 4.363 rad/s
    speed = calculate_club_head_speed(1.1684, 4.363)
    assert speed == pytest.approx(5.098, rel=1e-3)
    
def test_calculate_club_head_speed_invalid():
    """Test error cases."""
    with pytest.raises(ValueError, match="Shaft length must be positive"):
        calculate_club_head_speed(0.0, 4.363)
    with pytest.raises(ValueError, match="Shaft length must be positive"):
        calculate_club_head_speed(-1.0, 4.363)
    