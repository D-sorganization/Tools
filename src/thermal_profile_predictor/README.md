# Thermal Profile Predictor

A thermal profile prediction tool for process units including heated vessels, reactors, and heat exchangers. This calculator solves the transient heat balance equation to predict temperature evolution over time with various power input profiles.

## Purpose

The Thermal Profile Predictor enables process engineers to:

- Predict temperature trajectories in heated vessels
- Model batch heating and cooling processes
- Evaluate different power input strategies (constant, ramp, step)
- Assess thermal lag and time-to-temperature
- Optimize heating cycles for energy efficiency

## Key Features

- **Multiple Power Profiles**: Constant, linear ramp, and step function heating
- **Transient Analysis**: Full time-dependent temperature prediction
- **Heat Loss Modeling**: Configurable heat loss coefficient for realistic predictions
- **Flexible Time Range**: Customizable simulation duration and resolution
- **Results Summary**: Key metrics including final, max, min temperatures
- **Sample Data Points**: Tabular output of temperature vs time
- **Catppuccin Mocha Theme**: Modern dark interface design

## Installation / Prerequisites

### Required Dependencies

```bash
pip install PyQt6 numpy scipy
```

### Optional Dependencies

```bash
# For enhanced plotting
pip install matplotlib
```

### Python Version

- Python 3.10 or higher recommended

## Usage Instructions

### Launch Desktop Application

```bash
python launch_pyqt6.py
```

### Programmatic Usage

```python
from upstream_drift_tools.process_calculators.thermal_profile_predictor import (
    predict_temperature_profile
)
import numpy as np

# Define power function (constant 5000 W)
power_func = lambda t: 5000

# Run prediction
times, temps = predict_temperature_profile(
    t_span=(0, 3600),
    t_eval=np.linspace(0, 3600, 100),
    initial_temp=25.0,
    thermal_mass=50000,
    heat_loss_coeff=50.0,
    ambient_temp=25.0,
    power_func=power_func,
)

print(f"Final temperature: {temps[-1]:.1f} C")
print(f"Temperature rise: {temps[-1] - temps[0]:.1f} C")
```

## Input Parameters

### Thermal Parameters

| Parameter             | Description                               | Range           | Units | Default |
| --------------------- | ----------------------------------------- | --------------- | ----- | ------- |
| Initial Temperature   | Starting temperature of the system        | -273.15 to 2000 | C     | 25.0    |
| Ambient Temperature   | Surrounding environment temperature       | -273.15 to 500  | C     | 25.0    |
| Thermal Mass          | Heat capacity of the system (m \* Cp)     | 1 to 10^9       | J/K   | 50,000  |
| Heat Loss Coefficient | Overall heat transfer coefficient \* area | 0 to 10,000     | W/K   | 50.0    |

### Time Parameters

| Parameter   | Description             | Range        | Units | Default |
| ----------- | ----------------------- | ------------ | ----- | ------- |
| Start Time  | Simulation start time   | 0 to 10^6    | s     | 0       |
| End Time    | Simulation end time     | 1 to 10^6    | s     | 3600    |
| Data Points | Number of output points | 10 to 10,000 | -     | 100     |

### Power Input Parameters

| Parameter     | Description                            | Range       | Units | Default  |
| ------------- | -------------------------------------- | ----------- | ----- | -------- |
| Power Profile | Type of heating profile                | -           | -     | Constant |
| Power         | Base power level                       | 0 to 10^6   | W     | 5000     |
| Ramp Rate     | Power increase rate (linear ramp only) | 0 to 10,000 | W/s   | 1.0      |
| Step Time     | Time when power turns off (step only)  | 0 to 10^6   | s     | 1800     |

### Power Profile Types

| Profile       | Description              | Equation                        |
| ------------- | ------------------------ | ------------------------------- |
| Constant      | Fixed power throughout   | P(t) = P_0                      |
| Linear Ramp   | Power increases linearly | P(t) = P_0 + r \* t             |
| Step Function | Power on until step time | P(t) = P_0 if t < t_step else 0 |

## Output Format

### Function Return Values

```python
times, temps = predict_temperature_profile(...)
```

| Return  | Type       | Description               |
| ------- | ---------- | ------------------------- |
| `times` | np.ndarray | Array of time points (s)  |
| `temps` | np.ndarray | Array of temperatures (C) |

### GUI Results Display

The results panel shows:

- Input parameters summary
- Final temperature
- Maximum temperature reached
- Minimum temperature
- Total temperature change
- Sample data points at regular intervals

## Mathematical Models

### Heat Balance Equation

The fundamental transient heat balance:

```
m * Cp * dT/dt = Q_in - Q_loss
```

Where:

- `m * Cp` = Thermal mass (J/K)
- `dT/dt` = Rate of temperature change (K/s)
- `Q_in` = Heat input power (W)
- `Q_loss` = Heat loss to surroundings (W)

### Heat Loss Model

Newton's law of cooling:

```
Q_loss = UA * (T - T_ambient)
```

Where:

- `UA` = Heat loss coefficient (W/K)
- `T` = System temperature (C or K)
- `T_ambient` = Ambient temperature (C or K)

### Governing ODE

```
dT/dt = [P(t) - UA * (T - T_ambient)] / (m * Cp)
```

Rearranged form:

```
dT/dt = P(t)/(m*Cp) - (UA/(m*Cp)) * (T - T_ambient)
```

### Steady-State Temperature

For constant power, equilibrium temperature:

```
T_steady = T_ambient + P / UA
```

### Time Constant

Thermal time constant:

```
tau = (m * Cp) / UA
```

Time to reach 63.2% of final temperature change.

### Analytical Solution (Constant Power)

For constant power with heat loss:

```
T(t) = T_steady - (T_steady - T_0) * exp(-t/tau)
```

Where:

- `T_steady` = Equilibrium temperature
- `T_0` = Initial temperature
- `tau` = Time constant

## Example Usage

### Batch Vessel Heating

```python
import numpy as np
from upstream_drift_tools.process_calculators.thermal_profile_predictor import (
    predict_temperature_profile
)

# 1000 L water vessel with 10 kW heater
# Water: rho = 1000 kg/m3, Cp = 4186 J/kg-K
# Vessel has 2 m2 surface, U = 25 W/m2-K

thermal_mass = 1000 * 4186  # 4.186 MJ/K
heat_loss = 25 * 2  # 50 W/K

times, temps = predict_temperature_profile(
    t_span=(0, 7200),  # 2 hours
    t_eval=np.linspace(0, 7200, 100),
    initial_temp=20.0,
    thermal_mass=thermal_mass,
    heat_loss_coeff=heat_loss,
    ambient_temp=20.0,
    power_func=lambda t: 10000,  # 10 kW
)

# Calculate time to reach target
target_temp = 80
for i, temp in enumerate(temps):
    if temp >= target_temp:
        print(f"Time to {target_temp}C: {times[i]/60:.1f} minutes")
        break
```

### Ramp Heating Profile

```python
# Gentle ramp to avoid thermal shock
power_base = 2000  # Start at 2 kW
ramp_rate = 2  # Increase 2 W/s

def ramp_power(t):
    return min(power_base + ramp_rate * t, 10000)  # Cap at 10 kW

times, temps = predict_temperature_profile(
    t_span=(0, 5000),
    t_eval=np.linspace(0, 5000, 100),
    initial_temp=25,
    thermal_mass=50000,
    heat_loss_coeff=50,
    ambient_temp=25,
    power_func=ramp_power,
)
```

### Cool-Down Analysis

```python
# Analyze cooling after heater shutdown
times, temps = predict_temperature_profile(
    t_span=(0, 3600),
    t_eval=np.linspace(0, 3600, 100),
    initial_temp=150.0,  # Start hot
    thermal_mass=50000,
    heat_loss_coeff=50,
    ambient_temp=25,
    power_func=lambda t: 0,  # No heating
)

# Calculate cooling rate
cooling_rate = (temps[0] - temps[-1]) / 60  # C/min average
print(f"Average cooling rate: {cooling_rate:.2f} C/min")
```

### Step Response (Heater Cycling)

```python
# Heater on for 30 min, then off
step_time = 1800  # seconds

def step_power(t):
    return 5000 if t < step_time else 0

times, temps = predict_temperature_profile(
    t_span=(0, 7200),  # 2 hours total
    t_eval=np.linspace(0, 7200, 200),
    initial_temp=25,
    thermal_mass=50000,
    heat_loss_coeff=50,
    ambient_temp=25,
    power_func=step_power,
)

max_temp = max(temps)
print(f"Peak temperature: {max_temp:.1f} C")
```

## Troubleshooting

| Issue                               | Cause                               | Solution                                       |
| ----------------------------------- | ----------------------------------- | ---------------------------------------------- |
| Temperature exceeds physical limits | Power too high or heat loss too low | Increase heat loss coefficient or reduce power |
| Temperature never stabilizes        | Heat loss coefficient is zero       | Add realistic heat loss (UA > 0)               |
| Negative temperatures (unphysical)  | Large negative power or heat loss   | Check input parameters for sign errors         |
| Solution diverges                   | Time step too large                 | Increase number of data points                 |
| Very slow temperature rise          | Thermal mass too large              | Verify thermal mass calculation                |
| Immediate temperature jump          | Thermal mass too small              | Check m\*Cp value (should include vessel)      |

### Parameter Estimation

**Thermal Mass (m \* Cp):**

```
Thermal Mass = mass (kg) * specific heat (J/kg-K)
Water: Cp = 4186 J/kg-K
Steel: Cp = 500 J/kg-K
Aluminum: Cp = 900 J/kg-K
```

**Heat Loss Coefficient (UA):**

```
UA = U * A
U: overall heat transfer coefficient (W/m2-K)
A: surface area (m2)

Typical U values:
- Natural convection (air): 5-25 W/m2-K
- Forced convection (air): 25-250 W/m2-K
- Water jacket: 300-2000 W/m2-K
- Insulated vessel: 0.5-3 W/m2-K
```

## Related Tools

- **ODE Solver**: General-purpose solver for custom thermal models
- **Steam Engine Calculator**: Steam properties for jacketed heating
- **Flow Rate Converter**: Mass flow for heating/cooling media
- **PSA Package**: Thermal regeneration of adsorption beds

## References

### Heat Transfer Equations

| Parameter             | Symbol     | SI Units |
| --------------------- | ---------- | -------- |
| Thermal mass          | m\*Cp or C | J/K      |
| Heat loss coefficient | UA         | W/K      |
| Power                 | P or Q     | W        |
| Temperature           | T          | C or K   |
| Time constant         | tau        | s        |

### Typical Time Constants

| System                     | Time Constant (tau) |
| -------------------------- | ------------------- |
| Small electronics          | 1-10 s              |
| Laboratory beaker (1 L)    | 100-500 s           |
| Industrial vessel (1000 L) | 2000-10000 s        |
| Building (thermal mass)    | 10-50 hours         |

### Energy Balance Verification

At steady state:

```
Q_in = Q_loss
P = UA * (T_steady - T_ambient)
```

Total energy input over time:

```
E_total = integral(P(t) dt)
E_stored = m * Cp * (T_final - T_initial)
E_lost = integral(UA * (T - T_ambient) dt)
E_total = E_stored + E_lost
```
