# Chapter 5 — Scientific Modeling Tools

**Parent Document:** [Tools User Manual](./TOOLS_USER_MANUAL.md)

---

## 5.1 Solar System Model

**Source:** `src/scientific_modeling/solar_system_model/`
**Status:** ✅ Implemented (MATLAB + Python launcher)

### 5.1.1 Purpose

N-body gravitational simulation of the solar system with visualization.

### 5.1.2 Mathematical Model

**Newton's Law of Universal Gravitation:**

$$\vec{F}_{ij} = -G \frac{m_i m_j}{|\vec{r}_{ij}|^3} \vec{r}_{ij}$$

**Equations of Motion (N-body):**

$$\ddot{\vec{r}}_i = -G \sum_{j \neq i} \frac{m_j}{|\vec{r}_i - \vec{r}_j|^3} (\vec{r}_i - \vec{r}_j)$$

### 5.1.3 Numerical Integration

The system of ODEs is integrated using adaptive step-size methods (typically RK45 or Verlet integration) to maintain energy conservation.

---

## 5.2 RRT Path Planner

**Source:** `src/scientific_modeling/rrt_path_planner/`
**Status:** ✅ Implemented (MATLAB + Python)

### 5.2.1 Purpose

Rapidly-exploring Random Tree (RRT) path planning for robotics and autonomous navigation. Includes a Star Wars-themed demonstration.

### 5.2.2 RRT Algorithm

1. Sample random configuration $q_{rand}$
2. Find nearest node $q_{near}$ in tree
3. Extend toward $q_{rand}$ by step size $\delta$:

$$q_{new} = q_{near} + \delta \cdot \frac{q_{rand} - q_{near}}{\|q_{rand} - q_{near}\|}$$

4. Check collision with obstacles
5. Add $q_{new}$ to tree if collision-free
6. Repeat until goal reached

### 5.2.3 RRT* Variant

RRT* adds rewiring for asymptotic optimality:

$$\text{cost}(q_{new}) = \min_{q_{near} \in \mathcal{N}} \left[\text{cost}(q_{near}) + \|q_{new} - q_{near}\|\right]$$

where $\mathcal{N}$ is the set of near nodes within radius $r = \gamma \cdot (\log n / n)^{1/d}$.

### 5.2.4 Features

- 2D and 3D path planning
- Configurable obstacle environments
- Ship/vehicle models with different dynamics
- Visualization with matplotlib

---

## 5.3 Function Generator

**Source:** `src/function_generator/`
**GUI:** PyQt6
**Status:** ✅ Implemented

### 5.3.1 Purpose

Interactive function generator for creating, combining, and visualizing mathematical functions. Leverages the Signal Toolkit's `SignalGenerator` and `PolynomialGeneratorWidget`.

### 5.3.2 Capabilities

- Generate standard waveforms (sine, square, triangle, sawtooth, chirp)
- Polynomial function creation and visualization
- Superposition of multiple signals
- FFT spectrum analysis
- Interactive parameter adjustment
- Export to CSV/JSON

### 5.3.3 Key Equations

All signal generation equations are documented in [Chapter 4 — Signal Processing Toolkit](./04_signal_toolkit.md#42-signal-generation).

---

*[← Signal Toolkit](./04_signal_toolkit.md) | [Back to Manual](./TOOLS_USER_MANUAL.md) | [Next: Robotics & 3D →](./06_robotics_3d.md)*
