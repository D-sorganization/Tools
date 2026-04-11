# Draft GitHub Issues

> Copy these into GitHub Issues at https://github.com/D-sorganization/Tools/issues/new
> After creating each issue, update the issue number in FEATURES.md.

---

## Issue 1: Net force & equivalent couple of two-hand club action

**Labels**: enhancement, pendulum-simulator

### Summary

Calculate the net force and equivalent couple of the two hands acting on the club in the golfer (8-DOF) model.

### Requirements

**Net force calculation**

- Sum the two hand forces (grip_right + grip_left constraint forces) to obtain the net force vector acting on the club
- The net force acts at a user-configurable point on the club (default: midpoint between the two grip locations)
- This action point should be adjustable via a slider or spinbox in the golfer controls

**Moment of net force**

- Calculate the moment of the net force vector about the action point
- Display as a vector overlay on the animation widget (new colour distinct from existing force overlays)

**Equivalent couple**

- Compute the couple required to match the full two-hand action (individual moments + individual forces) when replaced by the net force at the action point
- Couple = total moment from two hands minus moment of net force at action point

**Force decomposition** — all three must be supported:

1. Overall forces (from full constrained dynamics)
2. ZTCF forces (from zero-torque counterfactual transfer matrix)
3. DELTA forces (from M-pseudoinverse, zero-velocity decomposition)

**Visualisation**

- Toggleable vector overlays for net force, moment, and couple
- Add corresponding data series to the data extractor registry for 2D plotting
- Display numerical values in the matrix widget or a dedicated readout

### Implementation notes

- Extend `jacobians_golfer.py` or create `club_forces.py`
- Use existing `constraint_forces()` output for the hand force components
- DRY/DbC/TDD: tests first, pre/postcondition assertions, reuse existing force infrastructure

### Acceptance criteria

- [ ] Net force vector computed and displayable
- [ ] Moment of net force computed
- [ ] Equivalent couple computed
- [ ] All three decompositions (overall, ZTCF, DELTA) supported
- [ ] User can adjust the action point on the club
- [ ] Data series registered in data_extractor.py
- [ ] Unit tests covering all calculations
- [ ] CI passes (ruff, mypy, pytest)

---

## Issue 2: Random perturbation / consistency analysis

**Labels**: enhancement, pendulum-simulator

### Summary

Add Monte Carlo perturbation analysis to assess swing consistency by injecting noise into torque profiles and measuring output variability.

### Requirements

**Noise injection**

- Perturb all joint torque profiles with configurable noise (white, pink, brown)
- Use the existing signal toolkit for noise generation and filtering
- User controls: noise amplitude (% of peak torque), noise type, number of trials

**Batch simulation**

- Run N simulations (default 100) with independent noise realisations
- Collect full trajectory data for each trial
- Use Rust batch evaluator where available for performance

**Variability analysis**

- Compute per-timestep statistics: mean, std, min, max, percentiles (5/95)
- Key metrics: tip velocity variability, tip position variability, club angle at impact
- Identify which swing patterns produce more consistent outcomes

**GUI**

- Batch run dialog with progress bar
- Summary statistics table (mean, std, CV for key metrics)
- Variability envelope plots (mean +/- 1/2 sigma bands)
- Histogram of final tip speed distribution

### Implementation notes

- Create `perturbation_analysis.py` module
- Integrate with `function_generator_dialog.py` noise capabilities
- Thread the batch runs to keep GUI responsive
- DRY/DbC/TDD

### Acceptance criteria

- [ ] Noise injection working via signal toolkit
- [ ] Batch simulation runs N trials
- [ ] Statistical summary computed
- [ ] Variability plots generated
- [ ] All models supported (double, triple, golfer)
- [ ] Unit tests for statistics and noise injection
- [ ] CI passes

---

## Issue 3: Massless hub standoff & adjustable rotation centre

**Labels**: enhancement, pendulum-simulator

### Summary

Make the hub standoff (connecting rotation centre to scapula origin) essentially massless and allow the user to position the rotation centre at the system centre of mass.

### Requirements

**Massless standoff**

- Add option to set hub standoff mass to effectively zero (epsilon ~ 1e-6 kg)
- Ensure mass matrix remains positive-definite with near-zero hub mass
- Update EOM derivation to handle the limiting case gracefully

**COM-tracking rotation centre**

- Calculate system centre of mass from all segment masses and positions
- Option to auto-position the hub at the system COM
- In zero-gravity mode, this should yield zero hub reaction force

**User controls**

- Checkbox: "Massless standoff"
- Checkbox: "Centre of rotation at COM"
- Manual offset input (x, y) for fine-tuning rotation centre
- Display current COM position in the GUI

### Implementation notes

- Modify `GolferParams` to add `massless_hub` flag and `hub_at_com` flag
- Add `hub_offset` parameter (x, y) to `GolferParams`
- Update `mass_matrix()` and `potential_energy()` for near-zero hub mass
- Create `compute_system_com()` utility function
- DRY/DbC/TDD

### Acceptance criteria

- [ ] Hub mass can be set to ~0 without numerical instability
- [ ] COM calculation correct for all configurations
- [ ] Hub-at-COM mode yields zero hub force in zero-g
- [ ] Manual offset works correctly
- [ ] GUI controls in golfer panel
- [ ] Unit tests for COM calculation and massless case
- [ ] CI passes

---

## Issue 4: 3D segment rendering options

**Labels**: enhancement, pendulum-simulator

### Summary

Provide multiple visual representations for pendulum segments beyond the current line rendering.

### Requirements

**Rendering modes**

- Lines (current default) — keep as default
- Cylinders (constant radius per segment)
- Ellipsoids (inertia-scaled, semi-axes from mass/length)
- Tapered cylinders (thicker at proximal end)

**Per-segment control**

- Global dropdown to set rendering mode for all segments
- Optional per-segment override via right-click context menu
- Segment radius/thickness parameter (auto-scaled from mass by default)

**Depth sorting**

- Correct occlusion ordering for 3D rendering modes
- Z-buffer or painter's algorithm for overlapping segments

**Performance**

- Efficient QPainter rendering (pre-compute cross-sections)
- No significant FPS drop compared to line mode

### Implementation notes

- Extend `base_pendulum_widget.py` with `SegmentStyle` enum
- Add rendering methods: `_draw_cylinder()`, `_draw_ellipsoid()`, `_draw_tapered()`
- Use the existing pseudo-3D projection infrastructure
- DRY/DbC/TDD

### Acceptance criteria

- [ ] Four rendering modes available
- [ ] Global and per-segment selection
- [ ] Correct depth sorting in 3D mode
- [ ] Performance within 10% of line rendering
- [ ] All three models supported
- [ ] Unit tests for geometry calculations
- [ ] CI passes

---

## Issue 5: Feature tracking document

**Labels**: documentation, pendulum-simulator

### Summary

Maintain a living document (FEATURES.md) that categorises all desired vs. implemented features to avoid relying on memory.

### Requirements

- Comprehensive feature inventory with status icons
- Categorised by: physics models, force analysis, GUI, analysis, signal generation, optimisation, backends, export, testing
- Planned features section with descriptions and acceptance criteria
- Updated whenever a feature is proposed, started, or completed

### Status

**COMPLETE** — `src/pendulum_simulator/FEATURES.md` created.

---
