# PR Agent Instructions — Pendulum Simulator Refactor

These instructions are for an agent to create a pull request from the current
uncommitted work on the `feat/integrate-programmatic-pid` branch and then
verify the changes are correct.

---

## 1. Understand the current state

The work lives in two places:

- **Git stash `stash@{0}`** ("pendulum UI work in progress" on branch
  `fix/pendulum-ui-polish`) — contains modifications to 12 tracked files.
- **Untracked file**:
  `src/pendulum_simulator/src/double_pendulum_golf/gui/base_pendulum_widget.py`

The parent branch is `feat/integrate-programmatic-pid`. The target merge
branch should be `main`.

## 2. Create the PR branch and commit

```bash
# Starting from repo root
cd /path/to/Tools

# Create a fresh branch off main
git checkout main
git pull origin main
git checkout -b fix/pendulum-ui-polish

# Apply the stashed changes
git stash apply stash@{0}

# Stage the new file that the stash doesn't track
git add src/pendulum_simulator/src/double_pendulum_golf/gui/base_pendulum_widget.py

# Stage all the pendulum simulator source changes (Python + Rust only)
git add \
  src/pendulum_simulator/src/double_pendulum_golf/gui/controls_widget.py \
  src/pendulum_simulator/src/double_pendulum_golf/gui/controls_widget_golfer.py \
  src/pendulum_simulator/src/double_pendulum_golf/gui/controls_widget_triple.py \
  src/pendulum_simulator/src/double_pendulum_golf/gui/function_generator_dialog.py \
  src/pendulum_simulator/src/double_pendulum_golf/gui/golfer_pendulum_widget.py \
  src/pendulum_simulator/src/double_pendulum_golf/gui/main_window.py \
  src/pendulum_simulator/src/double_pendulum_golf/gui/matrix_widget.py \
  src/pendulum_simulator/src/double_pendulum_golf/gui/matrix_widget_golfer.py \
  src/pendulum_simulator/src/double_pendulum_golf/gui/pendulum_widget.py \
  src/pendulum_simulator/src/double_pendulum_golf/gui/simulation_panel.py \
  src/pendulum_simulator/src/double_pendulum_golf/gui/toolstrip_widget.py \
  src/pendulum_simulator/pendulum-core/src/lib.rs

# Commit
git commit -m "refactor(pendulum): extract base class, fix UI bugs, add DbC contracts

- Extract BasePendulumWidget shared base class from PendulumWidget and
  GolferPendulumWidget to eliminate DRY violations (~500 lines removed)
- Fix force_scale not applied in _draw_force_vectors
- Fix toolbar checkbox state desync when switching model tabs
- Fix non-active panel events leaking to toolstrip (run button, frame slider)
- Add ground plane, tilt plane, and golf ball visualizations
- Replace fragile sys.path climbing with _find_sibling_package() helper
- Replace Rust for-loops with copy_from_slice and iterator chains in lib.rs
- Add DbC assertions across all widget set_*/slider/simulation methods

Closes #1041, #1097, #1100, #1101, #1102, #1113, #1115, #1116, #1118

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"

# Push
git push -u origin fix/pendulum-ui-polish
```

## 3. Create the pull request

```bash
gh pr create \
  --base main \
  --head fix/pendulum-ui-polish \
  --title "refactor(pendulum): extract base class, fix UI bugs, add DbC contracts" \
  --body "$(cat <<'EOF'
## Summary

- **DRY**: Extracted `BasePendulumWidget` abstract base class from `PendulumWidget` and `GolferPendulumWidget`, removing ~500 lines of duplicated zoom/pan, coordinate mapping, grid, trail, joint drawing, and feature toggle code.
- **Bug fixes**: Fixed force scale slider having no effect, toolbar checkbox states not syncing on tab switch, and non-active simulation panels leaking events to the shared toolstrip.
- **New visuals**: Added gradient ground plane, tilt plane indicator, and golf ball rendering to the base widget.
- **Import fix**: Replaced fragile `sys.path` climbing loop with a named `_find_sibling_package()` helper with assertions.
- **Rust**: Replaced imperative for-loop patterns with `copy_from_slice` and iterator chains in `lib.rs`.
- **DbC**: Added pre-condition assertions to slider setters, simulation result setters, drawing helpers, and playback speed across 8 files.

## Files changed

| File | Change |
|------|--------|
| `gui/base_pendulum_widget.py` | **NEW** — shared abstract base class |
| `gui/pendulum_widget.py` | Rewritten to inherit from base class |
| `gui/golfer_pendulum_widget.py` | Rewritten to inherit from base class |
| `gui/main_window.py` | Tab-switch sync + panel event guards + import fix |
| `gui/simulation_panel.py` | DbC on speed/result + playback invariant |
| `gui/controls_widget.py` | DbC on slider + torque import |
| `gui/controls_widget_golfer.py` | DbC on slider |
| `gui/controls_widget_triple.py` | DbC on slider |
| `gui/matrix_widget.py` | DbC on set_simulation |
| `gui/matrix_widget_golfer.py` | DbC on set_simulation |
| `gui/function_generator_dialog.py` | Import mechanism fix |
| `gui/toolstrip_widget.py` | Minor adjustments |
| `pendulum-core/src/lib.rs` | Idiomatic Rust replacements |

## Test plan

- [ ] `python -m pytest tests/ -x --tb=short` — all 282 tests pass (8 skipped for headless)
- [ ] `python -m py_compile` succeeds on all 13 modified/new Python files
- [ ] `cargo check` passes in `pendulum-core/` (requires Rust toolchain)
- [ ] Manual: launch GUI, switch tabs, verify checkbox states sync
- [ ] Manual: run simulation, verify force scale slider affects arrow sizes
- [ ] Manual: verify only active tab animates when pressing Run
- [ ] Manual: verify ground plane and tilt plane render correctly
- [ ] Review DbC assertions fire correctly with bad inputs (negative scale, out-of-range slider)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

## 4. Verification checklist for the reviewing agent

After the PR is created, the reviewing agent should perform these checks:

### 4a. Automated checks

```bash
cd src/pendulum_simulator

# Run the full test suite
python -m pytest tests/ -x --tb=short -q

# Verify all Python files compile
for f in \
  src/double_pendulum_golf/gui/base_pendulum_widget.py \
  src/double_pendulum_golf/gui/pendulum_widget.py \
  src/double_pendulum_golf/gui/golfer_pendulum_widget.py \
  src/double_pendulum_golf/gui/main_window.py \
  src/double_pendulum_golf/gui/simulation_panel.py \
  src/double_pendulum_golf/gui/controls_widget.py \
  src/double_pendulum_golf/gui/controls_widget_golfer.py \
  src/double_pendulum_golf/gui/controls_widget_triple.py \
  src/double_pendulum_golf/gui/matrix_widget.py \
  src/double_pendulum_golf/gui/matrix_widget_golfer.py \
  src/double_pendulum_golf/gui/function_generator_dialog.py \
  src/double_pendulum_golf/gui/toolstrip_widget.py \
  src/double_pendulum_golf/simulation_panel.py; do
  python -m py_compile "$f" 2>/dev/null && echo "OK: $f" || echo "FAIL: $f"
done

# If Rust toolchain is available
cd pendulum-core && cargo check && cargo test
```

### 4b. Code review checks

The agent should read each file and verify:

1. **BasePendulumWidget (`base_pendulum_widget.py`)**
   - Has all 5 abstract methods: `_get_total_length`, `_draw_model`,
     `_draw_info`, `_draw_placeholder`, `_has_result`
   - Contains zoom/pan mouse handlers with `_zoom` clamped to `[0.1, 20.0]`
   - `_compute_base_scale()` has post-condition `assert result >= 30.0`
   - `_draw_ball()` has `assert radius_m > 0`
   - `_draw_joint()` has `assert radius > 0`
   - `_catmull_rom_smooth()` has `assert n_sub >= 1`
   - `_draw_ground_plane()` uses QLinearGradient
   - `_draw_tilt_plane()` short-circuits when `abs(tilt_angle) < 1e-4`

2. **PendulumWidget (`pendulum_widget.py`)**
   - Inherits from `BasePendulumWidget`, NOT `QWidget`
   - Implements all 5 abstract methods
   - `_draw_force_vectors` includes `* self._force_scale` in scale calc
   - Does NOT duplicate zoom/pan/grid/trail code from base class

3. **GolferPendulumWidget (`golfer_pendulum_widget.py`)**
   - Inherits from `BasePendulumWidget`, NOT `QWidget`
   - Implements all 5 abstract methods
   - Overrides `_shoulder_y_fraction()` returning `0.30`
   - Does NOT import from `pendulum_widget` for trail/grid

4. **MainWindow (`main_window.py`)**
   - `_on_tab_changed()` syncs these to the active panel's widget:
     forces, zero-torque, mob ellipsoids, force ellipsoids, COM,
     force scale, mob scale, force_ell scale, visible segments
   - `_wire_toolstrip()` uses `_p is self._active_panel()` guards on
     `sim_started`, `sim_finished`, `frame_changed` connections
   - Import uses `_find_sibling_package()`, not raw sys.path loop

5. **SimulationPanel (`simulation_panel.py`)**
   - `_on_speed_change` has `assert speed > 0`
   - `_on_sim_done` has `assert result is not None` and attribute checks
   - `_advance_frame` has `assert self._playback_speed > 0`

6. **Controls widgets** (all three variants)
   - `set_slider_range(max_val)` has `assert max_val >= 0`
   - `set_slider_value(val)` has `assert 0 <= val <= self.slider.maximum()`

7. **Matrix widgets** (both variants)
   - `set_simulation(result)` has `assert result is not None` and
     `assert result.n_steps >= 1`

8. **Rust `lib.rs`**
   - No `for i in 0..8 { q_arr[i] = q[i]; }` patterns remain
   - Uses `q_arr.copy_from_slice(&q[..8])` instead
   - Mass matrix and constraint Jacobian use iterator chains

### 4c. Regression checks

```bash
# Verify no new circular imports
python -c "
import sys; sys.path.insert(0, 'src')
from double_pendulum_golf.simulation import run_simulation
from double_pendulum_golf.simulation_golfer import run_simulation as run_golfer
from double_pendulum_golf.simulation_result_base import TrajectoryResultMixin
print('No circular import issues')
"

# Verify DbC assertions actually fire
python -c "
import sys; sys.path.insert(0, 'src')
from double_pendulum_golf.gui.base_pendulum_widget import BasePendulumWidget

# Test catmull_rom_smooth assertion
try:
    BasePendulumWidget._catmull_rom_smooth([(0,0)]*5, n_sub=0)
    print('FAIL: n_sub=0 should have raised AssertionError')
except AssertionError:
    print('OK: n_sub assertion fires')
"
```

### 4d. Issue closure verification

Confirm the PR description references these issues and the fixes are present:

| Issue       | Fix location                 | What to verify                                                 |
| ----------- | ---------------------------- | -------------------------------------------------------------- |
| #1041       | DRY base class extraction    | `base_pendulum_widget.py` exists, both widgets inherit from it |
| #1097       | Fractional frame accumulator | `_advance_frame` uses `_anim_frac`                             |
| #1100-#1102 | Segment visibility           | `_visible_segments` in base class, synced in `_on_tab_changed` |
| #1113       | Tilt plane                   | `_draw_tilt_plane()` in base class, called in paintEvent       |
| #1115       | Real-time playback           | `frames_per_tick` uses `_sim_dt`                               |
| #1116       | Catmull-Rom trails           | `_catmull_rom_smooth()` in base class                          |
| #1118       | View azimuth                 | `_view_azimuth` in base class, used in `_world_to_pixel`       |
