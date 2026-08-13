"""Per-tab help content for both UIs (#4120 V4).

One tested source of truth for the '?' help panels of the PyQt6 app;
``web/src/model/helptext.ts`` carries the web wording (same tab keys
where the tabs exist on both sides). Every entry is written for a cold
user — what the tab does, a workflow, a control reference, and tips —
and the contract test asserts every tab has substantive help.
"""

from __future__ import annotations

from dataclasses import dataclass

from ._contracts import ensure

__all__ = ["HELP_TEXTS", "HelpEntry"]


@dataclass(frozen=True)
class HelpEntry:
    """Rich-text help for one tab."""

    title: str
    html: str


def _entry(title: str, html: str) -> HelpEntry:
    return HelpEntry(title=title, html=html)


#: Tab key -> help entry. Keys match the main-window tab order.
HELP_TEXTS: dict[str, HelpEntry] = {
    "capability_optimization": _entry(
        "Shot Optimizer",
        "<h3>What this tab does</h3>"
        "<p>Builds an auditable player-and-club capability profile and searches "
        "for robust launch conditions with the full Waterloo/Penner flight model. "
        "Ball speed, launch angle, launch direction, their measured variability, "
        "fixed spin, target geometry, objective, budgets, and seed are all visible.</p>"
        "<h3>Workflow</h3><ol><li>Enter the three launch centers and standard "
        "deviations, then review the explicit total-spin and spin-axis settings. "
        "Positive launch direction and spin tilt mean right/fade in the target "
        "frame.</li><li>Set the landing target, objective, candidate budget, trials "
        "per candidate, "
        "retained alternatives, and deterministic seed.</li><li>Run the optimization "
        "in the background; progress and cancellation occur at sample boundaries.</li>"
        "<li>Compare ranked alternatives, choose any parameter, flight metric, or "
        "target diagnostic for the managed scatter axes, and page through raw rows. "
        "Save/load the strict workflow or export every observation as "
        "CSV/JSON.</li></ol>"
        "<h3>Tips and limits</h3><p>The current evaluator models still-air carry "
        "to the first "
        "ground crossing. Wind, bounce, roll, and total distance are not silently "
        "included. Python uses adaptive RK45 while the web uses fixed-step RK4; both "
        "record their integrator and use tolerance-based parity rather than claiming "
        "bit-identical rankings near a tie.</p>",
    ),
    "clubhead": _entry(
        "3D Clubhead",
        "<h3>What this tab does</h3>"
        "<p>Shows an animated 3D clubhead delivering the scenario you set "
        "in the left panel: the head sweeps through impact while the "
        "face closes at your chosen rotation rates, so you can see the "
        "geometry behind every number.</p>"
        "<h3>Workflow</h3>"
        "<ol><li>Set the delivery in the left panel (speed, SPV/HTV "
        "rotation rates, lie, impact offsets) or pick a preset.</li>"
        "<li>Use the playback bar to play, pause, and scrub the impact "
        "animation; pick a speed preset for slow motion.</li>"
        "<li>Switch between head-fixed and head-moving display modes to "
        "watch either the face rotation or the delivery arc.</li>"
        "<li>Optionally load a clubhead STL or generate the procedural "
        "head from the Club picker for a realistic face.</li></ol>"
        "<h3>Tips</h3>"
        "<p>The screw-axis overlay (toggle in the view controls) draws "
        "the instantaneous axis the head is actually rotating about — "
        "the R_ISA metric in the left panel is the distance to it. "
        "Click any result row on the left for a plain-language "
        "explanation of that number.</p>",
    ),
    "plots": _entry(
        "Plots",
        "<h3>What this tab does</h3>"
        "<p>The investigative plotting suite: ready-made advanced plots "
        "(closure sweep, delivery vs impact-time, launch offset maps, "
        "swing time series, flight profiles) plus a wizard for building "
        "your own plot from any of the 40 catalogued variables.</p>"
        "<h3>Workflow</h3>"
        "<ol><li>Pick a plot in the list on the left — the canvas "
        "renders it for the current scenario (and the last Simulation "
        "run, when one exists).</li>"
        "<li>Press <b>Add Built-in</b> to add another prepared plot, or "
        "<b>Custom Plot…</b> to open the 3-step wizard: choose the data "
        "scope, pick X/Y variables from the catalog (grouped by "
        "category), then style the plot with a live preview.</li>"
        "<li>Duplicate or remove plots to curate your working set.</li>"
        "<li>Export the image (PNG/SVG), the exact plotted data "
        "(CSV/JSON), or the plot definition (.json) to reload later or "
        "open in the web app.</li></ol>"
        "<h3>Tips</h3>"
        "<p>Sweep-kind plots re-run the full swing → impact → flight "
        "simulation per grid point, so they answer 'what if' questions "
        "— e.g. carry vs impact-time offset. Run a simulation first to "
        "unlock the run-scoped variables (swing samples, launch "
        "numbers).</p>",
    ),
    "calculation_description": _entry(
        "Calculation Description",
        "<h3>What this tab does</h3>"
        "<p>The full mathematical story of the app, typeset step by "
        "step with your current numbers substituted live: the closure "
        "chain (frame conventions to path gap), the impact model "
        "(impulse-momentum with COR, effective mass, friction spin, "
        "D-plane, gear effect), the active ball-flight model with its "
        "literature citation, and — when a pendulum swing source is "
        "selected in the Simulation tab — the pendulum equations of "
        "motion.</p>"
        "<h3>Workflow</h3>"
        "<ol><li>Change any input in the left panel and watch the "
        "numeric line under each formula update.</li>"
        "<li>Switch the flight model or swing source in the Simulation "
        "tab — the matching sections here rewrite themselves to the "
        "active configuration.</li>"
        "<li>Scroll: sections are ordered the way the physics runs "
        "(delivery → impact → flight).</li></ol>"
        "<h3>Tips</h3>"
        "<p>Every result row's explanation names the step that produces "
        "it, and the Glossary tab defines every symbol and term used "
        "here. Formulas are selectable text plus rendered math, so you "
        "can trace any reported number to its source equation.</p>",
    ),
    "simulation": _entry(
        "Simulation",
        "<h3>What this tab does</h3>"
        "<p>Runs the full swing → impact → flight pipeline: pick a "
        "swing source (manual constant twist, double pendulum, or "
        "triple pendulum), orient the swing plane, choose a club and a "
        "flight model, and press Run. The result is one coherent "
        "simulation with launch numbers, three scale-separated viewers, "
        "an inspector, and a goal-driven solver.</p>"
        "<h3>Workflow</h3>"
        "<ol><li>Choose the swing source and set the three plane tilts "
        "(hover any input for its suggested range and source).</li>"
        "<li>Pick a club (drives the head model) and a flight model "
        "(all 7 literature models available).</li>"
        "<li>Press <b>Run Simulation</b>, then scrub the impact-time "
        "slider to move impact earlier or later in the swing — the "
        "delivery readout updates live and the run recomputes on "
        "release.</li>"
        "<li>Read the launch rows (click one for its explanation) and "
        "explore the Strike (face scale), Swing (metres), Kinetics "
        "(joint torques/powers/forces of the pendulum swing), and "
        "Flight (tens of metres) viewer sub-tabs.</li>"
        "<li>Use the Inspector for the raw run data (CSV/JSON export) "
        "and the Solver to find deliveries that hit goal launch "
        "numbers.</li></ol>"
        "<h3>Tips</h3>"
        "<p>'Auto' places impact at maximum clubhead speed. The Swing "
        "viewer keeps the scene at swing scale — enable 'Show Ball "
        "Flight' only when you want the full flight envelope (it dwarfs "
        "the swing).</p>",
    ),
    "flight_explorer": _entry(
        "Flight Explorer",
        "<h3>What this tab does</h3>"
        "<p>A standalone ball-flight laboratory — no swing required. "
        "Enter launch conditions directly (ball speed, launch angle, "
        "azimuth, spin, spin-axis tilt) or describe an impact delivery "
        "and let the impact model derive the launch, then integrate the "
        "flight with any of the 7 literature models.</p>"
        "<h3>Workflow</h3>"
        "<ol><li>Pick the entry mode: <b>Direct launch</b> (launch-"
        "monitor style numbers) or <b>Impact delivery</b> (club path, "
        "face, AoA, loft, offsets through the rigid-body impact "
        "solve).</li>"
        "<li>Type the values — every field's tooltip gives its typical "
        "range and source; the speed field has a mph / m/s unit "
        "drop-down.</li>"
        "<li>Choose a flight model and press <b>Run Flight</b>.</li>"
        "<li>Read the result rows (click for explanations) and the "
        "side / top-down / 3D trajectory views.</li></ol>"
        "<h3>Tips</h3>"
        "<p>Sign conventions match launch monitors: positive azimuth = "
        "right of target, positive spin-axis tilt = fade side. Compare "
        "models by re-running with a different model at identical "
        "launch conditions.</p>",
    ),
    "regional_surfaces": _entry(
        "Ground Surfaces",
        "<h3>What this tab does</h3>"
        "<p>Builds a strict versioned request for one flat, stationary SI base "
        "surface and bounded coplanar material overlays. Every material value, "
        "interval, identity, precedence, source revision, and qualification is "
        "visible before validation.</p>"
        "<h3>Workflow</h3><ol><li>Replace the clearly marked illustrative base "
        "material with traceable values and identify the source revision.</li>"
        "<li>Add up to eight overlays, keeping each metre interval inside the "
        "base domain and every region, precedence, and surface ID unique.</li>"
        "<li>Validate and inspect the canonical request readback. Errors preserve "
        "the draft so the reported field can be corrected.</li></ol>"
        "<h3>Tips and qualification</h3><p>This first slice is an unvalidated, "
        "session-only "
        "request editor. It does not execute ground physics or playback, and it "
        "does not persist model inputs. Geometry is fixed to the target frame as "
        "flat, static, and coplanar because those are the regional v1 limits.</p>",
    ),
    "launch_monitor_analytics": _entry(
        "Launch Monitor Analytics",
        "<h3>What this tab does</h3>"
        "<p>Imports local CSV or JSON launch-monitor records without dropping "
        "source columns, then runs flexible correlation, multivariable ordinary "
        "least-squares regression, uncertainty, residual, and grouped analysis. "
        "The built-in demonstration data lets you inspect the workflow before "
        "loading a file.</p>"
        "<h3>Workflow</h3>"
        "<ol><li>Import data and confirm the retained row and column counts.</li>"
        "<li>Choose the documented interpretation convention, outcome, any "
        "number of predictors, missing-data policy, method, and grouping.</li>"
        "<li>Run the analysis and inspect pair counts, corrected p-values, OLS "
        "coefficient intervals, fit metrics, residual diagnostics, group results, "
        "and the SHA-256 dataset fingerprint.</li>"
        "<li>Export retained records and the complete analysis JSON.</li></ol>"
        "<h3>Tips and Scientific Boundary</h3>"
        "<p>Association and predictive fit do not establish causality. Aggregate "
        "records never enter regression. TrackMan-Comparable and Foresight-"
        "Comparable describe sourced definition frames; they do not emulate or "
        "certify vendor devices.</p>",
    ),
    "variation": _entry(
        "Variation",
        "<h3>What this tab does</h3>"
        "<p>Monte-Carlo variation studies: give any set of inputs a "
        "noise distribution, run the pipeline N times with a fixed "
        "seed, and read dispersion statistics, sensitivities, and the "
        "2σ landing ellipse — which inputs actually drive your "
        "outcomes?</p>"
        "<h3>Workflow</h3>"
        "<ol><li>Pick the pipeline mode (delivery → impact → flight, "
        "swing → impact → flight, or launch → flight) and the base "
        "scenario source.</li>"
        "<li>Add noise rows: choose a variable from the registry "
        "(grouped by category), a distribution (normal / uniform / "
        "triangular), a scale in the variable's own unit, and optional "
        "clipping.</li>"
        "<li>Set the run count and seed, then press <b>Run</b> — "
        "progress is live and Cancel is safe.</li>"
        "<li>Read the results tabs: summary statistics, one-at-a-time "
        "sensitivity heat table, Spearman correlations, and the landing "
        "scatter with its 2σ ellipse.</li>"
        "<li>Export the dataset (CSV/JSON) or save the plan to rerun "
        "the exact study later.</li></ol>"
        "<h3>Tips</h3>"
        "<p>The same seed always reproduces the same dataset (per-"
        "variable streams), and plans saved here load in the web app. "
        "Failed runs are recorded as NaN rows, never aborting the "
        "batch.</p>",
    ),
    "putting": _entry(
        "Putting",
        "<h3>What this tab does</h3>"
        "<p>A putting laboratory on a uniform sloped green: pick a "
        "putter, set the stroke pace (directly or as a pendulum "
        "backstroke length), dial in the green speed (stimp) and the "
        "slope, and read the full story of the putt — roll-out, the "
        "skid-to-roll transition, time, break, and whether the ball "
        "drops or how far it misses.</p>"
        "<h3>Workflow</h3>"
        "<ol><li>Choose a putter (the club-library putter by default) "
        "and how you want to set the pace: clubhead speed at impact, "
        "or a backstroke length through the pendulum-stroke "
        "proxy.</li>"
        "<li>Set the green: stimp (6 slow — 14 tournament fast), "
        "slope grade in percent, and the downhill direction relative "
        "to the putt line (+90° = low side on your left).</li>"
        "<li>Set the distance to the hole and read the result rows — "
        "click any row for its plain-language explanation with "
        "glossary links.</li>"
        "<li>Watch the top-down green view: the skid phase and the "
        "pure-roll phase are colour-coded along the path, the arrow "
        "shows the downhill direction, and the speed-vs-distance "
        "plot marks the capture-speed bound at the hole.</li></ol>"
        "<h3>Tips</h3>"
        "<p>A putt only drops if it crosses the hole at or below the "
        "capture speed (the ball must fall half its diameter while "
        "crossing the lip) — dying the ball in beats charging it. "
        "On a cross-slope, note how most of the break happens in the "
        "last third of the putt, where the ball is slowest.</p>",
    ),
    "glossary": _entry(
        "Glossary",
        "<h3>What this tab does</h3>"
        "<p>Defines every technical term used across the app — delivery "
        "terms (path, face, AoA, dynamic loft, spin loft, D-plane), "
        "closure metrics (CCV, HTV, SPV, R_ISA), impact physics (COR, "
        "effective mass, MOI, gear effect, bulge/roll), flight terms, "
        "and the variation-study vocabulary — each with a sourced 1-3 "
        "sentence definition.</p>"
        "<h3>Workflow</h3>"
        "<ol><li>Type in the search box to filter; matching covers both "
        "term names and definition text.</li>"
        "<li>Click a term to read its definition in the pane on the "
        "right.</li>"
        "<li>Or arrive via any explanation panel's <b>Glossary</b> link "
        "— the matching term is pre-selected.</li></ol>"
        "<h3>Tips</h3>"
        "<p>Each definition names its source (the AffineDrift review, "
        "the Cheetham 2014 dossier, or the swing_sim module that "
        "implements the physics), so you can go one level deeper when "
        "you need the full derivation.</p>",
    ),
}


def _validate() -> None:
    ensure(len(HELP_TEXTS) >= 7, "every tab must have help")
    for key, entry in HELP_TEXTS.items():
        ensure(len(entry.html) > 300, f"help for {key} must be substantive")
        ensure(bool(entry.title.strip()), f"help for {key} must be titled")


_validate()
