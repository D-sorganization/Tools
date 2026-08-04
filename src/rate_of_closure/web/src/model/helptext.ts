/**
 * Per-tab help content (#4120 V4) — the "How to Use This Page"
 * sections. Written for someone who arrives with ZERO context: what
 * the page does, a workflow, and tips. The vitest contract test
 * asserts every tab has substantive help (>300 characters).
 */

export interface HelpEntry {
  title: string;
  /** Plain-text paragraphs (rendered as separate <p> blocks). */
  paragraphs: string[];
}

/** Tab name -> help entry (keys match the App TABS labels). */
export const HELP_TEXTS: Record<string, HelpEntry> = {
  Explorer: {
    title: "How to Use This Page",
    paragraphs: [
      "A rotating clubhead is a rigid body, so the point that strikes " +
        "the ball moves in a different direction than the tracked " +
        "reference point. This page quantifies that gap. Set the " +
        "delivery on the left — clubhead speed, the two rotation rates " +
        "(in-plane SPV and about-shaft HTV), shaft lie, and where on " +
        "the face the ball is struck. Hover any input for its typical " +
        "range and source; the unit drop-downs convert everything in " +
        "place.",
      "Read the results on the right: the deviation rows show how far " +
        "the impact point's path and attack angle differ from the " +
        "reported delivery, and the closure metrics translate the same " +
        "rotation into every framing the literature uses (CCV deg/s, " +
        "deg/ft, R_ISA, time-to-square). Click any row — it highlights " +
        "and its plain-language explanation appears below, with a " +
        "Glossary link for every technical term. The 3D clubhead " +
        "animates your exact scenario.",
    ],
  },
  "Calculation Description": {
    title: "How to Use This Page",
    paragraphs: [
      "This page is the full mathematical story behind every number in " +
        "the app, typeset step by step with your current inputs " +
        "substituted live into each formula. It is organised in " +
        "sections that follow the physics: the closure chain (frame " +
        "conventions to the path gap), the impact model (impulse-" +
        "momentum with COR, effective mass, the friction spin cap, the " +
        "D-plane, gear effect), and the active ball-flight model with " +
        "its literature citation.",
      "Change any input on the Explorer page and return here — the " +
        "numeric line under each formula updates to match. Unfamiliar " +
        "terms (spin loft, R_ISA, Coriolis…) are all defined in the " +
        "Glossary page.",
    ],
  },
  Simulation: {
    title: "How to Use This Page",
    paragraphs: [
      "This page runs a complete swing → impact → flight simulation in " +
        "the browser. Press Run Simulation to generate the swing, " +
        "solve the impact at the chosen instant, and integrate the " +
        "ball flight; the launch numbers (ball speed, launch angle, " +
        "spin, carry…) appear as rows on the left.",
      "Use the impact-time slider to scrub impact earlier or later in " +
        "the swing — the delivery readout updates live. The Strike / " +
        "Swing / Flight buttons switch between the face-scale impact " +
        "zone (with the delivered path/face/AoA vectors), the swing-" +
        "scale scene with playback, and the flight profiles (side and " +
        "top-down). 'Show Ball Flight' expands the swing scene to " +
        "flight scale — expect the swing to look tiny. The Solver " +
        "section searches for deliveries that hit goal launch numbers, " +
        "and the JSON download captures the whole run.",
    ],
  },
  Plots: {
    title: "How to Use This Page",
    paragraphs: [
      "An investigative plotting workbench. Pick one of the built-in " +
        "advanced plots (closure sweep, launch-vs-offset maps, flight " +
        "profiles…) — it renders immediately for the scenario set on " +
        "the Explorer page — or build your own plot by choosing X and " +
        "Y variables from the catalog.",
      "Sweep plots re-run the simulation across a grid of the X " +
        "variable, so they answer 'what happens to carry if I change " +
        "the impact time?' style questions. Export the image as PNG, " +
        "the plotted data as CSV/JSON, or the plot definition as a " +
        ".json file that also loads in the desktop app.",
    ],
  },
  "Flight Explorer": {
    title: "How to Use This Page",
    paragraphs: [
      "A standalone ball-flight calculator — no swing or impact " +
        "needed. Type launch-monitor style numbers (ball speed with a " +
        "mph / m/s unit picker, launch angle, azimuth, spin rate, and " +
        "spin-axis tilt) and press Run Flight to integrate the " +
        "trajectory with the Waterloo/Penner aerodynamics model.",
      "The result rows give carry, apex, flight time, landing angle, " +
        "and the lateral landing offset; the canvases show the side " +
        "profile and top-down view with the landing point annotated. " +
        "Signs follow launch-monitor conventions: positive azimuth and " +
        "positive spin-axis tilt both mean right of target (fade side " +
        "for a right-handed player). Hover any field for its typical " +
        "range and source.",
    ],
  },
  Variation: {
    title: "How to Use This Page",
    paragraphs: [
      "Monte-Carlo variation studies: discover how sensitive your " +
        "outcomes are to input scatter. Choose a pipeline mode, then " +
        "add noise rows — each row picks a variable, a distribution " +
        "(normal, uniform, or triangular), and a scale in the " +
        "variable's own unit. Set the number of runs and a seed, then " +
        "press Run.",
      "The same plan + seed always reproduces the same dataset, and " +
        "plan files are interchangeable with the desktop app. Results " +
        "include summary statistics for every output, a sensitivity " +
        "table showing which inputs drive which outputs, and the " +
        "landing scatter with its 2σ dispersion ellipse (roughly the " +
        "95% landing zone). Export the dataset as CSV or JSON for " +
        "further analysis.",
    ],
  },
  Glossary: {
    title: "How to Use This Page",
    paragraphs: [
      "Every technical term used across the app, defined in 1-3 " +
        "sentences with its source — delivery terms (club path, face " +
        "angle, dynamic loft, spin loft, D-plane), closure metrics " +
        "(CCV, HTV, SPV, R_ISA), impact physics (COR, effective mass, " +
        "MOI, gear effect, bulge/roll), flight aerodynamics, pendulum " +
        "dynamics, and the Monte-Carlo vocabulary.",
      "Type in the search box to filter — matching covers both the " +
        "term names and the definition text — then click a term to " +
        "read its definition. Explanation cards across the app link " +
        "straight here with the relevant term pre-selected.",
    ],
  },
};
