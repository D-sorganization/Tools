/**
 * Glossary entry data for the Putting tab (#4125 H3) — see glossary.ts.
 *
 * Additive split module (500-LOC file budget pattern). Mirrors
 * `src/rate_of_closure/glossary_entries_putting.py` key-for-key.
 */

import type { GlossaryEntry } from "./glossaryTypes";

export const ENTRIES: Record<string, GlossaryEntry> = {
  break: {
    term: "Break",
    definition:
      "The lateral drift of a putt off its starting line, driven by the " +
      "in-plane component of gravity on a sloped green. Most of the break " +
      "accumulates late in the putt, when the ball is slowest " +
      "(swing_sim.putting.green derivation).",
  },
  capture_speed: {
    term: "Capture Speed",
    definition:
      "The fastest a ball can cross the hole and still drop: it must fall " +
      "half its diameter while traversing the opening, giving roughly " +
      "0.8-1.6 m/s depending on the crossing chord (geometric derivation " +
      "in swing_sim.putting.green; Holmes, Am. J. Phys. 59, 1991).",
  },
  pure_roll: {
    term: "Pure Roll",
    definition:
      "The rolling state where the ball's contact point is momentarily at " +
      "rest — forward speed equals surface spin speed (v = ωr). From here " +
      "only rolling resistance, set by the green speed, slows the ball " +
      "(swing_sim.putting.roll derivation).",
  },
  skid: {
    term: "Skid Phase",
    definition:
      "The opening stretch of a putt where the ball slides rather than " +
      "rolls: struck with backspin, it is decelerated and spun up by " +
      "sliding friction until v = ωr. With no initial spin the classic " +
      "result is pure roll at 5/7 of launch speed " +
      "(swing_sim.putting.roll derivation).",
  },
  stimp: {
    term: "Stimp (Green Speed)",
    definition:
      "The stimpmeter reading in feet: how far a ball rolls out when " +
      "released from the USGA's 36-inch ramp at 20 degrees (~1.83 m/s). " +
      "Inverting the roll-out formula maps stimp to the green's rolling " +
      "resistance, mu = v²/(2gS) (swing_sim.putting.roll derivation).",
  },
};
