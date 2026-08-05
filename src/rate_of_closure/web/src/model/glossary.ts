/**
 * Glossary of every technical term used across the app (#4120 V4).
 *
 * TypeScript mirror of `src/rate_of_closure/glossary.py` — same keys,
 * same terms, same sourced definitions. The vitest parity test pins the
 * key list so the two glossaries cannot drift apart. Entry data lives
 * in the split glossaryEntries*.ts modules (500-LOC file budget).
 */

import { ENTRIES as ENTRIES_A_L } from "./glossaryEntriesAL";
import { ENTRIES as ENTRIES_L_Z } from "./glossaryEntriesLZ";
import { ENTRIES as ENTRIES_PUTTING } from "./glossaryEntriesPutting";
import type { GlossaryEntry } from "./glossaryTypes";

export type { GlossaryEntry } from "./glossaryTypes";

/** Every term used across the app, keyed snake_case (Python parity). */
export const GLOSSARY: Record<string, GlossaryEntry> = Object.fromEntries(
  Object.entries({ ...ENTRIES_A_L, ...ENTRIES_L_Z, ...ENTRIES_PUTTING }).sort(([a], [b]) =>
    a < b ? -1 : a > b ? 1 : 0,
  ),
);

/** Explanation field (camelCase, web keys) -> glossary term. */
export const FIELD_TO_TERM: Record<string, string> = {
  pathDeviationDeg: "club_path",
  aoaDeviationDeg: "attack_angle",
  tangentialSpeedMph: "twist",
  speedDeltaMph: "twist",
  closureRateDps: "ccv",
  normalizedClosureDegPerFt: "r_isa",
  closureDuringContactDeg: "contact_duration",
  loftGainDuringContactDeg: "dynamic_loft",
  ccvDps: "ccv",
  closureDegPerFt: "r_isa",
  closureDegPerInch: "closure_rate",
  closureDegPerMs: "closure_rate",
  rIsaM: "r_isa",
  rIsaFt: "r_isa",
  timeToSquareFrom1DegOpenMs: "time_to_square",
  toeHeelSpeedDeltaMph: "lever_arm",
  ballSpeedMph: "smash_factor",
  launchAngleDeg: "launch_angle",
  launchAzimuthDeg: "launch_azimuth",
  spinRpm: "spin_rate",
  carryM: "carry",
  maxHeightM: "apex",
  flightTimeS: "flight_time",
  landingAngleDeg: "landing_angle",
  lateralM: "lateral_offset",
  // Kinetics panel fields (#4125 H2)
  jointTorques: "inverse_dynamics",
  jointPower: "power",
  reactionForces: "joint_reaction_force",
  zeroTorqueCounterfactual: "zero_torque_counterfactual",
  puttRolloutM: "stimp",
  puttSkidM: "skid",
  puttSkidPct: "skid",
  puttTimeS: "pure_roll",
  puttBreakM: "break",
  puttSpeedAtHoleMps: "capture_speed",
  puttMargin: "capture_speed",
};

/** Glossary keys whose term or definition matches `query`. */
export function searchTerms(query: string): string[] {
  const needle = query.trim().toLowerCase();
  return Object.keys(GLOSSARY).filter(
    (key) =>
      !needle ||
      GLOSSARY[key].term.toLowerCase().includes(needle) ||
      GLOSSARY[key].definition.toLowerCase().includes(needle),
  );
}
