/** Deterministic UI defaults for the authority-backed Morris workflow. */

import { getClub, type ClubSpec } from "./club";
import type { ImpactScenario } from "./impact";
import {
  type MorrisAuthorityBase,
} from "./morrisAuthorityRequest";

const DRIVER_TEE_HEIGHT_M = 0.0381;

const authorityClubValue = (club: ClubSpec, field: keyof ClubSpec): unknown => (
  field === "headStyle" ? club.headStyle ?? "Auto" : club[field]
);

const assertCanonicalAuthorityClub = (club: ClubSpec): void => {
  let canonical: ClubSpec;
  try {
    canonical = getClub(club.name);
  } catch {
    throw new RangeError(
      `Morris authority requires a canonical library club; ${club.name} is unsupported`,
    );
  }
  const incompatibleField = (Object.keys(canonical) as (keyof ClubSpec)[]).find(
    (field) => authorityClubValue(club, field) !== authorityClubValue(canonical, field),
  );
  if (incompatibleField !== undefined) {
    throw new RangeError(
      `Morris authority cannot represent custom club field ${incompatibleField}; `
      + `select the canonical ${club.name} specification or restore its library value`,
    );
  }
};

export function defaultMorrisAuthorityBase(
  club: ClubSpec,
  scenario: ImpactScenario,
): MorrisAuthorityBase {
  assertCanonicalAuthorityClub(club);
  const pinnedScenario = {
    clubheadSpeedMph: 113, omegaPlaneDps: 1870, omegaShaftDps: 1307,
    lieAngleDeg: 58, comToFaceMm: 40, contactDurationUs: 450,
  };
  const incompatible = Object.entries(pinnedScenario).find(([key, value]) => (
    scenario[key as keyof ImpactScenario] !== value
  ));
  if (incompatible !== undefined) {
    throw new RangeError(
      `Morris authority requires pinned ${incompatible[0]}=${incompatible[1]}; `
      + `current value is unsupported (${String(scenario[incompatible[0] as keyof ImpactScenario])}). `
      + "Restore the pinned value in Simulation setup",
    );
  }
  const supportMode = club.clubType === "Driver" ? "tee" : "ground";
  return Object.freeze({
    clubName: club.name,
    supportMode,
    teeHeightM: supportMode === "tee" ? DRIVER_TEE_HEIGHT_M : 0,
    planeYawDeg: 0,
    planeSideTiltDeg: -45,
    planeForwardTiltDeg: 0,
    pendulumM1Kg: 7.5,
    pendulumL1M: 0.75,
    pendulumLc1M: 0.3375,
    pendulumI1KgM2: 1.205859375,
    pendulumM2Kg: 0.35,
    pendulumL2M: 1,
    pendulumLc2M: 0.7557142857142858,
    pendulumI2KgM2: 0.240235,
    dampingShoulder: 0.4,
    dampingWrist: 0.25,
    swingDurationS: 0.05,
    flightModel: "waterloo_penner",
    impactOffsetToeMm: scenario.impactOffsetToeMm,
    impactOffsetHighMm: scenario.impactOffsetHighMm,
  });
}
