/** Request-bound default and sampled input mapping for browser swing variation. */

import { DRIVER_TEE_HEIGHT_M, type BallSetup } from "./ballSetup";
import { effectiveDoublePendulumDurationS, golfDefaultParams } from "./doublePendulum";
import { DEFAULT_IMPACT_CLUB, type SimulationInput } from "./simulation";
import { type VariationPlanTs } from "./variationSchema";
import {
  CATEGORY_CLUB, CATEGORY_DELIVERY, CATEGORY_SWING, TEE_HEIGHT_VARIATION_KEY,
} from "./variationRegistry";
import {
  localizedTorqueExecution, type LocalizedTorqueExecutionTs,
} from "./variationLocalizedTorque";

const key = (category: string, name: string): string => `${category}.${name}`;
const YAW = key(CATEGORY_SWING, "yaw_deg");
const SIDE_TILT = key(CATEGORY_SWING, "side_tilt_deg");
const FORWARD_TILT = key(CATEGORY_SWING, "forward_tilt_deg");
const IMPACT_TIME_OFFSET = key(CATEGORY_SWING, "impact_time_offset_s");
const DAMPING_SHOULDER = key(CATEGORY_SWING, "damping_shoulder");
const DAMPING_WRIST = key(CATEGORY_SWING, "damping_wrist");
const TOE_OFFSET = key(CATEGORY_DELIVERY, "impact_offset_toe_mm");
const HIGH_OFFSET = key(CATEGORY_DELIVERY, "impact_offset_high_mm");
const HEAD_MASS = key(CATEGORY_CLUB, "head_mass_kg");
const HEAD_MOI = key(CATEGORY_CLUB, "head_moi_kg_m2");
const COR = key(CATEGORY_CLUB, "cor");

export function defaultSwingVariationInput(ballSetup?: BallSetup): SimulationInput {
  return {
    sourceKind: "double_pendulum", clubheadSpeedMph: 30, omegaDps: [0, 0, 0],
    loftDeg: 10.5, impactOffsetToeMm: 0, impactOffsetHighMm: 0,
    planeYawDeg: 0, planeSideTiltDeg: -45, planeForwardTiltDeg: 0,
    impactTimeS: null, impactTimeOffsetS: 0, swingDurationS: 1.5,
    pendulumParameters: golfDefaultParams(), club: { ...DEFAULT_IMPACT_CLUB },
    ballSetup: ballSetup ?? { supportMode: "tee", teeHeightM: DRIVER_TEE_HEIGHT_M },
  };
}

export function swingVariationInputForValues(
  plan: VariationPlanTs,
  values: Readonly<Record<string, number>>,
  base: SimulationInput = defaultSwingVariationInput(plan.ballSetup),
): { input: SimulationInput; localized: LocalizedTorqueExecutionTs } {
  const parameters = base.pendulumParameters ?? golfDefaultParams();
  const setup = base.ballSetup ?? { supportMode: "ground" as const, teeHeightM: 0 };
  const teeHeight = values[TEE_HEIGHT_VARIATION_KEY];
  if (teeHeight !== undefined && setup.supportMode !== "tee") {
    throw new Error("Tee Height variation requires Tee support");
  }
  const input: SimulationInput = {
    ...base,
    planeYawDeg: values[YAW] ?? base.planeYawDeg,
    planeSideTiltDeg: values[SIDE_TILT] ?? base.planeSideTiltDeg,
    planeForwardTiltDeg: values[FORWARD_TILT] ?? base.planeForwardTiltDeg,
    impactTimeOffsetS: values[IMPACT_TIME_OFFSET] ?? base.impactTimeOffsetS ?? 0,
    impactOffsetToeMm: values[TOE_OFFSET] ?? base.impactOffsetToeMm,
    impactOffsetHighMm: values[HIGH_OFFSET] ?? base.impactOffsetHighMm,
    pendulumParameters: {
      ...parameters, d1: values[DAMPING_SHOULDER] ?? parameters.d1,
      d2: values[DAMPING_WRIST] ?? parameters.d2,
    },
    club: {
      headMassKg: values[HEAD_MASS] ?? base.club?.headMassKg ?? DEFAULT_IMPACT_CLUB.headMassKg,
      moiAboutShaftKgM2: values[HEAD_MOI] ?? base.club?.moiAboutShaftKgM2 ?? DEFAULT_IMPACT_CLUB.moiAboutShaftKgM2,
      coefficientOfRestitution: values[COR] ?? base.club?.coefficientOfRestitution ?? DEFAULT_IMPACT_CLUB.coefficientOfRestitution,
    },
    ballSetup: teeHeight === undefined ? setup : { supportMode: "tee", teeHeightM: teeHeight },
  };
  const localized = localizedTorqueExecution(
    plan, values, effectiveDoublePendulumDurationS(input.swingDurationS),
    input.sourceKind, input.doublePendulumRun,
  );
  return { input: { ...input, doublePendulumRun: localized.runConfig }, localized };
}
