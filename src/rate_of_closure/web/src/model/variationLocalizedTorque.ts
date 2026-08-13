/** Map authored localized variation loci into executable double-pendulum commands. */

import {
  PASSIVE_DOUBLE_PENDULUM_RUN,
  passiveDoublePendulumRun,
  prescribedDoublePendulumRun,
  type DoublePendulumRunConfig,
} from "./doublePendulum";
import {
  LOCALIZED_TORQUE_PROVENANCE,
  LOCALIZED_TORQUE_UNIT,
  type LocalizedTorqueCommandTs,
} from "./localizedTorque";
import { localizedTorqueJointId, variableDef } from "./variationRegistry";
import { isGlobalSpec, stableSpecId, type VariationPlanTs } from "./variationSchema";

export const LOCALIZED_TORQUE_EXECUTION_CAPABILITY = Object.freeze({
  sourceKind: "double_pendulum" as const,
  backend: "typescript_reference_rk4" as const,
  joints: Object.freeze(["joint.shoulder", "joint.wrist"] as const),
  semantics: "additive_half_open_every_rk4_substep" as const,
});

export interface LocalizedTorqueExecutionTs {
  readonly runConfig: DoublePendulumRunConfig;
  readonly commands: readonly LocalizedTorqueCommandTs[];
}

export function localizedTorqueExecution(
  plan: VariationPlanTs,
  sampledValues: Readonly<Record<string, number>>,
  durationS: number,
  sourceKind: string,
  baseConfig: DoublePendulumRunConfig = PASSIVE_DOUBLE_PENDULUM_RUN,
): LocalizedTorqueExecutionTs {
  const localizedSpecs = plan.noise.filter((spec) => !isGlobalSpec(spec));
  if (localizedSpecs.length === 0) {
    return { runConfig: baseConfig, commands: Object.freeze([]) };
  }
  if (sourceKind !== LOCALIZED_TORQUE_EXECUTION_CAPABILITY.sourceKind) {
    throw new Error("localized torque execution requires the double_pendulum source");
  }
  const commands = localizedSpecs.map((spec): LocalizedTorqueCommandTs => {
    const jointId = localizedTorqueJointId(spec.variableKey);
    if (jointId === null || spec.timeWindowS == null || spec.pointIds?.[0] !== jointId) {
      throw new Error(`unsupported localized variation locus: ${stableSpecId(spec)}`);
    }
    const torqueNm = sampledValues[spec.variableKey];
    if (!Number.isFinite(torqueNm)) {
      throw new Error(`localized torque sample must be finite: ${stableSpecId(spec)}`);
    }
    const [start, end] = spec.timeWindowS;
    if (!(0 <= start && start < end && end <= durationS)) {
      throw new Error(`localized torque window exceeds run: ${stableSpecId(spec)}`);
    }
    if (variableDef(spec.variableKey)?.unit !== "N·m") {
      throw new Error(`localized torque registry unit mismatch: ${spec.variableKey}`);
    }
    return Object.freeze({
      specId: stableSpecId(spec),
      variableKey: spec.variableKey,
      jointId,
      timeWindowS: Object.freeze([start, end]) as readonly [number, number],
      torqueNm,
      unit: LOCALIZED_TORQUE_UNIT,
      provenance: LOCALIZED_TORQUE_PROVENANCE,
    });
  });
  const offsets = commands.map(({ jointId, timeWindowS, torqueNm }) =>
    Object.freeze({ jointId, timeWindowS, torqueNm }));
  const runConfig = baseConfig.mode === "prescribed"
    ? prescribedDoublePendulumRun(
        baseConfig.profile,
        baseConfig.jointLocks,
        [...(baseConfig.commandedTorqueOffsets ?? []), ...offsets],
      )
    : passiveDoublePendulumRun(
        baseConfig.jointLocks,
        [...(baseConfig.commandedTorqueOffsets ?? []), ...offsets],
      );
  return { runConfig, commands: Object.freeze(commands) };
}
