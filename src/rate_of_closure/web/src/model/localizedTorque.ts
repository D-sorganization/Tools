/** Strict localized commanded-torque contracts shared by swing execution/export. */

import { SHOULDER_JOINT_ID, WRIST_JOINT_ID } from "./jointLocks";

export const LOCALIZED_TORQUE_UNIT = "N*m";
export const LOCALIZED_TORQUE_PROVENANCE =
  "variation_plan.v2:additive_commanded_torque";

export interface LocalizedTorqueOffsetTs {
  readonly jointId: typeof SHOULDER_JOINT_ID | typeof WRIST_JOINT_ID;
  readonly timeWindowS: readonly [number, number];
  readonly torqueNm: number;
}

export interface LocalizedTorqueCommandTs extends LocalizedTorqueOffsetTs {
  readonly specId: string;
  readonly variableKey: string;
  readonly unit: typeof LOCALIZED_TORQUE_UNIT;
  readonly provenance: typeof LOCALIZED_TORQUE_PROVENANCE;
}

const finiteNumber = (value: unknown, label: string): number => {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`${label} must be a finite number`);
  }
  return value;
};

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

export function normalizeLocalizedTorqueOffsets(
  value: unknown,
  durationS?: number,
): readonly LocalizedTorqueOffsetTs[] {
  if (!Array.isArray(value)) throw new Error("localized torque offsets must be an array");
  const duration = durationS === undefined
    ? null
    : finiteNumber(durationS, "localized torque run duration");
  if (duration !== null && !(duration > 0)) {
    throw new Error("localized torque run duration must be > 0");
  }
  return Object.freeze(value.map((raw): LocalizedTorqueOffsetTs => {
    if (!isRecord(raw)) throw new Error("localized torque offset must be an object");
    const jointId = raw.jointId;
    if (jointId !== SHOULDER_JOINT_ID && jointId !== WRIST_JOINT_ID) {
      throw new Error("localized torque jointId must be joint.shoulder or joint.wrist");
    }
    if (!Array.isArray(raw.timeWindowS) || raw.timeWindowS.length !== 2) {
      throw new Error("localized torque timeWindowS must contain two values");
    }
    const start = finiteNumber(raw.timeWindowS[0], "localized torque window start");
    const end = finiteNumber(raw.timeWindowS[1], "localized torque window end");
    if (!(0 <= start && start < end) || (duration !== null && end > duration)) {
      throw new Error("localized torque window must satisfy 0 <= start < end <= duration");
    }
    return Object.freeze({
      jointId,
      timeWindowS: Object.freeze([start, end]) as readonly [number, number],
      torqueNm: finiteNumber(raw.torqueNm, "localized torque offset"),
    });
  }));
}

export function addLocalizedTorqueOffsets(
  base: readonly [number, number],
  offsets: readonly LocalizedTorqueOffsetTs[],
  timeS: number,
): readonly [number, number] {
  const time = finiteNumber(timeS, "localized torque sample time");
  const torques = [
    finiteNumber(base[0], "base shoulder torque"),
    finiteNumber(base[1], "base wrist torque"),
  ];
  offsets.forEach((offset) => {
    const [start, end] = offset.timeWindowS;
    if (start <= time && time < end) {
      const index = offset.jointId === SHOULDER_JOINT_ID ? 0 : 1;
      torques[index] += offset.torqueNm;
    }
  });
  return torques as [number, number];
}
