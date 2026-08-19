/** Frame-explicit inputs for the prescribed manual delivery source. */

import {
  CURRENT_SIMULATION_DOCUMENT_VERSION,
  simulationDocumentFormat,
} from "./simulationDocumentFormat";

export type ShaftAxisDatum = "tracked_reference" | "generated_hosel";

export interface ManualDelivery {
  readonly manualAttackAngleDeg: number;
  readonly manualClubPathDeg: number;
  readonly manualForwardShaftLeanDeg: number;
  readonly shaftAxisDatum: ShaftAxisDatum;
}

export const MANUAL_DELIVERY_BOUNDS = Object.freeze({
  manualAttackAngleDeg: [-89, 89] as const,
  manualClubPathDeg: [-89, 89] as const,
  manualForwardShaftLeanDeg: [-60, 60] as const,
});

export const DEFAULT_MANUAL_DELIVERY: ManualDelivery = Object.freeze({
  manualAttackAngleDeg: 0,
  manualClubPathDeg: 0,
  manualForwardShaftLeanDeg: 0,
  shaftAxisDatum: "tracked_reference",
});

export const MAX_ABSOLUTE_DELIVERED_DYNAMIC_LOFT_DEG = 89;

/** Validate the rigid manual pose's delivered loft (`nominal - forward lean`). */
export function validateDeliveredDynamicLoft(
  nominalLoftDeg: number,
  forwardShaftLeanDeg: number,
): number {
  const deliveredDynamicLoftDeg = nominalLoftDeg - forwardShaftLeanDeg;
  if (!Number.isFinite(deliveredDynamicLoftDeg) ||
      Math.abs(deliveredDynamicLoftDeg) > MAX_ABSOLUTE_DELIVERED_DYNAMIC_LOFT_DEG) {
    throw new RangeError(
      `manual delivered dynamic loft ${String(deliveredDynamicLoftDeg)} deg ` +
      "must be finite and within [-89, 89] deg",
    );
  }
  return deliveredDynamicLoftDeg;
}

const finiteBounded = (
  value: unknown,
  name: keyof typeof MANUAL_DELIVERY_BOUNDS,
): number => {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new RangeError(`${name} must be finite`);
  }
  const [minimum, maximum] = MANUAL_DELIVERY_BOUNDS[name];
  if (value < minimum || value > maximum) {
    throw new RangeError(`${name} must be within [${minimum}, ${maximum}]`);
  }
  return value;
};

/** Resolve optional legacy fields to one validated, immutable contract. */
export function resolveManualDelivery(
  value: Partial<ManualDelivery>,
): ManualDelivery {
  const shaftAxisDatum = value.shaftAxisDatum ?? DEFAULT_MANUAL_DELIVERY.shaftAxisDatum;
  if (shaftAxisDatum !== "tracked_reference" && shaftAxisDatum !== "generated_hosel") {
    throw new RangeError("shaftAxisDatum must be tracked_reference or generated_hosel");
  }
  return Object.freeze({
    manualAttackAngleDeg: finiteBounded(
      value.manualAttackAngleDeg ?? DEFAULT_MANUAL_DELIVERY.manualAttackAngleDeg,
      "manualAttackAngleDeg",
    ),
    manualClubPathDeg: finiteBounded(
      value.manualClubPathDeg ?? DEFAULT_MANUAL_DELIVERY.manualClubPathDeg,
      "manualClubPathDeg",
    ),
    manualForwardShaftLeanDeg: finiteBounded(
      value.manualForwardShaftLeanDeg ?? DEFAULT_MANUAL_DELIVERY.manualForwardShaftLeanDeg,
      "manualForwardShaftLeanDeg",
    ),
    shaftAxisDatum,
  });
}

const record = (value: unknown, name: string): Record<string, unknown> => {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new TypeError(`${name} must be an object`);
  }
  return value as Record<string, unknown>;
};

const CURRENT_MANUAL_DELIVERY_FIELDS = [
  "attack_angle_deg",
  "club_path_deg",
  "forward_shaft_lean_deg",
  "shaft_axis_datum",
] as const;

/** Read manual settings from a versioned run; absent legacy fields use legacy defaults. */
export function manualDeliveryFromSimulationDocument(document: unknown): ManualDelivery {
  const root = record(document, "simulation document");
  const format = simulationDocumentFormat(root);
  const parameters = root.parameters === undefined
    ? {}
    : record(root.parameters, "parameters");
  const nested = parameters.manual_delivery === undefined
    ? null
    : record(parameters.manual_delivery, "parameters.manual_delivery");
  if (format?.version === CURRENT_SIMULATION_DOCUMENT_VERSION) {
    if (nested === null) {
      throw new Error(
        `Simulation schema version ${CURRENT_SIMULATION_DOCUMENT_VERSION} requires manual_delivery.`,
      );
    }
    const missing = CURRENT_MANUAL_DELIVERY_FIELDS.find(
      (field) => nested[field] === undefined,
    );
    if (missing !== undefined) {
      throw new Error(
        `Simulation schema version ${CURRENT_SIMULATION_DOCUMENT_VERSION} ` +
        `requires manual_delivery.${missing}.`,
      );
    }
  }
  return resolveManualDelivery({
    manualAttackAngleDeg: (nested?.attack_angle_deg ??
      parameters.manualAttackAngleDeg) as number | undefined,
    manualClubPathDeg: (nested?.club_path_deg ??
      parameters.manualClubPathDeg) as number | undefined,
    manualForwardShaftLeanDeg: (nested?.forward_shaft_lean_deg ??
      parameters.manualForwardShaftLeanDeg) as number | undefined,
    shaftAxisDatum: (nested?.shaft_axis_datum ??
      parameters.shaftAxisDatum) as ShaftAxisDatum | undefined,
  });
}
