/** Registry-backed launch-direction conversion and lossless legacy migration. */

import {
  conventionRegistry,
  type ConventionId,
  type ParameterDefinitionTs,
} from "./launchMonitorConventions";

const SUPPORTED_DIRECTION_CONVENTIONS = [
  "app_native",
  "trackman_comparable",
] as const satisfies readonly ConventionId[];
export type LaunchDirectionConvention =
  (typeof SUPPORTED_DIRECTION_CONVENTIONS)[number];

export interface LaunchDirectionValue {
  degrees: number;
  convention: LaunchDirectionConvention;
}

const registry = conventionRegistry();

export const LAUNCH_DIRECTION_DEFINITIONS: Readonly<
  Record<LaunchDirectionConvention, ParameterDefinitionTs>
> = Object.freeze(
  Object.fromEntries(
    SUPPORTED_DIRECTION_CONVENTIONS.map((convention) => [
      convention,
      registry.definition(convention, "launch_direction"),
    ]),
  ) as Record<LaunchDirectionConvention, ParameterDefinitionTs>,
);

const LEGACY_CONVENTIONS: Readonly<Record<string, LaunchDirectionConvention>> =
  Object.freeze({ launch_monitor_comparable: "trackman_comparable" });

function validatedDegrees(value: unknown): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new TypeError("launch direction must be a finite number");
  }
  if (value < -180 || value > 180) {
    throw new RangeError("launch direction must be within [-180, 180] degrees");
  }
  return value;
}

function validatedConvention(value: unknown): LaunchDirectionConvention {
  if (typeof value !== "string") {
    throw new RangeError(`unknown launch-direction convention: ${String(value)}`);
  }
  const canonical = LEGACY_CONVENTIONS[value] ?? value;
  if (
    !SUPPORTED_DIRECTION_CONVENTIONS.includes(
      canonical as LaunchDirectionConvention,
    )
  ) {
    throw new RangeError(`unknown launch-direction convention: ${String(value)}`);
  }
  const convention = canonical as LaunchDirectionConvention;
  const definition = LAUNCH_DIRECTION_DEFINITIONS[convention];
  if (definition.signRule !== "positive_right") {
    throw new RangeError(
      `unsupported launch-direction sign rule: ${definition.signRule}`,
    );
  }
  return convention;
}

export function convertLaunchDirection(
  degrees: number,
  source: LaunchDirectionConvention,
  target: LaunchDirectionConvention,
): number {
  const validDegrees = validatedDegrees(degrees);
  validatedConvention(source);
  validatedConvention(target);
  return validDegrees;
}

export function launchDirectionToFlightAzimuth(
  degrees: number,
  convention: LaunchDirectionConvention,
): number {
  validatedConvention(convention);
  return -validatedDegrees(degrees);
}

export function launchDirectionSignLabels(
  convention: LaunchDirectionConvention,
): Readonly<{ positive: string; negative: string }> {
  validatedConvention(convention);
  return Object.freeze({
    positive: "right of the target line",
    negative: "left of the target line",
  });
}

export function migrateLaunchDirectionRecord(
  values: Readonly<Record<string, unknown>>,
): Record<string, unknown> {
  const keys = ["launchDirectionDeg", "launchAzimuthDeg", "azimuthDeg"] as const;
  const present = keys
    .filter((key) => Object.prototype.hasOwnProperty.call(values, key))
    .map((key) => [key, validatedDegrees(values[key])] as const);
  if (present.length === 0) {
    throw new Error("no launch-direction field found");
  }
  const [firstKey, firstValue] = present[0];
  for (const [key, value] of present.slice(1)) {
    if (Math.abs(firstValue - value) > 1e-12) {
      throw new Error(
        `conflicting launch-direction values in '${firstKey}' and '${key}'`,
      );
    }
  }
  const convention = validatedConvention(
    values.launchDirectionConvention ?? "app_native",
  );
  return {
    ...values,
    launchDirectionDeg: firstValue,
    launchDirectionConvention: convention,
    launchDirectionSchemaVersion: 1,
  };
}

export function launchDirectionFromRecord(
  values: Readonly<Record<string, unknown>>,
): LaunchDirectionValue {
  const migrated = migrateLaunchDirectionRecord(values);
  return {
    degrees: validatedDegrees(migrated.launchDirectionDeg),
    convention: validatedConvention(migrated.launchDirectionConvention),
  };
}
