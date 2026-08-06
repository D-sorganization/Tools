/** Explicit launch-direction conventions and lossless legacy migration. */

export type LaunchDirectionConvention =
  | "app_native"
  | "launch_monitor_comparable"
  | "flight_frame";

export interface LaunchDirectionValue {
  degrees: number;
  convention: LaunchDirectionConvention;
}

export interface LaunchDirectionDefinition {
  positiveDirection: string;
  negativeDirection: string;
  reference: string;
  sourceUrl?: string;
  retrievedOn?: string;
  definitionVersion: string;
  comparabilityStatus: string;
}

export const LAUNCH_DIRECTION_DEFINITIONS: Record<
  LaunchDirectionConvention,
  LaunchDirectionDefinition
> = {
  app_native: {
    positiveDirection: "right of the target line",
    negativeDirection: "left of the target line",
    reference: "horizontal angle from the target line",
    definitionVersion: "roc-launch-direction-v1",
    comparabilityStatus: "canonical",
  },
  launch_monitor_comparable: {
    positiveDirection: "right of the target line",
    negativeDirection: "left of the target line",
    reference: "horizontal ball-CG motion relative to the target line after separation",
    sourceUrl: "https://www.trackman.com/blog/golf/what-is-launch-direction",
    retrievedOn: "2026-08-06",
    definitionVersion: "trackman-public-definition-2026-08-06",
    comparabilityStatus: "definition-and-sign-comparable",
  },
  flight_frame: {
    positiveDirection: "left of the target line (+y flight)",
    negativeDirection: "right of the target line (-y flight)",
    reference: "horizontal angle from +x in the internal flight frame",
    definitionVersion: "swing-sim-flight-frame-v1",
    comparabilityStatus: "internal-only",
  },
};

const CONVENTIONS = new Set<LaunchDirectionConvention>([
  "app_native",
  "launch_monitor_comparable",
  "flight_frame",
]);

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
  if (typeof value !== "string" || !CONVENTIONS.has(value as LaunchDirectionConvention)) {
    throw new RangeError(`unknown launch-direction convention: ${String(value)}`);
  }
  return value as LaunchDirectionConvention;
}

export function convertLaunchDirection(
  degrees: number,
  source: LaunchDirectionConvention,
  target: LaunchDirectionConvention,
): number {
  const validDegrees = validatedDegrees(degrees);
  validatedConvention(source);
  validatedConvention(target);
  const rightPositive = source === "flight_frame" ? -validDegrees : validDegrees;
  return target === "flight_frame" ? -rightPositive : rightPositive;
}

export function launchDirectionToFlightAzimuth(
  degrees: number,
  convention: LaunchDirectionConvention,
): number {
  return convertLaunchDirection(degrees, convention, "flight_frame");
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
      throw new Error(`conflicting launch-direction values in '${firstKey}' and '${key}'`);
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
