/** Exact capability basis supported by the current interactive controls. */

import type { ClubCapability } from "./capabilityContract";

export const CANONICAL_INTERACTIVE_PARAMETERS = [
  ["ball_speed", "m/s"],
  ["launch_angle", "deg"],
  ["launch_direction", "deg"],
] as const;

export function validateInteractiveCapabilityBasis(
  club: ClubCapability,
): void {
  if (club.matrixKind !== "correlation") {
    throw new RangeError("interactive workflow requires a correlation matrix");
  }
  const canonical = CANONICAL_INTERACTIVE_PARAMETERS.every(
    ([parameterId, unit], index) =>
      club.parameters[index]?.parameterId === parameterId &&
      club.parameters[index]?.unit === unit,
  );
  if (
    !canonical ||
    club.parameters.length !== CANONICAL_INTERACTIVE_PARAMETERS.length
  ) {
    throw new RangeError(
      "interactive workflow requires canonical parameter order and units: " +
        "ball_speed m/s, launch_angle deg, launch_direction deg",
    );
  }
  const dimension = CANONICAL_INTERACTIVE_PARAMETERS.length;
  if (
    club.matrix.length !== dimension ||
    club.matrix.some((row) => row.length !== dimension)
  ) {
    throw new RangeError("interactive workflow requires a 3x3 correlation matrix");
  }
}
