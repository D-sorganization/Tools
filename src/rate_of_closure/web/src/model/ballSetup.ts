import type { ClubSpec } from "./club";

/** Serialized values match the Python configuration; UI labels use Title Case. */
export type SupportMode = "ground" | "tee";

export interface BallSetup {
  supportMode: SupportMode;
  /** Vertical clearance from the ground plane to the bottom of the ball [m]. */
  teeHeightM: number;
}

export const GOLF_BALL_RADIUS_M = 0.04267 / 2;
export const DRIVER_TEE_HEIGHT_M = 0.0381;
export const SUGGESTED_MAX_TEE_HEIGHT_M = 0.1;
export const BALL_HEIGHT_REFERENCE = "ground_plane_to_ball_bottom" as const;
export const GROUND_BALL_SETUP: Readonly<BallSetup> = Object.freeze({
  supportMode: "ground",
  teeHeightM: 0,
});

const heightError = () =>
  new RangeError("Tee height must be a finite, non-negative value.");

export function resolveBallSetup(setup?: BallSetup | null): BallSetup {
  if (setup === undefined || setup === null) return { ...GROUND_BALL_SETUP };
  if (setup.supportMode !== "ground" && setup.supportMode !== "tee") {
    throw new RangeError("Ball support mode must be Ground or Tee.");
  }
  if (setup.supportMode === "ground") {
    if (!Number.isFinite(setup.teeHeightM) || setup.teeHeightM !== 0) {
      throw new RangeError("Ground support requires tee height to be exactly 0 m.");
    }
    return { supportMode: "ground", teeHeightM: 0 };
  }
  if (
    !Number.isFinite(setup.teeHeightM) ||
    setup.teeHeightM < 0
  ) {
    throw heightError();
  }
  return { supportMode: "tee", teeHeightM: setup.teeHeightM };
}

export function ballSetupToJson(setup: BallSetup) {
  const resolved = resolveBallSetup(setup);
  return {
    support_mode: resolved.supportMode,
    tee_height_m: resolved.teeHeightM,
    height_reference: BALL_HEIGHT_REFERENCE,
    ball_center_m: ballCenterPosition(resolved),
  } as const;
}

/** Parse the canonical Python JSON shape plus the provisional camel-case web shape. */
export function ballSetupFromJson(value: unknown): BallSetup {
  if (typeof value !== "object" || value === null) {
    throw new Error("Ball setup must be an object.");
  }
  const data = value as Record<string, unknown>;
  const setup = resolveBallSetup({
    supportMode: (data.supportMode ?? data.support_mode ?? "ground") as SupportMode,
    teeHeightM: Number(data.teeHeightM ?? data.tee_height_m ?? 0),
  });
  const reference = data.height_reference ?? data.tee_height_reference;
  if (reference !== undefined && reference !== BALL_HEIGHT_REFERENCE) {
    throw new Error(`Unsupported ball setup height reference: ${String(reference)}.`);
  }
  if (data.ball_center_m !== undefined) {
    const center = Array.from(data.ball_center_m as Iterable<unknown>, Number);
    const expected = ballCenterPosition(setup);
    if (center.length !== 3 || center.some((number, index) =>
      !Number.isFinite(number) || Math.abs(number - expected[index]) > 1e-12)) {
      throw new Error("ball_center_m must match the derived ball setup geometry.");
    }
  }
  return setup;
}

export function defaultBallSetupForClub(club?: ClubSpec | null): BallSetup {
  return club?.clubType === "Driver"
    ? { supportMode: "tee", teeHeightM: DRIVER_TEE_HEIGHT_M }
    : { ...GROUND_BALL_SETUP };
}

export function ballCenterPosition(
  setup?: BallSetup | null,
): [number, number, number] {
  const resolved = resolveBallSetup(setup);
  return [0, GOLF_BALL_RADIUS_M + resolved.teeHeightM, 0];
}
