/**
 * Wire `swing_sim.putting_result/2` — TypeScript mirror of
 * `shared/python/swing_sim/putting/result_wire.py` (#4800 P5).
 *
 * One versioned, fail-closed, byte-deterministic record of a single
 * integrated putt: the v1 scalar summary (roll-out, skid, time, break,
 * capture) plus the 2-D quantities #4800 P1/P2 produced — start
 * azimuth and sidespin off the face, a break-trajectory summary, and
 * the capture margin under the published Holmes/Penner effective-radius
 * model — with the provenance saying which putter and which stroke
 * produced it.
 *
 * **v2 supersedes v1 — no silent migration.** `puttingResultFromJson`
 * refuses a v1 payload rather than upgrading it (defaulting a missing
 * `start_azimuth_deg` to 0 would turn "unknown" into "square"), and
 * `puttingResultV1ArchiveFromJson` reads v1 as archive evidence only,
 * returning a distinct shape with no upgrade path. Writers only ever
 * emit v2.
 *
 * Derivations (full text in the Python docstrings):
 * - Break summary comes from the retained integration samples: the
 *   signed lateral excursion of largest magnitude and its station, the
 *   offset at rest/capture, and the travel direction at the closest
 *   approach (`+` = right, matching `startAzimuthDeg`).
 * - Capture margin is `R_eff(v_closest) - closest_approach`, positive
 *   iff the ball passed inside the effective hole. The v1 speed margin
 *   is retained beside it: the two answer different questions.
 *
 * Float formatting is runtime-local (JS shortest-round-trip prints `90`
 * where Python repr prints `90.0`); cross-runtime interchange is by
 * JSON value, byte determinism holds within each runtime.
 */

import {
  effectiveHoleRadiusM,
  type PuttLaunch,
  type PuttResult,
} from "./puttingGreen";

export const PUTTING_RESULT_FORMAT = "swing_sim.putting_result/2";
export const PUTTING_RESULT_FORMAT_V1 = "swing_sim.putting_result/1";

/** Integration kernel identity pinned into every v2 record. */
export const PUTTING_RESULT_KERNEL = "RK4-2ms-v1";

/** Where the putter came from (mirrors `golf_club.putter_head/1`). */
export const PUTTER_SOURCES = ["mesh", "library", "minimal"] as const;
export type PutterSource = (typeof PUTTER_SOURCES)[number];

/** Declared stroke parameters, or a P4 interchange import. */
export const STROKE_SOURCES = ["declared", "interchange"] as const;
export type StrokeSource = (typeof STROKE_SOURCES)[number];

export interface PuttingResultProvenance {
  putterSource: PutterSource;
  putterName: string;
  strokeSource: StrokeSource;
  captureModel: string;
  putterMeshSha256: string | null;
  putterLibraryName: string | null;
  strokeSourceId: string | null;
  kernel: string;
}

export interface PuttingResultDocument {
  provenance: PuttingResultProvenance;
  ballSpeedMps: number;
  launchAngleDeg: number;
  horizontalSpeedMps: number;
  spinRadS: number;
  sidespinRadS: number;
  startAzimuthDeg: number;
  effectiveLoftDeg: number;
  skidDistanceM: number;
  totalDistanceM: number;
  timeS: number;
  apexBreakM: number;
  apexBreakAtM: number;
  finalBreakM: number;
  entryAzimuthDeg: number;
  holeDistanceM: number;
  closestApproachM: number;
  speedAtClosestMps: number;
  effectiveHoleRadiusM: number;
  captureMarginM: number;
  holed: boolean;
  speedAtHoleMps: number | null;
  marginMps: number | null;
  missDistanceM: number | null;
}

/** A retained v1 summary, readable as evidence and nothing else. */
export interface PuttingResultV1Archive {
  skidDistanceM: number;
  totalDistanceM: number;
  timeS: number;
  breakM: number;
  holed: boolean;
  speedAtHoleMps: number | null;
  marginMps: number | null;
  missDistanceM: number | null;
}

const LAUNCH_FIELDS = [
  "ball_speed_mps",
  "launch_angle_deg",
  "horizontal_speed_mps",
  "spin_rad_s",
  "sidespin_rad_s",
  "start_azimuth_deg",
  "effective_loft_deg",
] as const;

const ROLL_FIELDS = [
  "skid_distance_m",
  "total_distance_m",
  "time_s",
  "apex_break_m",
  "apex_break_at_m",
  "final_break_m",
  "entry_azimuth_deg",
] as const;

const CAPTURE_REQUIRED = [
  "hole_distance_m",
  "closest_approach_m",
  "speed_at_closest_mps",
  "effective_hole_radius_m",
  "capture_margin_m",
] as const;

const CAPTURE_OPTIONAL = [
  "speed_at_hole_mps",
  "margin_mps",
  "miss_distance_m",
] as const;

const PROVENANCE_FIELDS = [
  "putter_source",
  "putter_name",
  "putter_mesh_sha256",
  "putter_library_name",
  "stroke_source",
  "stroke_source_id",
  "capture_model",
  "kernel",
] as const;

const V1_SUMMARY_FIELDS = [
  "skid_distance_m",
  "total_distance_m",
  "time_s",
  "break_m",
  "holed",
  "speed_at_hole_mps",
  "margin_mps",
  "miss_distance_m",
] as const;

function canonicalJson(value: unknown): string {
  if (value === null) return "null";
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "number") {
    if (!Number.isFinite(value)) {
      throw new Error("canonical JSON requires finite numbers");
    }
    return JSON.stringify(value);
  }
  if (typeof value === "string") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  if (typeof value === "object") {
    const record = value as Record<string, unknown>;
    const parts = Object.keys(record)
      .sort()
      .map((key) => `${JSON.stringify(key)}:${canonicalJson(record[key])}`);
    return `{${parts.join(",")}}`;
  }
  throw new Error("unsupported canonical JSON value");
}

function requireIdentifier(value: unknown, name: string): string {
  if (typeof value !== "string" || value.trim() === "") {
    throw new Error(`${name} must be a name`);
  }
  return value;
}

function finiteNumber(value: unknown, name: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`${name} must be a finite number`);
  }
  return value;
}

function optionalNumber(
  data: Record<string, unknown>,
  name: string,
): number | null {
  const value = data[name];
  return value === null || value === undefined
    ? null
    : finiteNumber(value, name);
}

function requireExactKeys(
  data: unknown,
  expected: readonly string[],
  what: string,
): Record<string, unknown> {
  if (typeof data !== "object" || data === null || Array.isArray(data)) {
    throw new Error(`${what} must be an object`);
  }
  const record = data as Record<string, unknown>;
  const keys = Object.keys(record).sort();
  const wanted = [...expected].sort();
  if (
    keys.length !== wanted.length ||
    keys.some((key, index) => key !== wanted[index])
  ) {
    throw new Error(`${what} fields must be exactly ${wanted.join(", ")}`);
  }
  return record;
}

/** Validate provenance fail-closed per source kind (Python twin). */
export function puttingResultProvenance(
  provenance: PuttingResultProvenance,
): PuttingResultProvenance {
  if (!PUTTER_SOURCES.includes(provenance.putterSource)) {
    throw new Error("putterSource must be mesh, library, or minimal");
  }
  if (!STROKE_SOURCES.includes(provenance.strokeSource)) {
    throw new Error("strokeSource must be declared or interchange");
  }
  requireIdentifier(provenance.putterName, "putterName");
  requireIdentifier(provenance.captureModel, "captureModel");
  requireIdentifier(provenance.kernel, "kernel");
  if (provenance.putterSource === "mesh") {
    if (
      typeof provenance.putterMeshSha256 !== "string" ||
      provenance.putterMeshSha256.length !== 64
    ) {
      throw new Error("mesh putters must carry a 64-character mesh SHA-256");
    }
    if (provenance.putterLibraryName !== null) {
      throw new Error("mesh putters must not carry putterLibraryName");
    }
  } else {
    if (provenance.putterMeshSha256 !== null) {
      throw new Error("only mesh putters may carry putterMeshSha256");
    }
    if (provenance.putterSource === "library") {
      requireIdentifier(provenance.putterLibraryName, "putterLibraryName");
    } else if (provenance.putterLibraryName !== null) {
      throw new Error("minimal putters must not carry putterLibraryName");
    }
  }
  if (provenance.strokeSource === "interchange") {
    requireIdentifier(provenance.strokeSourceId, "strokeSourceId");
  } else if (provenance.strokeSourceId !== null) {
    throw new Error("declared strokes must not carry strokeSourceId");
  }
  return provenance;
}

function breakSummary(
  result: PuttResult,
  holeDistanceM: number,
): [number, number, number, number, number] {
  let apexIndex = 0;
  for (let i = 1; i < result.pathYM.length; i += 1) {
    if (Math.abs(result.pathYM[i]) > Math.abs(result.pathYM[apexIndex])) {
      apexIndex = i;
    }
  }
  let closestIndex = 0;
  let closest = Math.hypot(result.pathXM[0] - holeDistanceM, result.pathYM[0]);
  for (let i = 1; i < result.pathXM.length; i += 1) {
    const distance = Math.hypot(
      result.pathXM[i] - holeDistanceM,
      result.pathYM[i],
    );
    if (distance < closest) {
      closest = distance;
      closestIndex = i;
    }
  }
  const entryIndex = Math.max(closestIndex, 1);
  const dx = result.pathXM[entryIndex] - result.pathXM[entryIndex - 1];
  const dy = result.pathYM[entryIndex] - result.pathYM[entryIndex - 1];
  const entryAzimuthDeg =
    dx === 0 && dy === 0 ? 0 : (Math.atan2(-dy, dx) * 180.0) / Math.PI;
  return [
    result.pathYM[apexIndex],
    result.pathXM[apexIndex],
    entryAzimuthDeg,
    closest,
    result.speedsMps[closestIndex],
  ];
}

/** Build the v2 record from one launch and its integrated result. */
export function puttingResultDocument(
  launch: PuttLaunch,
  result: PuttResult,
  provenance: PuttingResultProvenance,
  holeDistanceM: number,
): PuttingResultDocument {
  finiteNumber(holeDistanceM, "holeDistanceM");
  const [apex, apexAt, entryAzimuth, closest, speedAtClosest] = breakSummary(
    result,
    holeDistanceM,
  );
  const radius = effectiveHoleRadiusM(speedAtClosest);
  const document: PuttingResultDocument = {
    provenance: puttingResultProvenance(provenance),
    ballSpeedMps: launch.ballSpeedMps,
    launchAngleDeg: launch.launchAngleDeg,
    horizontalSpeedMps: launch.horizontalSpeedMps,
    spinRadS: launch.spinRadS,
    sidespinRadS: launch.sidespinRadS ?? 0.0,
    startAzimuthDeg: launch.startAzimuthDeg ?? 0.0,
    effectiveLoftDeg: launch.effectiveLoftDeg,
    skidDistanceM: result.skidDistanceM,
    totalDistanceM: result.totalDistanceM,
    timeS: result.timeS,
    apexBreakM: apex,
    apexBreakAtM: apexAt,
    finalBreakM: result.breakM,
    entryAzimuthDeg: entryAzimuth,
    holeDistanceM,
    closestApproachM: closest,
    speedAtClosestMps: speedAtClosest,
    effectiveHoleRadiusM: radius,
    captureMarginM: radius - closest,
    holed: result.holed,
    speedAtHoleMps: result.speedAtHoleMps,
    marginMps: result.marginMps,
    missDistanceM: result.missDistanceM,
  };
  return requirePuttingResultDocument(document);
}

/** Structural contract shared by the builder and the parser. */
export function requirePuttingResultDocument(
  document: PuttingResultDocument,
): PuttingResultDocument {
  puttingResultProvenance(document.provenance);
  if (typeof document.holed !== "boolean") {
    throw new Error("holed must be boolean");
  }
  for (const value of [
    document.skidDistanceM,
    document.totalDistanceM,
    document.timeS,
  ]) {
    if (!(value >= 0)) throw new Error("roll summaries must be non-negative");
  }
  if (document.skidDistanceM > document.totalDistanceM + 1e-9) {
    throw new Error("skid cannot exceed the total roll");
  }
  if (document.closestApproachM < 0 || document.effectiveHoleRadiusM < 0) {
    throw new Error("capture distances must be non-negative");
  }
  if (Math.abs(document.apexBreakM) < Math.abs(document.finalBreakM) - 1e-12) {
    throw new Error("apex break must be the largest lateral excursion");
  }
  const coherent = document.holed
    ? document.speedAtHoleMps !== null &&
      document.marginMps !== null &&
      document.missDistanceM === null
    : document.marginMps === null && document.missDistanceM !== null;
  if (!coherent) {
    throw new Error("capture summaries are internally inconsistent");
  }
  return document;
}

/** Serialize v2 deterministically; identical runs are byte-identical. */
export function puttingResultToJson(document: PuttingResultDocument): string {
  return canonicalJson({
    format: PUTTING_RESULT_FORMAT,
    provenance: {
      putter_source: document.provenance.putterSource,
      putter_name: document.provenance.putterName,
      putter_mesh_sha256: document.provenance.putterMeshSha256,
      putter_library_name: document.provenance.putterLibraryName,
      stroke_source: document.provenance.strokeSource,
      stroke_source_id: document.provenance.strokeSourceId,
      capture_model: document.provenance.captureModel,
      kernel: document.provenance.kernel,
    },
    launch: {
      ball_speed_mps: document.ballSpeedMps,
      launch_angle_deg: document.launchAngleDeg,
      horizontal_speed_mps: document.horizontalSpeedMps,
      spin_rad_s: document.spinRadS,
      sidespin_rad_s: document.sidespinRadS,
      start_azimuth_deg: document.startAzimuthDeg,
      effective_loft_deg: document.effectiveLoftDeg,
    },
    roll: {
      skid_distance_m: document.skidDistanceM,
      total_distance_m: document.totalDistanceM,
      time_s: document.timeS,
      apex_break_m: document.apexBreakM,
      apex_break_at_m: document.apexBreakAtM,
      final_break_m: document.finalBreakM,
      entry_azimuth_deg: document.entryAzimuthDeg,
    },
    capture: {
      hole_distance_m: document.holeDistanceM,
      closest_approach_m: document.closestApproachM,
      speed_at_closest_mps: document.speedAtClosestMps,
      effective_hole_radius_m: document.effectiveHoleRadiusM,
      capture_margin_m: document.captureMarginM,
      holed: document.holed,
      speed_at_hole_mps: document.speedAtHoleMps,
      margin_mps: document.marginMps,
      miss_distance_m: document.missDistanceM,
    },
  });
}

function provenanceFromJson(data: unknown): PuttingResultProvenance {
  const section = requireExactKeys(data, PROVENANCE_FIELDS, "provenance");
  for (const name of [
    "putter_mesh_sha256",
    "putter_library_name",
    "stroke_source_id",
  ]) {
    const value = section[name];
    if (value !== null && typeof value !== "string") {
      throw new Error(`${name} must be a string`);
    }
  }
  return puttingResultProvenance({
    putterSource: requireIdentifier(
      section.putter_source,
      "putter_source",
    ) as PutterSource,
    putterName: requireIdentifier(section.putter_name, "putter_name"),
    strokeSource: requireIdentifier(
      section.stroke_source,
      "stroke_source",
    ) as StrokeSource,
    captureModel: requireIdentifier(section.capture_model, "capture_model"),
    putterMeshSha256: (section.putter_mesh_sha256 ?? null) as string | null,
    putterLibraryName: (section.putter_library_name ?? null) as string | null,
    strokeSourceId: (section.stroke_source_id ?? null) as string | null,
    kernel: requireIdentifier(section.kernel, "kernel"),
  });
}

/** Parse a **v2** record; v1, unknown fields, and drift are refused. */
export function puttingResultFromJson(text: string): PuttingResultDocument {
  if (typeof text !== "string") throw new Error("text must be a string");
  const data: unknown = JSON.parse(text);
  if (typeof data !== "object" || data === null || Array.isArray(data)) {
    throw new Error("putting result must be an object");
  }
  const record = data as Record<string, unknown>;
  if (record.format === PUTTING_RESULT_FORMAT_V1) {
    throw new Error(
      "putting_result/1 is superseded and is not migrated; read it as " +
        "archive evidence with puttingResultV1ArchiveFromJson",
    );
  }
  if (record.format !== PUTTING_RESULT_FORMAT) {
    throw new Error(`format must be ${PUTTING_RESULT_FORMAT}`);
  }
  requireExactKeys(
    record,
    ["format", "provenance", "launch", "roll", "capture"],
    "putting result",
  );
  const launch = requireExactKeys(record.launch, LAUNCH_FIELDS, "launch");
  const roll = requireExactKeys(record.roll, ROLL_FIELDS, "roll");
  const capture = requireExactKeys(
    record.capture,
    [...CAPTURE_REQUIRED, ...CAPTURE_OPTIONAL, "holed"],
    "capture",
  );
  if (typeof capture.holed !== "boolean") {
    throw new Error("holed must be boolean");
  }
  return requirePuttingResultDocument({
    provenance: provenanceFromJson(record.provenance),
    ballSpeedMps: finiteNumber(launch.ball_speed_mps, "ball_speed_mps"),
    launchAngleDeg: finiteNumber(launch.launch_angle_deg, "launch_angle_deg"),
    horizontalSpeedMps: finiteNumber(
      launch.horizontal_speed_mps,
      "horizontal_speed_mps",
    ),
    spinRadS: finiteNumber(launch.spin_rad_s, "spin_rad_s"),
    sidespinRadS: finiteNumber(launch.sidespin_rad_s, "sidespin_rad_s"),
    startAzimuthDeg: finiteNumber(
      launch.start_azimuth_deg,
      "start_azimuth_deg",
    ),
    effectiveLoftDeg: finiteNumber(
      launch.effective_loft_deg,
      "effective_loft_deg",
    ),
    skidDistanceM: finiteNumber(roll.skid_distance_m, "skid_distance_m"),
    totalDistanceM: finiteNumber(roll.total_distance_m, "total_distance_m"),
    timeS: finiteNumber(roll.time_s, "time_s"),
    apexBreakM: finiteNumber(roll.apex_break_m, "apex_break_m"),
    apexBreakAtM: finiteNumber(roll.apex_break_at_m, "apex_break_at_m"),
    finalBreakM: finiteNumber(roll.final_break_m, "final_break_m"),
    entryAzimuthDeg: finiteNumber(roll.entry_azimuth_deg, "entry_azimuth_deg"),
    holeDistanceM: finiteNumber(capture.hole_distance_m, "hole_distance_m"),
    closestApproachM: finiteNumber(
      capture.closest_approach_m,
      "closest_approach_m",
    ),
    speedAtClosestMps: finiteNumber(
      capture.speed_at_closest_mps,
      "speed_at_closest_mps",
    ),
    effectiveHoleRadiusM: finiteNumber(
      capture.effective_hole_radius_m,
      "effective_hole_radius_m",
    ),
    captureMarginM: finiteNumber(capture.capture_margin_m, "capture_margin_m"),
    holed: capture.holed,
    speedAtHoleMps: optionalNumber(capture, "speed_at_hole_mps"),
    marginMps: optionalNumber(capture, "margin_mps"),
    missDistanceM: optionalNumber(capture, "miss_distance_m"),
  });
}

/** Read a retained **v1** summary as archive evidence only. */
export function puttingResultV1ArchiveFromJson(
  text: string,
): PuttingResultV1Archive {
  if (typeof text !== "string") throw new Error("text must be a string");
  const data: unknown = JSON.parse(text);
  if (typeof data !== "object" || data === null || Array.isArray(data)) {
    throw new Error("putting result must be an object");
  }
  const record = data as Record<string, unknown>;
  if (record.format === PUTTING_RESULT_FORMAT) {
    throw new Error(
      "this is a putting_result/2 record; read it with puttingResultFromJson",
    );
  }
  if (record.format !== PUTTING_RESULT_FORMAT_V1) {
    throw new Error(`format must be ${PUTTING_RESULT_FORMAT_V1}`);
  }
  requireExactKeys(record, ["format", "summary"], "putting result");
  const summary = requireExactKeys(
    record.summary,
    V1_SUMMARY_FIELDS,
    "summary",
  );
  if (typeof summary.holed !== "boolean") {
    throw new Error("holed must be boolean");
  }
  return {
    skidDistanceM: finiteNumber(summary.skid_distance_m, "skid_distance_m"),
    totalDistanceM: finiteNumber(summary.total_distance_m, "total_distance_m"),
    timeS: finiteNumber(summary.time_s, "time_s"),
    breakM: finiteNumber(summary.break_m, "break_m"),
    holed: summary.holed,
    speedAtHoleMps: optionalNumber(summary, "speed_at_hole_mps"),
    marginMps: optionalNumber(summary, "margin_mps"),
    missDistanceM: optionalNumber(summary, "miss_distance_m"),
  };
}
