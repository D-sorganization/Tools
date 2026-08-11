/** Selected-club STL serialization and browser download boundary. */

import { buildParametricHead, type ClubSpec } from "./club";
import { writeBinaryStl, type Triangle, type Vec3 } from "./mesh";

export const CLUBHEAD_STL_HEADER =
  "ROC;units=mm;frame=head;axes=x=target,y=up,z=toe;mesh=parametric";

const MILLIMETRES_PER_METRE = 1000;
const MAX_FILENAME_STEM_LENGTH = 80;
const UNSAFE_FILENAME = /[^a-z0-9]+/gu;
const CLUB_TYPES = new Set([
  "Driver",
  "Wood",
  "Hybrid",
  "Iron",
  "Wedge",
  "Putter",
]);
const HEAD_STYLES = new Set(["Auto", "Mallet", "Blade"]);
const WINDOWS_RESERVED_STEMS = new Set([
  "aux",
  "con",
  "nul",
  "prn",
  ...Array.from({ length: 9 }, (_, index) => `com${index + 1}`),
  ...Array.from({ length: 9 }, (_, index) => `lpt${index + 1}`),
]);

/** Browser operations injected at the download boundary for deterministic tests. */
export interface ClubheadDownloadRuntime {
  createObjectUrl: (blob: Blob) => string;
  clickDownload: (url: string, filename: string) => void;
  revokeObjectUrl: (url: string) => void;
}

function browserDownloadRuntime(): ClubheadDownloadRuntime {
  return {
    createObjectUrl: (blob) => URL.createObjectURL(blob),
    clickDownload: (url, filename) => {
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = filename;
      document.body.append(anchor);
      try {
        anchor.click();
      } finally {
        anchor.remove();
      }
    },
    revokeObjectUrl: (url) => URL.revokeObjectURL(url),
  };
}

function assertMeshSpec(spec: ClubSpec): void {
  if (!CLUB_TYPES.has(spec.clubType)) {
    throw new Error("clubType must identify a supported parametric profile");
  }
  if (spec.headStyle !== undefined && !HEAD_STYLES.has(spec.headStyle)) {
    throw new Error("headStyle must identify a supported parametric profile");
  }
  if (!(spec.headMassKg >= 0.1 && spec.headMassKg <= 0.5)) {
    throw new Error("headMassKg must be finite and within [0.1, 0.5]");
  }
  if (!(spec.loftDeg >= 0 && spec.loftDeg <= 70)) {
    throw new Error("loftDeg must be finite and within [0, 70]");
  }
  for (const [field, radius] of [
    ["faceBulgeRadiusM", spec.faceBulgeRadiusM],
    ["faceRollRadiusM", spec.faceRollRadiusM],
  ] as const) {
    if (radius !== null && !(radius >= 0.1 && radius <= 2)) {
      throw new Error(`${field} must be null or finite and within [0.1, 2]`);
    }
  }
}

function scaleTriangleToMillimetres(triangle: Triangle): Triangle {
  return triangle.map(
    (vertex) =>
      vertex.map((coordinate) => coordinate * MILLIMETRES_PER_METRE) as Vec3,
  ) as Triangle;
}

/** Return the PyQt-parity portable, lowercase default STL filename. */
export function defaultClubheadStlFilename(spec: ClubSpec): string {
  if (typeof spec.name !== "string") {
    throw new Error("club name must be a string");
  }
  let stem = spec.name
    .toLowerCase()
    .replace(UNSAFE_FILENAME, "-")
    .replace(/^-+|-+$/gu, "");
  stem =
    stem.slice(0, MAX_FILENAME_STEM_LENGTH).replace(/-+$/u, "") ||
    "clubhead";
  if (WINDOWS_RESERVED_STEMS.has(stem)) stem = `clubhead-${stem}`;
  return `${stem}.stl`;
}

/** Serialize the selected representative head as mm-based binary STL. */
export function serializeClubheadStl(spec: ClubSpec): ArrayBuffer {
  assertMeshSpec(spec);
  const triangles = buildParametricHead(spec).map(scaleTriangleToMillimetres);
  return writeBinaryStl(triangles, CLUBHEAD_STL_HEADER);
}

/** Download the selected representative head and return its portable filename. */
export function downloadClubheadStl(
  spec: ClubSpec,
  runtime: ClubheadDownloadRuntime = browserDownloadRuntime(),
): string {
  const filename = defaultClubheadStlFilename(spec);
  const blob = new Blob([serializeClubheadStl(spec)], { type: "model/stl" });
  const url = runtime.createObjectUrl(blob);
  if (!url) throw new Error("browser did not create an STL download URL");
  try {
    runtime.clickDownload(url, filename);
  } finally {
    runtime.revokeObjectUrl(url);
  }
  return filename;
}
