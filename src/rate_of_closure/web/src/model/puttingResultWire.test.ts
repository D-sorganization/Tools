/**
 * `swing_sim.putting_result/2` wire gates — the TypeScript mirror of
 * `swing_sim/putting/tests/test_result_wire.py`, test for test
 * (#4800 P5).
 *
 * The parity block pins the Python reference document value-for-value;
 * the rest gate the derived fields, the fail-closed provenance, the
 * wire posture, and the **supersede** contract (v2 refuses v1, the v1
 * archive reader refuses v2, neither migrates).
 */

import { describe, expect, it } from "vitest";

import {
  effectiveHoleRadiusM,
  planarSurface,
  simulatePuttOnSurface,
  type PuttLaunch,
} from "./puttingGreen";
import { MINIMAL_PUTTERS, strike } from "./putting";
import {
  PUTTING_RESULT_FORMAT,
  PUTTING_RESULT_FORMAT_V1,
  puttingResultDocument,
  puttingResultFromJson,
  puttingResultProvenance,
  puttingResultToJson,
  puttingResultV1ArchiveFromJson,
  type PuttingResultDocument,
  type PuttingResultProvenance,
} from "./puttingResultWire";

const BLADE = MINIMAL_PUTTERS[0];
const FLAT = planarSurface(0.0, 0.0);
const BREAKING = planarSurface(2.0, 90.0);

const MINIMAL_PROVENANCE: PuttingResultProvenance = {
  putterSource: "minimal",
  putterName: "Blade Putter",
  strokeSource: "declared",
  captureModel: "effective_radius",
  putterMeshSha256: null,
  putterLibraryName: null,
  strokeSourceId: null,
  kernel: "RK4-2ms-v1",
};

interface DocumentOptions {
  surface?: ReturnType<typeof planarSurface>;
  speedMps?: number;
  holeDistanceM?: number;
  aimDeg?: number;
  faceAngleDeg?: number;
  pathAngleDeg?: number;
  strikeOffsetToeMm?: number;
}

function buildLaunch(options: DocumentOptions): PuttLaunch {
  return strike(BLADE, options.speedMps ?? 1.6, 0.0, {
    aimDeg: options.aimDeg ?? 0.0,
    faceAngleDeg: options.faceAngleDeg ?? 0.0,
    pathAngleDeg: options.pathAngleDeg ?? 0.0,
    strikeOffsetToeMm: options.strikeOffsetToeMm ?? 0.0,
  });
}

function buildDocument(options: DocumentOptions = {}): PuttingResultDocument {
  const holeDistanceM = options.holeDistanceM ?? 3.0;
  const launch = buildLaunch(options);
  const result = simulatePuttOnSurface(launch, options.surface ?? FLAT, {
    stimpFt: 10.0,
    holeDistanceM,
  });
  return puttingResultDocument(
    launch,
    result,
    MINIMAL_PROVENANCE,
    holeDistanceM,
  );
}

describe("putting_result/2 parity with the Python reference", () => {
  it("reproduces the Python document value-for-value", () => {
    const document = buildDocument({
      surface: BREAKING,
      speedMps: 1.9,
      aimDeg: 1.5,
      faceAngleDeg: 0.5,
      pathAngleDeg: -0.5,
      strikeOffsetToeMm: 4.0,
    });
    expect(document.ballSpeedMps).toBeCloseTo(2.9809651324824222, 12);
    expect(document.startAzimuthDeg).toBeCloseTo(1.8181423953076394, 12);
    expect(document.sidespinRadS).toBeCloseTo(-1.110166802733224, 12);
    expect(document.skidDistanceM).toBeCloseTo(0.5637172608227324, 12);
    expect(document.totalDistanceM).toBeCloseTo(4.867743544949591, 12);
    expect(document.timeS).toBeCloseTo(4.581999999999717, 9);
    expect(document.apexBreakM).toBeCloseTo(0.7463663039983797, 12);
    expect(document.apexBreakAtM).toBeCloseTo(4.7178402714461605, 12);
    expect(document.finalBreakM).toBeCloseTo(0.7463663039983797, 12);
    expect(document.entryAzimuthDeg).toBeCloseTo(-8.475142837117797, 12);
    expect(document.closestApproachM).toBeCloseTo(0.12105330322045058, 12);
    expect(document.speedAtClosestMps).toBeCloseTo(1.347777011501835, 12);
    expect(document.effectiveHoleRadiusM).toBe(0.0);
    expect(document.captureMarginM).toBeCloseTo(-0.12105330322045058, 12);
    expect(document.missDistanceM).toBeCloseTo(1.872975669341815, 12);
    expect(document.holed).toBe(false);
  });
});

describe("putting_result/2 derived fields", () => {
  it("computes the capture margin from the closed-form radius", () => {
    const document = buildDocument({ surface: BREAKING });
    expect(document.captureMarginM).toBeCloseTo(
      effectiveHoleRadiusM(document.speedAtClosestMps) -
        document.closestApproachM,
      15,
    );
  });

  it("reports a holed putt as passing inside the effective hole", () => {
    const document = buildDocument({ speedMps: 1.6 });
    expect(document.holed).toBe(true);
    expect(document.captureMarginM).toBeGreaterThanOrEqual(0);
    expect(document.missDistanceM).toBeNull();
    expect(document.marginMps).not.toBeNull();
  });

  it("reports how much hole a missed putt needed", () => {
    const document = buildDocument({ surface: BREAKING });
    expect(document.holed).toBe(false);
    expect(document.captureMarginM).toBeLessThan(0);
    expect(document.missDistanceM).not.toBeNull();
  });

  it("never leaves the line on a straight flat putt", () => {
    const document = buildDocument();
    expect(document.apexBreakM).toBe(0);
    expect(document.finalBreakM).toBe(0);
    expect(document.entryAzimuthDeg).toBeCloseTo(0, 12);
    expect(document.startAzimuthDeg).toBe(0);
  });

  it("bounds the final break by the apex break", () => {
    const document = buildDocument({ surface: BREAKING });
    expect(Math.abs(document.apexBreakM)).toBeGreaterThanOrEqual(
      Math.abs(document.finalBreakM),
    );
    expect(document.apexBreakAtM).toBeGreaterThanOrEqual(0);
  });

  it("carries the start azimuth from the launch and drifts right", () => {
    const launch = strike(BLADE, 1.6, 0.0, { aimDeg: 1.5, faceAngleDeg: 0.5 });
    const result = simulatePuttOnSurface(launch, FLAT, {
      stimpFt: 10.0,
      holeDistanceM: 3.0,
    });
    const document = puttingResultDocument(
      launch,
      result,
      MINIMAL_PROVENANCE,
      3.0,
    );
    expect(document.startAzimuthDeg).toBe(launch.startAzimuthDeg);
    expect(document.sidespinRadS).toBe(launch.sidespinRadS);
    expect(document.finalBreakM).toBeLessThan(0);
  });
});

describe("putting_result/2 provenance fails closed", () => {
  const withProvenance =
    (
      patch: Partial<PuttingResultProvenance>,
    ): (() => PuttingResultProvenance) =>
    () =>
      puttingResultProvenance({ ...MINIMAL_PROVENANCE, ...patch });

  it("requires a mesh digest on a mesh putter", () => {
    expect(withProvenance({ putterSource: "mesh" })).toThrow();
  });

  it("refuses a library name on a mesh putter", () => {
    expect(
      withProvenance({
        putterSource: "mesh",
        putterMeshSha256: "a".repeat(64),
        putterLibraryName: "Blade Putter",
      }),
    ).toThrow();
  });

  it("requires a library name on a library putter", () => {
    expect(withProvenance({ putterSource: "library" })).toThrow();
  });

  it("refuses a library name on a minimal putter", () => {
    expect(withProvenance({ putterLibraryName: "Blade Putter" })).toThrow();
  });

  it("requires a source id on an imported stroke", () => {
    expect(withProvenance({ strokeSource: "interchange" })).toThrow();
  });

  it("refuses a source id on a declared stroke", () => {
    expect(withProvenance({ strokeSourceId: "mjcf-fixture" })).toThrow();
  });

  it("refuses an unknown source kind", () => {
    expect(
      withProvenance({
        putterSource: "guessed" as PuttingResultProvenance["putterSource"],
      }),
    ).toThrow();
  });
});

describe("putting_result/2 wire posture", () => {
  it("round-trips byte identically", () => {
    const text = puttingResultToJson(buildDocument({ surface: BREAKING }));
    expect(puttingResultToJson(puttingResultFromJson(text))).toBe(text);
  });

  it("serializes identical putts byte identically", () => {
    expect(puttingResultToJson(buildDocument())).toBe(
      puttingResultToJson(buildDocument()),
    );
  });

  it("declares the v2 format and the kernel", () => {
    const payload = JSON.parse(puttingResultToJson(buildDocument()));
    expect(payload.format).toBe(PUTTING_RESULT_FORMAT);
    expect(payload.provenance.kernel).toBe("RK4-2ms-v1");
  });

  it("sorts keys", () => {
    const keys = Object.keys(JSON.parse(puttingResultToJson(buildDocument())));
    expect(keys).toEqual([...keys].sort());
  });

  it("refuses an unknown field", () => {
    const payload = JSON.parse(puttingResultToJson(buildDocument()));
    payload.extra = 1;
    expect(() => puttingResultFromJson(JSON.stringify(payload))).toThrow();
  });

  it("refuses a missing field", () => {
    const payload = JSON.parse(puttingResultToJson(buildDocument()));
    delete payload.launch.sidespin_rad_s;
    expect(() => puttingResultFromJson(JSON.stringify(payload))).toThrow();
  });

  it("refuses a non-finite value", () => {
    const payload = JSON.parse(puttingResultToJson(buildDocument()));
    payload.roll.final_break_m = "NaN";
    expect(() => puttingResultFromJson(JSON.stringify(payload))).toThrow();
  });

  it("refuses a wrong format", () => {
    const payload = JSON.parse(puttingResultToJson(buildDocument()));
    payload.format = "swing_sim.putting_result/3";
    expect(() => puttingResultFromJson(JSON.stringify(payload))).toThrow();
  });

  it("refuses a non-string payload", () => {
    expect(() => puttingResultFromJson(42 as unknown as string)).toThrow();
  });
});

function v1Payload(): Record<string, unknown> {
  return {
    format: PUTTING_RESULT_FORMAT_V1,
    summary: {
      skid_distance_m: 0.31,
      total_distance_m: 3.02,
      time_s: 4.1,
      break_m: 0.0,
      holed: false,
      speed_at_hole_mps: null,
      margin_mps: null,
      miss_distance_m: 0.02,
    },
  };
}

describe("putting_result/2 supersedes v1", () => {
  it("refuses a v1 payload and says why", () => {
    expect(() => puttingResultFromJson(JSON.stringify(v1Payload()))).toThrow(
      /superseded/,
    );
  });

  it("refuses a v2 payload in the archive reader", () => {
    const text = puttingResultToJson(buildDocument());
    expect(() => puttingResultV1ArchiveFromJson(text)).toThrow(
      /putting_result\/2/,
    );
  });

  it("reads v1 as archive evidence", () => {
    const archive = puttingResultV1ArchiveFromJson(JSON.stringify(v1Payload()));
    expect(archive.missDistanceM).toBeCloseTo(0.02, 12);
    expect(archive.marginMps).toBeNull();
  });

  it("carries no 2-D evidence on the archive record", () => {
    const archive = puttingResultV1ArchiveFromJson(
      JSON.stringify(v1Payload()),
    ) as unknown as Record<string, unknown>;
    for (const name of ["startAzimuthDeg", "sidespinRadS", "captureMarginM"]) {
      expect(name in archive).toBe(false);
    }
  });

  it("refuses an unknown v1 field", () => {
    const payload = v1Payload();
    (payload.summary as Record<string, unknown>).start_azimuth_deg = 0.0;
    expect(() =>
      puttingResultV1ArchiveFromJson(JSON.stringify(payload)),
    ).toThrow();
  });
});
