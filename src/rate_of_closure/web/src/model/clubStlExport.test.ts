import { describe, expect, it, vi } from "vitest";

import { buildParametricHead, getClub, type ClubSpec } from "./club";
import {
  CLUBHEAD_STL_HEADER,
  defaultClubheadStlFilename,
  downloadClubheadStl,
  serializeClubheadStl,
  type ClubheadDownloadRuntime,
} from "./clubStlExport";
import { parseStl } from "./mesh";

const DRIVER = "Driver 10.5°";

function headerOf(payload: ArrayBuffer): string {
  return new TextDecoder()
    .decode(new Uint8Array(payload, 0, 80))
    .replace(/\0+$/u, "");
}

describe("selected-club STL contract", () => {
  it("uses the same portable bounded filename policy as PyQt", () => {
    const driver = getClub(DRIVER);
    expect(defaultClubheadStlFilename(driver)).toBe("driver-10-5.stl");
    expect(defaultClubheadStlFilename({ ...driver, name: "高尔夫" })).toBe(
      "clubhead.stl",
    );
    expect(defaultClubheadStlFilename({ ...driver, name: "CON" })).toBe(
      "clubhead-con.stl",
    );
    expect(
      defaultClubheadStlFilename({ ...driver, name: "a".repeat(200) }),
    ).toHaveLength(84);
  });

  it("serializes deterministic millimetre geometry and canonical axes", () => {
    const spec = {
      ...getClub(DRIVER),
      loftDeg: 9,
      faceBulgeRadiusM: 0.28,
    };
    const first = serializeClubheadStl(spec);
    const second = serializeClubheadStl(spec);

    expect(new Uint8Array(first)).toEqual(new Uint8Array(second));
    expect(headerOf(first)).toBe(CLUBHEAD_STL_HEADER);
    const exported = parseStl(first);
    const expected = buildParametricHead(spec);
    expect(exported).toHaveLength(expected.length);
    for (
      let triangleIndex = 0;
      triangleIndex < exported.length;
      triangleIndex += 1
    ) {
      for (let vertexIndex = 0; vertexIndex < 3; vertexIndex += 1) {
        for (let axis = 0; axis < 3; axis += 1) {
          expect(exported[triangleIndex][vertexIndex][axis]).toBeCloseTo(
            expected[triangleIndex][vertexIndex][axis] * 1000,
            4,
          );
        }
      }
    }
  });

  it("does not imply that name, shaft, lie, CG, or scalar MOI drive the mesh", () => {
    const base = getClub(DRIVER);
    const nonMeshEdit = {
      ...base,
      name: "Same representative envelope",
      lengthM: 0.9,
      lieDeg: 70,
      moiAboutShaftKgM2: 1e-3,
      cgDepthM: 0.04,
      cgHeightM: 0.04,
    };

    expect(new Uint8Array(serializeClubheadStl(nonMeshEdit))).toEqual(
      new Uint8Array(serializeClubheadStl(base)),
    );
  });

  it("rejects invalid runtime mesh fields at the public serialization boundary", () => {
    const driver = getClub(DRIVER);
    expect(() =>
      serializeClubheadStl({
        ...driver,
        clubType: "Alien",
      } as unknown as ClubSpec),
    ).toThrow(/clubType/);
    expect(() =>
      serializeClubheadStl({ ...driver, headMassKg: Number.NaN }),
    ).toThrow(/headMassKg/);
    expect(() =>
      serializeClubheadStl({ ...driver, faceBulgeRadiusM: 0.01 }),
    ).toThrow(/faceBulgeRadiusM/);
  });

  it("creates one model/stl download and always releases its object URL", () => {
    const createObjectUrl = vi.fn<(blob: Blob) => string>(() => "blob:clubhead");
    const clickDownload = vi.fn();
    const revokeObjectUrl = vi.fn();
    const runtime: ClubheadDownloadRuntime = {
      createObjectUrl,
      clickDownload,
      revokeObjectUrl,
    };

    expect(downloadClubheadStl(getClub(DRIVER), runtime)).toBe(
      "driver-10-5.stl",
    );
    expect(clickDownload).toHaveBeenCalledWith(
      "blob:clubhead",
      "driver-10-5.stl",
    );
    expect(revokeObjectUrl).toHaveBeenCalledWith("blob:clubhead");
    const blob = createObjectUrl.mock.calls[0][0];
    expect(blob.type).toBe("model/stl");
    expect(blob.size).toBeGreaterThan(84);
  });

  it("releases the object URL when the browser rejects the synthetic click", () => {
    const runtime: ClubheadDownloadRuntime = {
      createObjectUrl: () => "blob:clubhead",
      clickDownload: () => {
        throw new Error("download blocked");
      },
      revokeObjectUrl: vi.fn(),
    };

    expect(() => downloadClubheadStl(getClub(DRIVER), runtime)).toThrow(
      /download blocked/,
    );
    expect(runtime.revokeObjectUrl).toHaveBeenCalledWith("blob:clubhead");
  });
});
