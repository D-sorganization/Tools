import { describe, expect, it, vi } from "vitest";

import driverGolden from "../vendored/fixtures/clubhead_engineering_sidecar_driver_10_5.json";

import { getClub } from "./club";
import {
  CLUBHEAD_ENGINEERING_FORMAT,
  defaultClubheadEngineeringFilename,
  downloadClubheadEngineeringSidecar,
  serializeClubheadEngineeringSidecar,
  type ClubheadEngineeringRuntime,
} from "./clubEngineeringSidecar";
import { serializeClubheadStl } from "./clubStlExport";

const DRIVER = "Driver 10.5\u00b0";
const DRIVER_STL_SHA256 =
  "3ea68a083099ce3780418e9eff0900e7178b835608261bf7d89825bddef243c8";

function digestRuntime(expectedStl: ArrayBuffer) {
  return {
    sha256Hex: vi.fn(async (payload: ArrayBuffer) => {
      expect(new Uint8Array(payload)).toEqual(new Uint8Array(expectedStl));
      return DRIVER_STL_SHA256;
    }),
  };
}

describe("selected-club engineering sidecar contract", () => {
  it("matches the PyQt golden capability and provenance boundary", async () => {
    const driver = getClub(DRIVER);
    const runtime = digestRuntime(serializeClubheadStl(driver));

    const serialized = await serializeClubheadEngineeringSidecar(driver, runtime);
    const document = JSON.parse(serialized);

    expect(serialized.endsWith("\n")).toBe(true);
    expect(document).toEqual(driverGolden);
    expect(document.format).toBe(CLUBHEAD_ENGINEERING_FORMAT);
    expect(document.subject).toEqual({
      kind: "selected_representative_clubhead",
      name: DRIVER,
    });
    expect(document.mesh).toMatchObject({
      byte_length: 89684,
      companion_filename: "driver-10-5.stl",
      format: "binary_stl",
      generator: "rate_of_closure.parametric_head/1",
      sha256: DRIVER_STL_SHA256,
    });
    expect(document.frames.head).toMatchObject({
      frame_id: "rate_of_closure.head",
      handedness: "right",
      length_unit: "m",
    });
    expect(document.frames.stl_from_head).toMatchObject({
      coordinate_scale_mm_per_m: 1000,
      rotation: [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ],
      status: "available",
      translation_m: [0, 0, 0],
    });
    expect(document.frames.world_from_head.status).toBe("unavailable");
    expect(document.mass_properties.head.mass_kg).toEqual({
      provenance:
        "selected ClubSpec representative input; not a measurement certificate",
      status: "available",
      value: 0.2,
    });
    expect(document.mass_properties.head.center_of_mass_m).toMatchObject({
      evidence_only: {
        cg_depth_m: { datum: "back from the face", value: 0.025 },
        cg_height_m: { datum: "above the sole plane", value: 0.028 },
      },
      status: "unavailable",
    });
    expect(
      document.mass_properties.head.inertia_tensor_at_com_kg_m2,
    ).toMatchObject({
      evidence_only: {
        moi_about_shaft_kg_m2: {
          reference:
            "shaft axis; not a full tensor about the complete head CG",
          value: 0.00052,
        },
      },
      status: "unavailable",
    });
    expect(document.mass_properties.assembly.status).toBe("unavailable");
    expect(document.capabilities).toMatchObject({
      assembly_mass_properties: { status: "unavailable" },
      head_center_of_mass: { status: "unavailable" },
      head_full_inertia_tensor: { status: "unavailable" },
      head_mass: { status: "available" },
      mesh_identity: { status: "available" },
      world_from_head_attitude: { status: "unavailable" },
    });
    expect(document.mass_properties.head.center_of_mass_m).not.toHaveProperty(
      "value",
    );
    expect(
      document.mass_properties.head.inertia_tensor_at_com_kg_m2,
    ).not.toHaveProperty("value");
  });

  it("fails closed when the digest boundary returns an invalid identity", async () => {
    await expect(
      serializeClubheadEngineeringSidecar(getClub(DRIVER), {
        sha256Hex: async () => "not-a-sha256",
      }),
    ).rejects.toThrow(/SHA-256/);
  });

  it("uses a portable filename and always releases the JSON object URL", async () => {
    const driver = getClub(DRIVER);
    const createObjectUrl = vi.fn<(blob: Blob) => string>(() => "blob:sidecar");
    const runtime: ClubheadEngineeringRuntime = {
      ...digestRuntime(serializeClubheadStl(driver)),
      createObjectUrl,
      clickDownload: vi.fn(),
      revokeObjectUrl: vi.fn(),
    };

    await expect(downloadClubheadEngineeringSidecar(driver, runtime)).resolves.toBe(
      "driver-10-5.engineering.json",
    );
    expect(defaultClubheadEngineeringFilename(driver)).toBe(
      "driver-10-5.engineering.json",
    );
    expect(runtime.clickDownload).toHaveBeenCalledWith(
      "blob:sidecar",
      "driver-10-5.engineering.json",
    );
    expect(runtime.revokeObjectUrl).toHaveBeenCalledWith("blob:sidecar");
    const blob = createObjectUrl.mock.calls[0][0];
    expect(blob.type).toBe("application/json");
    expect(blob.size).toBeGreaterThan(1000);
  });

  it("rejects inconsistent selected CG and scalar evidence at runtime", async () => {
    const driver = getClub(DRIVER);
    const runtime = digestRuntime(serializeClubheadStl(driver));

    await expect(
      serializeClubheadEngineeringSidecar(
        { ...driver, cgDepthM: Number.NaN },
        runtime,
      ),
    ).rejects.toThrow(/cgDepthM/);
    await expect(
      serializeClubheadEngineeringSidecar(
        { ...driver, moiAboutShaftKgM2: Number.POSITIVE_INFINITY },
        runtime,
      ),
    ).rejects.toThrow(/moiAboutShaftKgM2/);
  });
});
