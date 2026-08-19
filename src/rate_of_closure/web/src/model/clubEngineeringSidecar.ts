/** Strict engineering sidecar for the selected representative clubhead. */

import { type ClubSpec } from "./club";
import {
  assertBindingMatchesSpec,
  type ClubAssemblyBinding,
} from "./clubAssemblyBinding";
import {
  browserArtifactDownloadRuntime,
  downloadClubArtifact,
  type ClubArtifactDownloadRuntime,
} from "./clubArtifactDownload";
import {
  BINDING_AUTHORITY_LIMITATION,
  bindingProvenance,
  boundCapabilityContract,
  boundFrameContract,
  boundMassProperties,
} from "./clubEngineeringBinding";
import {
  defaultClubheadStlFilename,
  serializeClubheadStl,
} from "./clubStlExport";

export const CLUBHEAD_ENGINEERING_FORMAT =
  "rate_of_closure.clubhead_engineering/1";
export const CLUBHEAD_ENGINEERING_MEDIA_TYPE = "application/json";

const ASSEMBLY_UNAVAILABLE =
  "No validated golf_club.ClubAssembly is bound to the selected Rate of Closure ClubSpec.";
const SHA256_HEX = /^[a-f0-9]{64}$/u;
const IDENTITY = [
  [1, 0, 0],
  [0, 1, 0],
  [0, 0, 1],
];

type JsonPrimitive = boolean | number | string | null;
type JsonValue = JsonPrimitive | JsonValue[] | { [key: string]: JsonValue };

/** Digest operation injected separately from browser download I/O. */
export interface ClubheadDigestRuntime {
  sha256Hex: (payload: ArrayBuffer) => Promise<string>;
}

/** Complete browser boundary for hashing and downloading a sidecar. */
export type ClubheadEngineeringRuntime = ClubheadDigestRuntime &
  ClubArtifactDownloadRuntime;

async function browserSha256Hex(payload: ArrayBuffer): Promise<string> {
  const subtle = globalThis.crypto?.subtle;
  if (!subtle) throw new Error("browser SHA-256 capability is unavailable");
  const digest = await subtle.digest("SHA-256", payload);
  return Array.from(new Uint8Array(digest), (value) =>
    value.toString(16).padStart(2, "0"),
  ).join("");
}

function browserEngineeringRuntime(): ClubheadEngineeringRuntime {
  return {
    ...browserArtifactDownloadRuntime(),
    sha256Hex: browserSha256Hex,
  };
}

function assertFiniteRange(
  value: number,
  field: string,
  lower: number,
  upper: number,
): void {
  if (!Number.isFinite(value) || value < lower || value > upper) {
    throw new Error(`${field} must be finite and within [${lower}, ${upper}]`);
  }
}

function assertMassPropertyEvidence(spec: ClubSpec): void {
  if (typeof spec.name !== "string" || !spec.name) {
    throw new Error("club name must be a non-empty string");
  }
  assertFiniteRange(spec.headMassKg, "headMassKg", 0.1, 0.5);
  assertFiniteRange(spec.cgDepthM, "cgDepthM", 0, 0.08);
  assertFiniteRange(spec.cgHeightM, "cgHeightM", 0, 0.06);
  assertFiniteRange(spec.moiAboutShaftKgM2, "moiAboutShaftKgM2", 5e-5, 2e-3);
}

function sortedJson(value: JsonValue): JsonValue {
  if (Array.isArray(value)) return value.map(sortedJson);
  if (value !== null && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value)
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([key, child]) => [key, sortedJson(child)]),
    );
  }
  return value;
}

function massProperties(spec: ClubSpec): JsonValue {
  return {
    assembly: {
      reason: ASSEMBLY_UNAVAILABLE,
      status: "unavailable",
    },
    head: {
      center_of_mass_m: {
        evidence_only: {
          cg_depth_m: { datum: "back from the face", value: spec.cgDepthM },
          cg_height_m: {
            datum: "above the sole plane",
            value: spec.cgHeightM,
          },
        },
        missing: [
          "heel_toe_coordinate_m",
          "reconciled_head_frame_origin_transform",
        ],
        reason:
          "The available offsets are not a complete vector in rate_of_closure.head.",
        status: "unavailable",
      },
      inertia_tensor_at_com_kg_m2: {
        evidence_only: {
          moi_about_shaft_kg_m2: {
            reference:
              "shaft axis; not a full tensor about the complete head CG",
            value: spec.moiAboutShaftKgM2,
          },
        },
        reason:
          "A single shaft-axis scalar does not determine the six independent components of a symmetric tensor.",
        status: "unavailable",
      },
      mass_kg: {
        provenance:
          "selected ClubSpec representative input; not a measurement certificate",
        status: "available",
        value: spec.headMassKg,
      },
    },
  };
}

function frameContract(): JsonValue {
  return {
    head: {
      axes: {
        x_positive: "toward target",
        y_positive: "up",
        z_positive: "toward toe",
      },
      frame_id: "rate_of_closure.head",
      handedness: "right",
      length_unit: "m",
      origin:
        "parametric mesh head-frame origin; not reconciled to the physical center of mass",
    },
    stl_from_head: {
      coordinate_scale_mm_per_m: 1000,
      rotation: IDENTITY,
      status: "available",
      translation_m: [0, 0, 0],
    },
    world_from_head: {
      reason:
        "The selected static ClubSpec and STL do not carry a complete world-from-head attitude.",
      status: "unavailable",
    },
  };
}

function capabilityContract(): JsonValue {
  return {
    assembly_mass_properties: {
      reason: ASSEMBLY_UNAVAILABLE,
      status: "unavailable",
    },
    head_center_of_mass: {
      missing: [
        "heel_toe_coordinate_m",
        "reconciled_head_frame_origin_transform",
      ],
      status: "unavailable",
    },
    head_full_inertia_tensor: {
      missing: [
        "all six independent tensor components about the complete head CG",
        "complete CG and shaft-axis reference transform",
      ],
      status: "unavailable",
    },
    head_mass: { status: "available" },
    mesh_identity: { status: "available" },
    world_from_head_attitude: {
      missing: ["complete world-from-head rotation"],
      status: "unavailable",
    },
  };
}

/** Build the sidecar without manufacturing missing CG, tensor, or attitude data. */
export async function buildClubheadEngineeringSidecar(
  spec: ClubSpec,
  runtime: ClubheadDigestRuntime = { sha256Hex: browserSha256Hex },
  binding?: ClubAssemblyBinding,
): Promise<JsonValue> {
  assertMassPropertyEvidence(spec);
  if (binding) assertBindingMatchesSpec(binding, spec);
  const stlPayload = serializeClubheadStl(spec);
  const sha256 = await runtime.sha256Hex(stlPayload);
  if (!SHA256_HEX.test(sha256)) {
    throw new Error("mesh digest must be lowercase SHA-256");
  }
  return {
    capabilities: binding
      ? (boundCapabilityContract() as JsonValue)
      : capabilityContract(),
    format: CLUBHEAD_ENGINEERING_FORMAT,
    frames: binding
      ? (boundFrameContract(
          binding,
          frameContract() as Record<string, unknown>,
        ) as JsonValue)
      : frameContract(),
    limitations: [
      "The representative render mesh is not a measured or density-integrated inertia CAD model.",
      "Two datum-relative CG offsets do not define a three-coordinate CG in the declared head frame.",
      "One shaft-axis scalar moment cannot determine a symmetric tensor about the head CG.",
      "A face normal or static loft does not define the complete world-from-head attitude.",
      ...(binding
        ? [BINDING_AUTHORITY_LIMITATION]
        : [
            "No validated shared golf-club assembly record is connected to this selected club specification.",
          ]),
    ],
    mass_properties: binding
      ? (boundMassProperties(binding) as JsonValue)
      : massProperties(spec),
    mesh: {
      byte_length: stlPayload.byteLength,
      companion_filename: defaultClubheadStlFilename(spec),
      format: "binary_stl",
      generator: "rate_of_closure.parametric_head/1",
      mesh_defining_inputs: {
        club_type: spec.clubType,
        face_bulge_radius_m: spec.faceBulgeRadiusM,
        face_roll_radius_m: spec.faceRollRadiusM,
        head_mass_kg: spec.headMassKg,
        head_style: spec.headStyle ?? "Auto",
        loft_deg: spec.loftDeg,
      },
      sha256,
    },
    provenance: {
      application: "Rate of Closure Impact Explorer",
      mass_property_authority: binding
        ? "validated ClubAssembly binding"
        : "selected ClubSpec fields only; no measured or CAD-integrated tensor source",
      selected_spec: {
        kind: "rate_of_closure.club.ClubSpec",
        name: spec.name,
      },
      ...(binding
        ? { assembly_binding: bindingProvenance(binding) as JsonValue }
        : {}),
    },
    subject: {
      kind: "selected_representative_clubhead",
      name: spec.name,
    },
  };
}

/** Serialize deterministic, key-sorted, versioned UTF-8 JSON. */
export async function serializeClubheadEngineeringSidecar(
  spec: ClubSpec,
  runtime: ClubheadDigestRuntime = { sha256Hex: browserSha256Hex },
  binding?: ClubAssemblyBinding,
): Promise<string> {
  const document = await buildClubheadEngineeringSidecar(
    spec,
    runtime,
    binding,
  );
  return `${JSON.stringify(sortedJson(document), null, 2)}\n`;
}

/** Return the portable sidecar filename paired with the selected STL name. */
export function defaultClubheadEngineeringFilename(spec: ClubSpec): string {
  return defaultClubheadStlFilename(spec).replace(
    /\.stl$/u,
    ".engineering.json",
  );
}

/** Download the sidecar after hashing the exact companion STL bytes. */
export async function downloadClubheadEngineeringSidecar(
  spec: ClubSpec,
  runtime: ClubheadEngineeringRuntime = browserEngineeringRuntime(),
  binding?: ClubAssemblyBinding,
): Promise<string> {
  const filename = defaultClubheadEngineeringFilename(spec);
  const payload = await serializeClubheadEngineeringSidecar(
    spec,
    runtime,
    binding,
  );
  downloadClubArtifact(
    payload,
    CLUBHEAD_ENGINEERING_MEDIA_TYPE,
    filename,
    runtime,
  );
  return filename;
}
