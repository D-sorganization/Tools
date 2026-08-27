import { describe, expect, it, vi } from "vitest";

import bindingFixture from "../vendored/fixtures/club_assembly_binding_driver_10_5.json";
import identityVectors from "../vendored/fixtures/club_assembly_binding_identity_vectors.json";

import { getClub } from "./club";
import {
  CLUB_ASSEMBLY_BINDING_FORMAT,
  MAX_BINDING_BYTES,
  parseClubAssemblyBinding,
  type ClubAssemblyBindingDigestRuntime,
} from "./clubAssemblyBinding";
import {
  clubAssemblyIdentity,
  clubAssemblyIdentityPayload,
  clubSpecIdentity,
  clubSpecIdentityPayload,
} from "./clubAssemblyBindingIdentity";
import { buildClubheadEngineeringSidecar } from "./clubEngineeringSidecar";
import { serializeClubheadStl } from "./clubStlExport";

const DRIVER = "Driver 10.5\u00b0";
const DRIVER_STL_SHA256 =
  "16ec9cf2f8fb2c5cba7c11821229d2b84dc4f779b9264c28f43a7c833c475f40";

interface BoundSidecarInspection {
  readonly capabilities: { readonly head_full_inertia_tensor: unknown };
  readonly mass_properties: {
    readonly head: { readonly center_of_mass_m: { readonly value: unknown } };
    readonly assembly: unknown;
  };
  readonly frames: { readonly world_from_head: { readonly status: unknown } };
}

function digestRuntime(): ClubAssemblyBindingDigestRuntime {
  return {
    sha256Hex: vi.fn(async (payload: ArrayBuffer) => {
      const decoded = new TextDecoder().decode(payload);
      if (decoded.includes("driver-qualified-2026-08")) {
        return bindingFixture.assembly_identity.sha256;
      }
      return bindingFixture.selected_spec_identity.sha256;
    }),
  };
}

describe("selected-spec to ClubAssembly binding", () => {
  it("matches Python canonical identity bytes and real Web Crypto digests", async () => {
    const specPayload = clubSpecIdentityPayload(getClub(DRIVER));
    const assemblyPayload = clubAssemblyIdentityPayload(
      bindingFixture.assembly,
    );

    expect(new TextDecoder().decode(specPayload)).toBe(
      identityVectors.selected_spec_identity_payload_utf8,
    );
    expect(new TextDecoder().decode(assemblyPayload)).toBe(
      identityVectors.assembly_identity_payload_utf8,
    );
    await expect(clubSpecIdentity(getClub(DRIVER))).resolves.toBe(
      identityVectors.selected_spec_sha256,
    );
    await expect(clubAssemblyIdentity(bindingFixture.assembly)).resolves.toBe(
      identityVectors.assembly_sha256,
    );
  });

  it("validates the shared versioned fixture and exact identities", async () => {
    const binding = await parseClubAssemblyBinding(
      getClub(DRIVER),
      JSON.stringify(bindingFixture),
      digestRuntime(),
    );

    expect(binding.format).toBe(CLUB_ASSEMBLY_BINDING_FORMAT);
    expect(binding.assemblyIdentity.assemblyId).toBe(
      "driver-qualified-2026-08",
    );
    expect(binding.sourceAuthority.kind).toBe("qualified_analysis");
    expect(binding.headBinding.headComponentId).toBe("head-qualified");
  });

  it("makes complete properties available only behind the validated binding", async () => {
    const spec = getClub(DRIVER);
    const binding = await parseClubAssemblyBinding(
      spec,
      JSON.stringify(bindingFixture),
      digestRuntime(),
    );
    const sidecar = (await buildClubheadEngineeringSidecar(
      spec,
      {
        sha256Hex: async (payload) => {
          expect(new Uint8Array(payload)).toEqual(
            new Uint8Array(serializeClubheadStl(spec)),
          );
          return DRIVER_STL_SHA256;
        },
      },
      binding,
    )) as unknown as BoundSidecarInspection;

    expect(sidecar.capabilities.head_full_inertia_tensor).toEqual({
      status: "available",
    });
    expect(sidecar.mass_properties.head.center_of_mass_m).toMatchObject({
      frame_id: "rate_of_closure.head",
      provenance: "validated_club_assembly_binding",
      status: "available",
    });
    const center = sidecar.mass_properties.head.center_of_mass_m
      .value as number[];
    expect(center[0]).toBeCloseTo(0.02, 14);
    expect(center[1]).toBeCloseTo(0.028, 14);
    expect(center[2]).toBeCloseTo(-0.005, 14);
    expect(sidecar.mass_properties.assembly).toMatchObject({
      component_ids: ["head-qualified", "shaft-qualified", "grip-qualified"],
      frame_id: "driver.assembly",
      status: "available",
    });
    expect(sidecar.frames.world_from_head.status).toBe("unavailable");
  });

  it("rejects spec mismatch, unsupported authority, and tampered assembly", async () => {
    const changedSpec = { ...getClub(DRIVER), loftDeg: 11 };
    await expect(
      parseClubAssemblyBinding(
        changedSpec,
        JSON.stringify(bindingFixture),
        digestRuntime(),
      ),
    ).rejects.toThrow(/selected ClubSpec identity/);

    const unqualified = structuredClone(bindingFixture);
    unqualified.source_authority.kind = "representative_default" as never;
    await expect(
      parseClubAssemblyBinding(
        getClub(DRIVER),
        JSON.stringify(unqualified),
        digestRuntime(),
      ),
    ).rejects.toThrow(/authority kind/);

    const tampered = structuredClone(bindingFixture);
    tampered.assembly.assembly_id = "substituted-assembly";
    await expect(
      parseClubAssemblyBinding(
        getClub(DRIVER),
        JSON.stringify(tampered),
        digestRuntime(),
      ),
    ).rejects.toThrow(/assembly identity/);
  });

  it("rejects duplicate fields and oversized documents before binding", async () => {
    const serialized = JSON.stringify(bindingFixture);
    const ambiguous = serialized.replace(
      '"format":"rate_of_closure.club_assembly_binding/1",',
      '"format":"rate_of_closure.club_assembly_binding/1","format":"rate_of_closure.club_assembly_binding/1",',
    );

    await expect(
      parseClubAssemblyBinding(getClub(DRIVER), ambiguous, digestRuntime()),
    ).rejects.toThrow(/duplicate field format/);
    await expect(
      parseClubAssemblyBinding(
        getClub(DRIVER),
        " ".repeat(MAX_BINDING_BYTES + 1),
        digestRuntime(),
      ),
    ).rejects.toThrow(/exceeds the 4 MiB limit/);
  });
});
