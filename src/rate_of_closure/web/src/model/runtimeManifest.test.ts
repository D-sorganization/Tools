import { describe, expect, it } from "vitest";

import fixture from "./__fixtures__/runtime_manifest_parity_v1.json";
import { canonicalNumericJson } from "./flightGroundContract";
import {
  createRuntimeManifest,
  parseRuntimeManifest,
  runtimeManifestFromJson,
  stableRuntimeManifestJson,
} from "./runtimeManifest";

const source = (): Record<string, unknown> => structuredClone(fixture.manifest);
const calculations = (value: Record<string, unknown>): Array<Record<string, unknown>> =>
  value.calculations as Array<Record<string, unknown>>;

describe("calculation runtime manifest v1", () => {
  it("matches the shared Python/TypeScript canonical fixture", () => {
    const parsed = parseRuntimeManifest(fixture.manifest);

    expect(stableRuntimeManifestJson(parsed)).toBe(fixture.expected_canonical_json);
    expect(JSON.parse(stableRuntimeManifestJson(parsed))).toEqual(fixture.manifest);
    expect(runtimeManifestFromJson(JSON.stringify(fixture.manifest))).toEqual(parsed);
  });

  it("builds only from explicit inputs and deeply freezes the result", () => {
    const parsed = parseRuntimeManifest(fixture.manifest);
    const rebuilt = createRuntimeManifest({
      surfaceId: parsed.surface_id,
      build: parsed.build,
      calculations: parsed.calculations,
      provenance: parsed.provenance,
    });

    expect(rebuilt).toEqual(parsed);
    expect(Object.isFrozen(rebuilt)).toBe(true);
    expect(Object.isFrozen(rebuilt.build)).toBe(true);
    expect(Object.isFrozen(rebuilt.calculations)).toBe(true);
    expect(Object.isFrozen(rebuilt.calculations[0].numerical_options)).toBe(true);
  });

  it.each([
    ["unknown top-level field", (value: Record<string, unknown>) => { value.extra = true; }],
    ["unknown nested field", (value: Record<string, unknown>) => {
      (value.build as Record<string, unknown>).extra = true;
    }],
    ["unsupported schema", (value: Record<string, unknown>) => {
      value.schema_version = "calculation-runtime-manifest/v2";
    }],
    ["unknown surface", (value: Record<string, unknown>) => { value.surface_id = "tools.cli"; }],
    ["non-SHA revision", (value: Record<string, unknown>) => {
      (value.build as Record<string, unknown>).tools_commit = "working-tree";
    }],
    ["leading-zero major version", (value: Record<string, unknown>) => {
      (value.build as Record<string, unknown>).package_version = "01.0.0";
    }],
    ["leading-zero minor version", (value: Record<string, unknown>) => {
      (value.build as Record<string, unknown>).package_version = "1.00.0";
    }],
    ["leading-zero patch version", (value: Record<string, unknown>) => {
      (value.build as Record<string, unknown>).package_version = "1.0.00";
    }],
    ["leading-zero prerelease version", (value: Record<string, unknown>) => {
      (value.build as Record<string, unknown>).package_version = "1.0.0-01";
    }],
    ["duplicate domain", (value: Record<string, unknown>) => {
      calculations(value)[2].domain = "flight";
    }],
    ["out-of-order domains", (value: Record<string, unknown>) => {
      calculations(value).reverse();
    }],
    ["available reason", (value: Record<string, unknown>) => {
      calculations(value)[0].reason = "fallback";
    }],
    ["available missing authority", (value: Record<string, unknown>) => {
      calculations(value)[0].implementation_authority = null;
    }],
    ["unavailable model leak", (value: Record<string, unknown>) => {
      calculations(value)[2].model_id = "unqualified";
    }],
    ["unavailable missing reason", (value: Record<string, unknown>) => {
      calculations(value)[2].reason = null;
    }],
    ["placeholder unavailable reason", (value: Record<string, unknown>) => {
      calculations(value)[2].reason = "Unknown";
    }],
    ["one-letter unavailable reason", (value: Record<string, unknown>) => {
      calculations(value)[2].reason = "x";
    }],
    ["abbreviated unavailable reason", (value: Record<string, unknown>) => {
      calculations(value)[2].reason = "n/a";
    }],
    ["whitespace sentinel unavailable reason", (value: Record<string, unknown>) => {
      calculations(value)[2].reason = " \tUNAVAILABLE\n";
    }],
    ["surrounding whitespace on explanatory reason", (value: Record<string, unknown>) => {
      calculations(value)[2].reason =
        " No qualified ground producer was selected for this run. ";
    }],
    ["duplicate option", (value: Record<string, unknown>) => {
      const options = calculations(value)[0].numerical_options as unknown[];
      options.push(structuredClone(options[0]));
    }],
    ["numeric option without unit", (value: Record<string, unknown>) => {
      const options = calculations(value)[1].numerical_options as Array<Record<string, unknown>>;
      options[0].unit = null;
    }],
    ["text option with unit", (value: Record<string, unknown>) => {
      const options = calculations(value)[0].numerical_options as Array<Record<string, unknown>>;
      options[0].unit = "1";
    }],
    ["duplicate evidence", (value: Record<string, unknown>) => {
      const provenance = value.provenance as Record<string, unknown>;
      (provenance.evidence_ids as string[]).push("issue-4261");
    }],
    ["surrogate provenance text", (value: Record<string, unknown>) => {
      const provenance = value.provenance as Record<string, unknown>;
      provenance.source_reference = "fixture-\uD800";
    }],
  ])("rejects %s", (_name, mutate) => {
    const value = source();
    mutate(value);
    expect(() => parseRuntimeManifest(value)).toThrow();
  });

  it.each([Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY])(
    "rejects nonfinite option %s", (optionValue) => {
      const value = source();
      const options = calculations(value)[1].numerical_options as Array<Record<string, unknown>>;
      options[0].value = optionValue;
      expect(() => parseRuntimeManifest(value)).toThrow(/finite/);
    },
  );

  it("rejects unsafe numbers and duplicate JSON fields", () => {
    const value = source();
    const options = calculations(value)[1].numerical_options as Array<Record<string, unknown>>;
    options[0].value = Number.MAX_SAFE_INTEGER + 1;
    expect(() => parseRuntimeManifest(value)).toThrow(/safe numeric magnitude/);
    expect(() => runtimeManifestFromJson(
      '{"schema_version":"first","schema_version":"second"}',
    )).toThrow(/duplicate JSON field/);
  });

  it.each([Number.MAX_SAFE_INTEGER + 1, 1e16, -1e20])(
    "rejects unsafe numeric magnitude %s", (optionValue) => {
      const value = source();
      const options = calculations(value)[1].numerical_options as Array<Record<string, unknown>>;
      options[0].value = optionValue;
      expect(() => parseRuntimeManifest(value)).toThrow(/safe numeric magnitude/);
    },
  );

  it.each([-Number.MAX_SAFE_INTEGER, Number.MAX_SAFE_INTEGER])(
    "accepts and canonically serializes safe numeric boundary %s", (optionValue) => {
      const value = source();
      const options = calculations(value)[1].numerical_options as Array<Record<string, unknown>>;
      options[0].value = optionValue;
      const parsed = parseRuntimeManifest(value);
      expect(stableRuntimeManifestJson(parsed)).toContain(String(optionValue));
    },
  );

  it("accepts an explanatory unavailable reason and strict SemVer", () => {
    const value = source();
    (value.build as Record<string, unknown>).package_version = "1.2.3-alpha.1+build.5";
    calculations(value)[2].reason = "No qualified ground producer was selected for this run.";
    expect(parseRuntimeManifest(value)).toEqual(value);
  });

  it("matches the shared cross-runtime numeric-policy fixture", () => {
    const cases = fixture.numeric_policy_cases;
    expect(canonicalNumericJson(cases.safe_boundaries)).toBe(cases.expected_canonical_json);
    for (const optionValue of cases.unsafe_magnitudes) {
      const value = source();
      const options = calculations(value)[1].numerical_options as Array<Record<string, unknown>>;
      options[0].value = optionValue;
      expect(() => parseRuntimeManifest(value)).toThrow(/safe numeric magnitude/);
    }
  });
});
