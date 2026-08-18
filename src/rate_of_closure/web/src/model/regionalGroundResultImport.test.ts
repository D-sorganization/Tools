import { describe, expect, it, vi } from "vitest";

import fixture from "./__fixtures__/regional_ground_scalar_ensemble_golden_v1.json";
import {
  MAX_REGIONAL_GROUND_RESULT_BYTES,
  readRegionalGroundResultFile,
  regionalGroundResultFromJson,
} from "./regionalGroundResultImport";

type MutableRecord = Record<string, unknown>;

const cloneFixture = (): MutableRecord =>
  JSON.parse(JSON.stringify(fixture)) as MutableRecord;

const rows = (payload: MutableRecord): MutableRecord[] =>
  payload.rows as MutableRecord[];

const attributes = (row: MutableRecord): MutableRecord =>
  row.attributes as MutableRecord;

const values = (row: MutableRecord): MutableRecord =>
  row.values as MutableRecord;

const bytes = (text: string): ArrayBuffer =>
  Uint8Array.from(new TextEncoder().encode(text)).buffer;

describe("regional scalar-ensemble result import", () => {
  it("preserves Python-owned metadata, ordering, digests, and typed nulls", () => {
    const parsed = regionalGroundResultFromJson(JSON.stringify(fixture));

    expect(parsed.schema_version).toBe("scalar-ensemble/v1");
    expect(parsed.provenance).toEqual(fixture.provenance);
    expect(parsed.stages).toEqual(fixture.stages);
    expect(parsed.categories).toEqual(fixture.categories);
    expect(parsed.variables).toEqual(fixture.variables);
    expect(parsed.cohorts).toEqual(fixture.cohorts);
    expect(parsed.rows.map(({ trial_index }) => trial_index)).toEqual([0, 1, 2, 3]);
    expect(parsed.rows.map(({ cohort }) => cohort)).toEqual([
      "complete", "partial", "failed", "unavailable",
    ]);
    expect(parsed.rows[0].attributes?.ground_model_id)
      .toBe("tools-ground-impact-bounce+tools-ground-skid-roll");
    expect(parsed.rows[0].attributes?.variation_input_sha256).toMatch(/^[0-9a-f]{64}$/);
    expect(parsed.rows[1].values["metric.total_distance"]).toBeNull();
    expect(parsed.rows[2].values["metric.total_distance"]).toBeNull();
    expect(parsed.rows[3].values["metric.total_distance"]).toBeNull();
    expect(parsed.rows[3].values["metric.total_distance"]).not.toBe(0);
    expect(Object.isFrozen(parsed.rows[0].values)).toBe(true);
  });

  it("rejects duplicate, extra, unsupported-version, and malformed shapes", () => {
    const text = JSON.stringify(fixture);
    const duplicate = text.replace(
      '"result_id":',
      '"result_id":"forged","result_id":',
    );
    expect(() => regionalGroundResultFromJson(duplicate)).toThrow(/duplicate/i);

    const extra = cloneFixture();
    extra.unexpected = true;
    expect(() => regionalGroundResultFromJson(JSON.stringify(extra))).toThrow(/fields/i);

    const nestedExtra = cloneFixture();
    rows(nestedExtra)[0].unexpected = true;
    expect(() => regionalGroundResultFromJson(JSON.stringify(nestedExtra)))
      .toThrow(/fields/i);

    const evidenceExtra = cloneFixture();
    attributes(rows(evidenceExtra)[0]).unexpected = "forged";
    expect(() => regionalGroundResultFromJson(JSON.stringify(evidenceExtra)))
      .toThrow(/fields/i);

    const version = cloneFixture();
    version.schema_version = "scalar-ensemble/v2";
    expect(() => regionalGroundResultFromJson(JSON.stringify(version)))
      .toThrow(/schema/i);

    expect(() => regionalGroundResultFromJson("[]")).toThrow(/object/i);
  });

  it("rejects nonfinite, unsafe, Boolean, surrogate, and forged digest values", () => {
    const text = JSON.stringify(fixture);
    const nonfinite = text.replace(
      /("metric\.total_distance":)-?\d+(?:\.\d+)?(?:e[+-]?\d+)?/i,
      (_match, prefix: string) => `${prefix}1e309`,
    );
    expect(() => regionalGroundResultFromJson(nonfinite)).toThrow(/finite/i);

    const unsafe = cloneFixture();
    rows(unsafe)[0].trial_index = Number.MAX_SAFE_INTEGER + 1;
    expect(() => regionalGroundResultFromJson(JSON.stringify(unsafe))).toThrow(/safe/i);

    const bool = cloneFixture();
    rows(bool)[0].trial_index = true;
    expect(() => regionalGroundResultFromJson(JSON.stringify(bool))).toThrow(/finite|integer/i);

    const boolValue = cloneFixture();
    values(rows(boolValue)[0])["input.ground.base.rolling_resistance"] = false;
    expect(() => regionalGroundResultFromJson(JSON.stringify(boolValue))).toThrow(/finite/i);

    const surrogate = cloneFixture();
    (surrogate.provenance as MutableRecord).source_provenance = "\ud800";
    expect(() => regionalGroundResultFromJson(JSON.stringify(surrogate))).toThrow(/surrogate/i);

    const digest = cloneFixture();
    attributes(rows(digest)[0]).variation_input_sha256 = "not-a-digest";
    expect(() => regionalGroundResultFromJson(JSON.stringify(digest))).toThrow(/sha256/i);

    const transfer = cloneFixture();
    attributes(rows(transfer)[3]).endpoint_qualification = "summary_unavailable";
    expect(() => regionalGroundResultFromJson(JSON.stringify(transfer)))
      .toThrow(/transfer failure.*unavailable/i);

    const model = cloneFixture();
    attributes(rows(model)[0]).ground_model_id = null;
    attributes(rows(model)[0]).ground_model_version = null;
    expect(() => regionalGroundResultFromJson(JSON.stringify(model)))
      .toThrow(/complete-rest.*model/i);
  });

  it("rejects reordered trials and any numeric promotion of censored failures", () => {
    const reordered = cloneFixture();
    (reordered.rows as unknown[]).reverse();
    expect(() => regionalGroundResultFromJson(JSON.stringify(reordered)))
      .toThrow(/trial ordering/i);

    [1, 2, 3].forEach((index) => {
      const promoted = cloneFixture();
      values(rows(promoted)[index])["metric.total_distance"] = 0;
      expect(() => regionalGroundResultFromJson(JSON.stringify(promoted)))
        .toThrow(/typed null/i);
    });
  });

  it("rejects forged row, series, definition, value, and cohort identities", () => {
    const rowId = cloneFixture();
    rows(rowId)[0].row_id = "forged-row";
    expect(() => regionalGroundResultFromJson(JSON.stringify(rowId)))
      .toThrow(/row_id/i);

    const series = cloneFixture();
    rows(series)[1].series_id = "forged-series";
    rows(series)[1].row_id = "series:forged-series/trial:1";
    expect(() => regionalGroundResultFromJson(JSON.stringify(series)))
      .toThrow(/trial ordering|series/i);

    const duplicateDefinition = cloneFixture();
    const definitions = duplicateDefinition.variables as MutableRecord[];
    definitions[1].key = definitions[0].key;
    expect(() => regionalGroundResultFromJson(JSON.stringify(duplicateDefinition)))
      .toThrow(/variable keys must be unique/i);

    const duplicateValue = JSON.stringify(fixture).replace(
      '"metric.carry_distance":',
      '"metric.carry_distance":12,"metric.carry_distance":',
    );
    expect(() => regionalGroundResultFromJson(duplicateValue)).toThrow(/duplicate/i);

    const cohort = cloneFixture();
    rows(cohort)[0].cohort = "unknown";
    expect(() => regionalGroundResultFromJson(JSON.stringify(cohort)))
      .toThrow(/cohort/i);
  });

  it("enforces the encoded byte bound before parsing", () => {
    expect(() => regionalGroundResultFromJson(
      " ".repeat(MAX_REGIONAL_GROUND_RESULT_BYTES + 1),
    )).toThrow(/maximum wire size/i);
  });

  it("imports a bounded file with fatal UTF-8 and distrusts declared size", async () => {
    const text = JSON.stringify(fixture);
    const imported = await readRegionalGroundResultFile({
      name: "result.json",
      size: new TextEncoder().encode(text).byteLength,
      arrayBuffer: vi.fn().mockResolvedValue(bytes(text)),
    });
    expect(imported.result_id).toBe(fixture.result_id);

    await expect(readRegionalGroundResultFile({
      name: "invalid.json",
      size: 2,
      arrayBuffer: vi.fn().mockResolvedValue(Uint8Array.from([0xc3, 0x28]).buffer),
    })).rejects.toThrow(/valid UTF-8/i);

    const read = vi.fn().mockResolvedValue(bytes("{}"));
    await expect(readRegionalGroundResultFile({
      name: "oversize.json",
      size: MAX_REGIONAL_GROUND_RESULT_BYTES + 1,
      arrayBuffer: read,
    })).rejects.toThrow(/maximum wire size/i);
    expect(read).not.toHaveBeenCalled();

    const dishonest = " ".repeat(MAX_REGIONAL_GROUND_RESULT_BYTES + 1);
    await expect(readRegionalGroundResultFile({
      name: "dishonest.json",
      size: 1,
      arrayBuffer: vi.fn().mockResolvedValue(bytes(dishonest)),
    })).rejects.toThrow(/maximum wire size/i);
  });
});
