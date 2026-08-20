/** Strict untrusted-value parser for the existing scalar-ensemble/v1 wire shape. */

import {
  SCALAR_ENSEMBLE_SCHEMA_VERSION,
  createScalarEnsemble,
  type ScalarEnsembleInput,
  type ScalarEnsembleProvenance,
  type ScalarEnsembleResult,
  type ScalarEnsembleRow,
  type ScalarEnsembleStage,
  type ScalarVariableDefinition,
} from "./scalarEnsembleContract";
import { array, exact, finiteRaw, integer, record } from "./flightGroundValidation";

const MAX_STAGES = 32;
const MAX_CATEGORIES = 64;
const MAX_VARIABLES = 256;
const MAX_COHORTS = 32;
const ROOT_FIELDS = [
  "schema_version", "result_id", "provenance", "stages", "categories",
  "variables", "cohorts", "rows",
] as const;
const PROVENANCE_FIELDS = [
  "adapter_id", "source_schema_version", "source_provenance",
] as const;
const LABEL_FIELDS = ["key", "label"] as const;
const VARIABLE_FIELDS = ["key", "label", "unit", "stage_key", "category_key"] as const;
const ROW_REQUIRED_FIELDS = ["row_id", "trial_index", "cohort", "values"] as const;
const ROW_OPTIONAL_FIELDS = ["series_id", "attributes"] as const;

const safeText = (value: unknown, name: string): string => {
  if (typeof value !== "string" || value.trim().length === 0) {
    throw new TypeError(name + " must be nonblank text");
  }
  if (/[\uD800-\uDFFF]/.test(value)) {
    throw new RangeError(name + " must not contain surrogate code points");
  }
  return value;
};

const boundedArray = (
  value: unknown,
  name: string,
  maximum: number,
): readonly unknown[] => {
  const items = array(value, name);
  if (items.length > maximum) throw new RangeError(`${name} exceeds ${maximum} entries`);
  return items;
};

const exactWithOptional = (
  value: Record<string, unknown>,
  name: string,
): void => {
  const actual = Object.keys(value);
  const allowed: ReadonlySet<string> = new Set([
    ...ROW_REQUIRED_FIELDS,
    ...ROW_OPTIONAL_FIELDS,
  ]);
  if (actual.some((key) => !allowed.has(key)) ||
      ROW_REQUIRED_FIELDS.some((key) => !Object.prototype.hasOwnProperty.call(value, key))) {
    throw new RangeError(name + " fields do not match v1 schema");
  }
};

const provenance = (value: unknown): ScalarEnsembleProvenance => {
  const item = record(value, "scalar ensemble provenance");
  exact(item, PROVENANCE_FIELDS, "scalar ensemble provenance");
  return Object.freeze({
    adapter_id: safeText(item.adapter_id, "provenance adapter_id"),
    source_schema_version: safeText(
      item.source_schema_version,
      "provenance source_schema_version",
    ),
    source_provenance: safeText(item.source_provenance, "provenance source_provenance"),
  });
};

const label = (value: unknown, name: string): ScalarEnsembleStage => {
  const item = record(value, name);
  exact(item, LABEL_FIELDS, name);
  return {
    key: safeText(item.key, name + " key"),
    label: safeText(item.label, name + " label"),
  };
};

const variable = (value: unknown, index: number): ScalarVariableDefinition => {
  const name = `scalar ensemble variable[${index}]`;
  const item = record(value, name);
  exact(item, VARIABLE_FIELDS, name);
  return {
    key: safeText(item.key, name + " key"),
    label: safeText(item.label, name + " label"),
    unit: safeText(item.unit, name + " unit"),
    stage_key: safeText(item.stage_key, name + " stage_key"),
    category_key: safeText(item.category_key, name + " category_key"),
  };
};

const scalarValues = (value: unknown, name: string): Record<string, number | null> => {
  const item = record(value, name);
  return Object.fromEntries(Object.entries(item).map(([key, raw]) => {
    safeText(key, name + " key");
    return [key, raw === null ? null : finiteRaw(raw, `${name}[${JSON.stringify(key)}]`)];
  }));
};

const attributes = (value: unknown, name: string): Record<string, string | null> => {
  const item = record(value, name);
  return Object.fromEntries(Object.entries(item).map(([key, raw]) => {
    safeText(key, name + " key");
    return [key, raw === null ? null : safeText(raw, `${name}[${JSON.stringify(key)}]`)];
  }));
};

const row = (value: unknown, index: number): ScalarEnsembleRow<string> => {
  const name = `scalar ensemble row[${index}]`;
  const item = record(value, name);
  exactWithOptional(item, name);
  const series = Object.prototype.hasOwnProperty.call(item, "series_id")
    ? safeText(item.series_id, name + " series_id")
    : undefined;
  const evidence = Object.prototype.hasOwnProperty.call(item, "attributes")
    ? attributes(item.attributes, name + " attributes")
    : undefined;
  return {
    row_id: safeText(item.row_id, name + " row_id"),
    trial_index: integer(item.trial_index, name + " trial_index"),
    ...(series === undefined ? {} : { series_id: series }),
    cohort: safeText(item.cohort, name + " cohort"),
    values: scalarValues(item.values, name + " values"),
    ...(evidence === undefined ? {} : { attributes: evidence }),
  };
};

const definitions = (item: Record<string, unknown>): Omit<
  ScalarEnsembleInput<string>, "result_id" | "provenance" | "rows"
> => ({
  stages: boundedArray(item.stages, "scalar ensemble stages", MAX_STAGES)
    .map((entry, index) => label(entry, `scalar ensemble stage[${index}]`)),
  categories: boundedArray(item.categories, "scalar ensemble categories", MAX_CATEGORIES)
    .map((entry, index) => label(entry, `scalar ensemble category[${index}]`)),
  variables: boundedArray(item.variables, "scalar ensemble variables", MAX_VARIABLES)
    .map(variable),
  cohorts: boundedArray(item.cohorts, "scalar ensemble cohorts", MAX_COHORTS)
    .map((entry, index) => label(entry, `scalar ensemble cohort[${index}]`)),
});

/** Parse and freeze one exact scalar-ensemble/v1 value without executing a model. */
export const parseScalarEnsembleWire = (
  value: unknown,
  maxRows: number,
): ScalarEnsembleResult<string> => {
  if (!Number.isSafeInteger(maxRows) || maxRows < 1) {
    throw new RangeError("maxRows must be a positive safe integer");
  }
  const item = record(value, "scalar ensemble result");
  exact(item, ROOT_FIELDS, "scalar ensemble result");
  if (item.schema_version !== SCALAR_ENSEMBLE_SCHEMA_VERSION) {
    throw new RangeError("unsupported scalar ensemble schema");
  }
  const rawRows = boundedArray(item.rows, "scalar ensemble rows", maxRows);
  const input: ScalarEnsembleInput<string> = {
    result_id: safeText(item.result_id, "scalar ensemble result_id"),
    provenance: provenance(item.provenance),
    ...definitions(item),
    rows: rawRows.map(row),
  };
  return createScalarEnsemble(input);
};
