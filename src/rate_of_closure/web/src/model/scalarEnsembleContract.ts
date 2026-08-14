/** UI-neutral scalar ensemble rows and paired-finite plot projections. */

export const SCALAR_ENSEMBLE_SCHEMA_VERSION = "scalar-ensemble/v1" as const;

export interface ScalarEnsembleProvenance {
  readonly adapter_id: string;
  readonly source_schema_version: string;
  readonly source_provenance: string;
}

export interface ScalarEnsembleStage {
  readonly key: string;
  readonly label: string;
}

export interface ScalarVariableCategory {
  readonly key: string;
  readonly label: string;
}

export interface ScalarVariableDefinition {
  readonly key: string;
  readonly label: string;
  readonly unit: string;
  readonly stage_key: string;
  readonly category_key: string;
}

export interface ScalarCohortDefinition<Cohort extends string> {
  readonly key: Cohort;
  readonly label: string;
}

export interface ScalarEnsembleRow<Cohort extends string> {
  readonly row_id: string;
  readonly trial_index: number;
  readonly series_id?: string;
  readonly cohort: Cohort;
  readonly values: Readonly<Record<string, number | null>>;
  readonly attributes?: Readonly<Record<string, string | null>>;
}

export interface ScalarEnsembleInput<Cohort extends string> {
  readonly result_id: string;
  readonly provenance: ScalarEnsembleProvenance;
  readonly stages: readonly ScalarEnsembleStage[];
  readonly categories: readonly ScalarVariableCategory[];
  readonly variables: readonly ScalarVariableDefinition[];
  readonly cohorts: readonly ScalarCohortDefinition<Cohort>[];
  readonly rows: readonly ScalarEnsembleRow<Cohort>[];
}

export interface ScalarEnsembleResult<Cohort extends string>
  extends ScalarEnsembleInput<Cohort> {
  readonly schema_version: typeof SCALAR_ENSEMBLE_SCHEMA_VERSION;
}

export interface ScalarScatterPoint<Cohort extends string> {
  readonly row_id: string;
  readonly trial_index: number;
  readonly series_id?: string;
  readonly cohort: Cohort;
  readonly x: number;
  readonly y: number;
}

export interface ScalarAvailability {
  readonly total_rows: number;
  readonly x_finite: number;
  readonly y_finite: number;
  readonly paired_finite: number;
  readonly unavailable: number;
}

export interface ScalarScatterData<Cohort extends string> {
  readonly x_variable: ScalarVariableDefinition;
  readonly y_variable: ScalarVariableDefinition;
  readonly points: readonly ScalarScatterPoint<Cohort>[];
  readonly availability: {
    readonly overall: ScalarAvailability;
    readonly by_cohort: Readonly<Record<Cohort, ScalarAvailability>>;
  };
}

const nonempty = (value: string, name: string): string => {
  if (!value.trim()) throw new RangeError(`${name} must be nonempty`);
  return value;
};

const assertUniqueKeys = (
  definitions: readonly { readonly key: string }[],
  name: string,
): void => {
  const keys = definitions.map(({ key }) => nonempty(key, `${name} key`));
  if (new Set(keys).size !== keys.length) throw new RangeError(`${name} keys must be unique`);
};

/** Return the canonical composite identity for one series/trial row. */
const encodeRfc3986 = (value: string): string => encodeURIComponent(value).replace(
  /[!'()*]/g,
  (character) => `%${character.charCodeAt(0).toString(16).toUpperCase()}`,
);

export function scalarEnsembleRowId(trialIndex: number, seriesId?: string): string {
  if (!Number.isInteger(trialIndex) || trialIndex < 0) {
    throw new RangeError("trialIndex must be a nonnegative integer");
  }
  if (seriesId === undefined) return `trial:${trialIndex}`;
  nonempty(seriesId, "seriesId");
  return `series:${encodeRfc3986(seriesId)}/trial:${trialIndex}`;
}

function validateDefinitions<Cohort extends string>(input: ScalarEnsembleInput<Cohort>): void {
  nonempty(input.result_id, "result_id");
  Object.entries(input.provenance).forEach(([key, value]) => nonempty(value, `provenance.${key}`));
  if (!input.stages.length || !input.categories.length || !input.variables.length) {
    throw new RangeError("stages, categories, and variables must be nonempty");
  }
  if (!input.cohorts.length) throw new RangeError("cohorts must be nonempty");
  assertUniqueKeys(input.stages, "stage");
  assertUniqueKeys(input.categories, "category");
  assertUniqueKeys(input.variables, "variable");
  assertUniqueKeys(input.cohorts, "cohort");
  [...input.stages, ...input.categories, ...input.cohorts].forEach((definition) => {
    nonempty(definition.label, `${definition.key} label`);
  });
  const stages = new Set(input.stages.map(({ key }) => key));
  const categories = new Set(input.categories.map(({ key }) => key));
  input.variables.forEach((variable) => {
    nonempty(variable.label, `variable ${variable.key} label`);
    nonempty(variable.unit, `variable ${variable.key} unit`);
    if (!stages.has(variable.stage_key)) throw new RangeError(`unknown stage ${variable.stage_key}`);
    if (!categories.has(variable.category_key)) throw new RangeError(`unknown category ${variable.category_key}`);
  });
}

function validateRow<Cohort extends string>(
  row: ScalarEnsembleRow<Cohort>,
  variableKeys: readonly string[],
  cohortKeys: ReadonlySet<Cohort>,
): void {
  if (row.row_id !== scalarEnsembleRowId(row.trial_index, row.series_id)) {
    throw new RangeError("row_id must equal its composite identity");
  }
  if (!cohortKeys.has(row.cohort)) throw new RangeError(`unknown row cohort ${row.cohort}`);
  const valueKeys = Object.keys(row.values).sort();
  if (valueKeys.join("\u0000") !== [...variableKeys].sort().join("\u0000")) {
    throw new RangeError("row values must contain exactly the declared variable keys");
  }
  Object.entries(row.values).forEach(([key, value]) => {
    if (value !== null && !Number.isFinite(value)) {
      throw new RangeError(`row value ${key} must be finite or null`);
    }
  });
  Object.entries(row.attributes ?? {}).forEach(([key, value]) => {
    nonempty(key, "attribute key");
    if (value !== null && typeof value !== "string") {
      throw new RangeError(`row attribute ${key} must be a string or null`);
    }
    if (value !== null) nonempty(value, `row attribute ${key}`);
  });
}

const frozenLabels = (
  definitions: readonly { readonly key: string; readonly label: string }[],
): readonly ScalarEnsembleStage[] => Object.freeze(definitions.map(({ key, label }) =>
  Object.freeze({ key, label }),
));

const frozenVariables = (
  variables: readonly ScalarVariableDefinition[],
): readonly ScalarVariableDefinition[] => Object.freeze(variables.map((variable) => Object.freeze({
  key: variable.key, label: variable.label, unit: variable.unit,
  stage_key: variable.stage_key, category_key: variable.category_key,
})));

function frozenRows<Cohort extends string>(
  rows: readonly ScalarEnsembleRow<Cohort>[],
): readonly ScalarEnsembleRow<Cohort>[] {
  return Object.freeze(rows.map((row) => Object.freeze({
    row_id: row.row_id,
    trial_index: row.trial_index,
    ...(row.series_id === undefined ? {} : { series_id: row.series_id }),
    cohort: row.cohort,
    values: Object.freeze({ ...row.values }),
    ...(row.attributes === undefined
      ? {}
      : { attributes: Object.freeze({ ...row.attributes }) }),
  })));
}

/** Validate and freeze a complete scalar ensemble at its adapter boundary. */
export function createScalarEnsemble<Cohort extends string>(
  input: ScalarEnsembleInput<Cohort>,
): ScalarEnsembleResult<Cohort> {
  validateDefinitions(input);
  const variableKeys = input.variables.map(({ key }) => key);
  const cohortKeys = new Set(input.cohorts.map(({ key }) => key));
  input.rows.forEach((row) => validateRow(row, variableKeys, cohortKeys));
  const rowIds = input.rows.map(({ row_id }) => row_id);
  if (new Set(rowIds).size !== rowIds.length) throw new RangeError("row_id values must be unique");
  return Object.freeze({
    schema_version: SCALAR_ENSEMBLE_SCHEMA_VERSION,
    result_id: input.result_id,
    provenance: Object.freeze({ ...input.provenance }),
    stages: frozenLabels(input.stages),
    categories: frozenLabels(input.categories),
    variables: frozenVariables(input.variables),
    cohorts: frozenLabels(input.cohorts) as readonly ScalarCohortDefinition<Cohort>[],
    rows: frozenRows(input.rows),
  });
}

const emptyAvailability = (): ScalarAvailability => ({
  total_rows: 0, x_finite: 0, y_finite: 0, paired_finite: 0, unavailable: 0,
});

function incrementAvailability(
  current: ScalarAvailability,
  xFinite: boolean,
  yFinite: boolean,
): ScalarAvailability {
  const pairedFinite = xFinite && yFinite;
  return {
    total_rows: current.total_rows + 1,
    x_finite: current.x_finite + Number(xFinite),
    y_finite: current.y_finite + Number(yFinite),
    paired_finite: current.paired_finite + Number(pairedFinite),
    unavailable: current.unavailable + Number(!pairedFinite),
  };
}

/** Derive paired-finite points and exact availability without changing raw rows. */
export function buildScalarEnsembleScatter<Cohort extends string>(
  ensemble: ScalarEnsembleResult<Cohort>,
  xKey: string,
  yKey: string,
): ScalarScatterData<Cohort> {
  const xVariable = ensemble.variables.find(({ key }) => key === xKey);
  const yVariable = ensemble.variables.find(({ key }) => key === yKey);
  if (!xVariable || !yVariable) throw new RangeError("scatter axes must be declared variables");
  const byCohort = Object.fromEntries(
    ensemble.cohorts.map(({ key }) => [key, emptyAvailability()]),
  ) as Record<Cohort, ScalarAvailability>;
  let overall = emptyAvailability();
  const points: ScalarScatterPoint<Cohort>[] = [];
  ensemble.rows.forEach((row) => {
    const x = row.values[xKey];
    const y = row.values[yKey];
    const xFinite = x !== null && Number.isFinite(x);
    const yFinite = y !== null && Number.isFinite(y);
    overall = incrementAvailability(overall, xFinite, yFinite);
    byCohort[row.cohort] = incrementAvailability(byCohort[row.cohort], xFinite, yFinite);
    if (xFinite && yFinite) {
      points.push({
        row_id: row.row_id, trial_index: row.trial_index,
        ...(row.series_id === undefined ? {} : { series_id: row.series_id }),
        cohort: row.cohort, x, y,
      });
    }
  });
  return Object.freeze({
    x_variable: xVariable, y_variable: yVariable, points: Object.freeze(points),
    availability: Object.freeze({
      overall: Object.freeze(overall),
      by_cohort: Object.freeze(Object.fromEntries(
        Object.entries(byCohort).map(([key, value]) => [key, Object.freeze(value)]),
      )) as Readonly<Record<Cohort, ScalarAvailability>>,
    }),
  });
}
