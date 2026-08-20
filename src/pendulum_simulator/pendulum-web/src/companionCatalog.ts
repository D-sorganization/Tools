import rawCatalog from "../../src/double_pendulum_golf/resources/companion_catalog.json";

export type CompanionModel = "double" | "triple" | "golfer";

export interface CompanionExperiment {
  id: string;
  title: string;
  model: CompanionModel;
  purpose: string;
  hypothesis: string;
  falsifier: string;
  workflow: string[];
  tips: string[];
  observables: string[];
  limitations: string[];
}

export interface GlossaryTerm {
  id: string;
  term: string;
  definition: string;
  plain_language: string;
  units: string;
  caution: string;
}

export interface CompanionCatalog {
  schema_version: string;
  title: string;
  scientific_status: string;
  experiments: CompanionExperiment[];
  glossary: GlossaryTerm[];
}

const MODELS = new Set<CompanionModel>(["double", "triple", "golfer"]);

function requireUniqueIds(items: Array<{ id: string }>, name: string): void {
  const identifiers = items.map((item) => item.id);
  if (new Set(identifiers).size !== identifiers.length) {
    throw new Error(`${name} IDs must be unique`);
  }
}

function validateCatalog(value: unknown): CompanionCatalog {
  if (typeof value !== "object" || value === null) {
    throw new TypeError("companion catalog must be an object");
  }
  const catalog = value as CompanionCatalog;
  if (!Array.isArray(catalog.experiments) || !Array.isArray(catalog.glossary)) {
    throw new TypeError("companion catalog collections must be arrays");
  }
  for (const experiment of catalog.experiments) {
    if (!MODELS.has(experiment.model)) {
      throw new Error(`unsupported companion model: ${experiment.model}`);
    }
    for (const field of [
      experiment.workflow,
      experiment.tips,
      experiment.observables,
      experiment.limitations,
    ]) {
      if (!Array.isArray(field) || field.length === 0) {
        throw new Error(`experiment ${experiment.id} has an empty learning contract`);
      }
    }
  }
  requireUniqueIds(catalog.experiments, "experiment");
  requireUniqueIds(catalog.glossary, "glossary");
  return catalog;
}

export const COMPANION_CATALOG = validateCatalog(rawCatalog);

export function searchGlossary(query: string): GlossaryTerm[] {
  const needle = query.trim().toLocaleLowerCase();
  if (!needle) return COMPANION_CATALOG.glossary;
  return COMPANION_CATALOG.glossary.filter((term) =>
    [term.term, term.definition, term.plain_language, term.caution]
      .join(" ")
      .toLocaleLowerCase()
      .includes(needle),
  );
}
