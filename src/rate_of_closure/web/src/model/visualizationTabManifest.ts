import manifestDocument from "../vendored/visualization_tabs.v1.json" with { type: "json" };

export type VisualizationSurface = "react" | "pyqt";
export type VisualizationClassification = "visual-first" | "form-led-live-preview" |
  "form-led-evidence" | "reference-utility";
export type VisualizationLandmarkKind = "visual" | "semantic-content";

export interface VisualizationTabEntry {
  surface: VisualizationSurface;
  tabId: string;
  purpose: string;
  dataPrerequisites: readonly string[];
  counterpartTabId: string;
  classification: VisualizationClassification;
  landmarkKind: VisualizationLandmarkKind;
  minimumVisibleHeightPx: number;
  primaryVisualLocator: string;
  states: Record<"empty" | "loading" | "result" | "error", string>;
}

const exactRecord = (value: unknown, keys: readonly string[], context: string) => {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error(`${context} must be an object`);
  }
  const result = Object.fromEntries(Object.entries(value));
  if (Object.keys(result).sort().join("|") !== [...keys].sort().join("|")) {
    throw new Error(`${context} fields must be exact`);
  }
  return result;
};

const text = (value: unknown, context: string): string => {
  if (typeof value !== "string" || value.trim().length === 0 || value.length > 200) {
    throw new Error(`${context} must be bounded nonempty text`);
  }
  return value;
};

const positiveInteger = (value: unknown, context: string): number => {
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value <= 0) {
    throw new Error(`${context} must be a positive safe integer`);
  }
  return value;
};

const nonemptyUniqueTextArray = (value: unknown, context: string): string[] => {
  if (!Array.isArray(value) || value.length === 0) {
    throw new Error(`${context} must be a nonempty array`);
  }
  const result = value.map((item) => text(item, context));
  if (new Set(result).size !== result.length) {
    throw new Error(`${context} must contain unique values`);
  }
  return result;
};

const deepFreeze = <T>(value: T): T => {
  if (typeof value === "object" && value !== null && !Object.isFrozen(value)) {
    Object.values(value).forEach((child) => deepFreeze(child));
    Object.freeze(value);
  }
  return value;
};

const tuple = (value: unknown, context: string): [number, number] => {
  if (!Array.isArray(value) || value.length !== 2) throw new Error(`${context} must be a pair`);
  const result: [number, number] = [positiveInteger(value[0], context), positiveInteger(value[1], context)];
  if (result.some((item) => item > 10_000)) throw new Error(`${context} exceeds its domain`);
  return result;
};

const surfaces: readonly VisualizationSurface[] = ["react", "pyqt"];
const classifications: readonly VisualizationClassification[] = [
  "visual-first", "form-led-live-preview", "form-led-evidence", "reference-utility",
];
const landmarkKinds: readonly VisualizationLandmarkKind[] = ["visual", "semantic-content"];

const parseEntry = (value: unknown): VisualizationTabEntry => {
  const entry = exactRecord(value, [
    "surface", "tab_id", "purpose", "data_prerequisites", "counterpart_tab_id",
    "classification", "landmark_kind", "minimum_visible_height_px",
    "primary_visual_locator", "states",
  ], "tab");
  const surfaceText = text(entry.surface, "surface");
  const classificationText = text(entry.classification, "classification");
  const landmarkText = text(entry.landmark_kind, "landmark kind");
  const surface = surfaces.find((candidate) => candidate === surfaceText);
  const classification = classifications.find((candidate) => candidate === classificationText);
  const landmarkKind = landmarkKinds.find((candidate) => candidate === landmarkText);
  if (surface === undefined) throw new Error("unknown surface");
  if (classification === undefined) throw new Error("unknown classification");
  if (landmarkKind === undefined) throw new Error("unknown landmark kind");
  if (classification === "reference-utility" && landmarkKind !== "semantic-content") {
    throw new Error("reference utilities require semantic content");
  }
  if (["visual-first", "form-led-live-preview", "form-led-evidence"].includes(classification) &&
      landmarkKind !== "visual") {
    throw new Error(`${classification} tabs require visual landmarks`);
  }
  const minimum = positiveInteger(entry.minimum_visible_height_px, "minimum visible height");
  if (minimum !== (landmarkKind === "visual" ? 240 : 1)) throw new Error("invalid landmark minimum");
  const states = exactRecord(entry.states, ["empty", "loading", "result", "error"], "states");
  return {
    surface,
    tabId: text(entry.tab_id, "tab id"),
    purpose: text(entry.purpose, "purpose"),
    dataPrerequisites: nonemptyUniqueTextArray(
      entry.data_prerequisites, "data prerequisite",
    ),
    counterpartTabId: text(entry.counterpart_tab_id, "counterpart tab id"),
    classification,
    landmarkKind,
    minimumVisibleHeightPx: minimum,
    primaryVisualLocator: (() => {
      const locator = text(entry.primary_visual_locator, "locator");
      if (surface === "pyqt" && /_(scroll|tabs|view)$/.test(locator)) {
        throw new Error("PyQt locator must identify a content leaf");
      }
      if (surface === "react" && landmarkKind === "visual" &&
          !/^(canvas|svg)\[aria-label/.test(locator)) {
        throw new Error("React visual locator must identify an accessible visual leaf");
      }
      return locator;
    })(),
    states: {
      empty: text(states.empty, "empty state"), loading: text(states.loading, "loading state"),
      result: text(states.result, "result state"), error: text(states.error, "error state"),
    },
  };
};

const parseEnvironment = (
  value: unknown, responsive: boolean, responsiveControlIds: readonly string[],
) => {
  const keys = responsive
    ? ["viewport_px", "additional_viewports_px", "responsive_minimum_visible_height_px",
      "minimum_visible_width_px", "responsive_minimum_visible_width_px",
      "responsive_control_locators", "dpi_scales"]
    : ["viewport_px", "minimum_visible_width_px", "dpi_scales"];
  const environment = exactRecord(value, keys, "reference environment");
  if (!Array.isArray(environment.dpi_scales) || environment.dpi_scales.some(
    (scale) => typeof scale !== "number" || !Number.isFinite(scale) || scale <= 0,
  )) throw new Error("DPI scales must be positive finite numbers");
  const additional = responsive ? environment.additional_viewports_px : [];
  if (!Array.isArray(additional)) throw new Error("additional viewports must be an array");
  const controls = responsive
    ? Object.fromEntries(Object.entries(exactRecord(
      environment.responsive_control_locators, responsiveControlIds, "responsive controls",
    )).map(([tabId, locator]) => [tabId, text(locator, "control locator")]))
    : {};
  const responsiveMinimumVisibleHeightPx = responsive
    ? positiveInteger(environment.responsive_minimum_visible_height_px, "responsive minimum") : 1;
  if (responsive && responsiveMinimumVisibleHeightPx < 180) {
    throw new Error("React responsive minimum must preserve meaningful visual height");
  }
  return deepFreeze({
    viewportPx: tuple(environment.viewport_px, "viewport"),
    additionalViewportsPx: additional.map((item) => tuple(item, "additional viewport")),
    responsiveMinimumVisibleHeightPx,
    minimumVisibleWidthPx: positiveInteger(environment.minimum_visible_width_px, "minimum width"),
    responsiveMinimumVisibleWidthPx: responsive
      ? positiveInteger(environment.responsive_minimum_visible_width_px, "responsive minimum width") : 1,
    responsiveControlLocators: controls,
    dpiScales: [...environment.dpi_scales],
  });
};

export const parseVisualizationTabManifest = (value: unknown) => {
  const document = exactRecord(value, ["schema_id", "schema_version", "artifact_policy",
    "reference_environments", "tabs"], "manifest");
  if (document.schema_id !== "rate-of-closure/visualization-tab-visibility" ||
      document.schema_version !== 1 || document.artifact_policy !== "diagnostic-only-not-approved-golden") {
    throw new Error("unsupported visualization manifest schema or policy");
  }
  const environments = exactRecord(document.reference_environments, ["react", "pyqt"], "environments");
  if (!Array.isArray(document.tabs)) throw new Error("tabs must be an array");
  const entries = document.tabs.map(parseEntry);
  const identities = entries.map((entry) => `${entry.surface}:${entry.tabId}`);
  if (new Set(identities).size !== identities.length) throw new Error("duplicate tab identity");
  const byIdentity = new Map(entries.map((entry) => [
    `${entry.surface}:${entry.tabId}`, entry,
  ]));
  entries.forEach((entry) => {
    const counterpartSurface = entry.surface === "react" ? "pyqt" : "react";
    const counterpart = byIdentity.get(`${counterpartSurface}:${entry.counterpartTabId}`);
    if (counterpart === undefined || counterpart.counterpartTabId !== entry.tabId) {
      throw new Error("visualization counterparts must exist and be reciprocal");
    }
  });
  const reactVisualIds = entries.filter(
    (entry) => entry.surface === "react" && entry.landmarkKind === "visual",
  ).map((entry) => entry.tabId);
  return deepFreeze({
    entries,
    environments: {
      react: parseEnvironment(environments.react, true, reactVisualIds),
      pyqt: parseEnvironment(environments.pyqt, false, []),
    },
  });
};

const parsed = parseVisualizationTabManifest(manifestDocument);
export const visualizationReferenceEnvironments = parsed.environments;
export const visualizationTabs = (surface: VisualizationSurface): readonly VisualizationTabEntry[] =>
  deepFreeze(parsed.entries.filter((entry) => entry.surface === surface));

export const auditRegisteredVisualizationTabs = (
  surface: VisualizationSurface, registered: readonly string[],
): string[] => {
  const documented = visualizationTabs(surface).map((entry) => entry.tabId);
  return [
    ...registered.filter((tabId) => !documented.includes(tabId))
      .map((tabId) => `missing manifest entry for ${surface} tab ${tabId}`),
    ...documented.filter((tabId) => !registered.includes(tabId))
      .map((tabId) => `unregistered manifest entry for ${surface} tab ${tabId}`),
  ];
};
