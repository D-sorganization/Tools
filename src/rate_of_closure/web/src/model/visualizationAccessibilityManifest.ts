import accessibilityDocument from "../../../visualization_accessibility.v1.json" with { type: "json" };

import { visualizationTabs, type VisualizationSurface } from "./visualizationTabManifest";

export type AccessibilityEvidence = "axe-core-wcag-a-aa-through-2.2" |
  "named-visible-focusable-semantic-controls";

export interface AccessibilityEvidenceTab {
  surface: VisualizationSurface;
  tabId: string;
  evidence: AccessibilityEvidence;
}

const exactRecord = (
  value: unknown, keys: readonly string[], context: string,
): Record<string, unknown> => {
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
    throw new Error(`${context} must be bounded text`);
  }
  return value;
};

const deepFreeze = <T>(value: T): T => {
  if (typeof value === "object" && value !== null && !Object.isFrozen(value)) {
    Object.values(value).forEach((child) => deepFreeze(child));
    Object.freeze(value);
  }
  return value;
};

const evidenceBySurface = {
  react: "axe-core-wcag-a-aa-through-2.2",
  pyqt: "named-visible-focusable-semantic-controls",
} as const;

export const parseVisualizationAccessibilityManifest = (value: unknown) => {
  const document = exactRecord(value, [
    "schema_id", "schema_version", "automated_claim", "manual_at", "tabs",
  ], "manifest");
  const manual = exactRecord(document.manual_at, ["status", "protocol_path"], "manual AT");
  if (document.schema_id !== "rate-of-closure/visualization-accessibility-evidence" ||
      document.schema_version !== 1 || document.automated_claim !==
      "protected-automated-semantics-not-manual-at-qualification" ||
      manual.status !== "protocol-ready-human-execution-required" ||
      manual.protocol_path !== "docs/development/rate-visualization-at-protocol.md") {
    throw new Error("unsupported accessibility manifest");
  }
  if (!Array.isArray(document.tabs)) throw new Error("tabs must be an array");
  const tabs = document.tabs.map((raw): AccessibilityEvidenceTab => {
    const entry = exactRecord(raw, ["surface", "tab_id", "evidence"], "tab");
    const rawSurface = entry.surface;
    if (rawSurface !== "react" && rawSurface !== "pyqt") {
      throw new Error("unknown accessibility surface");
    }
    const evidence = text(entry.evidence, "evidence");
    if (evidence !== evidenceBySurface[rawSurface]) {
      throw new Error("unsupported accessibility evidence");
    }
    return {
      surface: rawSurface,
      tabId: text(entry.tab_id, "tab id"),
      evidence: evidenceBySurface[rawSurface],
    };
  });
  const identities = tabs.map((entry) => `${entry.surface}:${entry.tabId}`);
  if (new Set(identities).size !== identities.length) {
    throw new Error("duplicate accessibility tab identity");
  }
  const expected = (["react", "pyqt"] as const).flatMap((surface) =>
    visualizationTabs(surface).map((entry) => `${surface}:${entry.tabId}`));
  if (identities.join("|") !== expected.join("|")) {
    throw new Error("accessibility tabs must exactly match visibility authority");
  }
  return deepFreeze({
    schemaId: document.schema_id,
    schemaVersion: document.schema_version,
    automatedClaim: document.automated_claim,
    manualAt: {
      status: manual.status,
      protocolPath: text(manual.protocol_path, "manual AT protocol path"),
    },
    tabs,
  });
};

const parsed = parseVisualizationAccessibilityManifest(accessibilityDocument);
export const visualizationAccessibilityTabs = (
  surface: VisualizationSurface,
): readonly AccessibilityEvidenceTab[] => parsed.tabs.filter((entry) => entry.surface === surface);
export const visualizationManualAt = parsed.manualAt;
