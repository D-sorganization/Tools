/** Exact public runtime mode for the Rate of Closure web application. */

import { exact, record, text } from "./flightGroundValidation";

export const WEB_RUNTIME_SCHEMA = "rate-of-closure/web-runtime/v1" as const;
export const WEB_RUNTIME_ELEMENT_ID = "rate-of-closure-web-runtime" as const;
export const REGIONAL_GROUND_AUTHORITY_ROOT =
  "/api/rate-of-closure/v1" as const;
const DEVELOPMENT_REVISION = "development";
const REVISION_PATTERN = /^[0-9a-f]{40}$/;
const STATIC_FIELDS = ["schema_version", "mode", "release_revision"] as const;
const COMPANION_FIELDS = [...STATIC_FIELDS, "authority_path"] as const;

export interface StaticInspectionWebRuntime {
  readonly schema_version: typeof WEB_RUNTIME_SCHEMA;
  readonly mode: "static_inspection";
  readonly release_revision: string;
}

export interface LocalCompanionWebRuntime {
  readonly schema_version: typeof WEB_RUNTIME_SCHEMA;
  readonly mode: "local_companion";
  readonly release_revision: string;
  readonly authority_path: typeof REGIONAL_GROUND_AUTHORITY_ROOT;
}

export type WebRuntime = StaticInspectionWebRuntime | LocalCompanionWebRuntime;

export const STATIC_INSPECTION_WEB_RUNTIME: StaticInspectionWebRuntime =
  Object.freeze({
    schema_version: WEB_RUNTIME_SCHEMA,
    mode: "static_inspection",
    release_revision: DEVELOPMENT_REVISION,
  });

export const LOCAL_COMPANION_WEB_RUNTIME: LocalCompanionWebRuntime =
  Object.freeze({
    schema_version: WEB_RUNTIME_SCHEMA,
    mode: "local_companion",
    release_revision: DEVELOPMENT_REVISION,
    authority_path: REGIONAL_GROUND_AUTHORITY_ROOT,
  });

const releaseRevision = (value: unknown): string => {
  const revision = text(value, "web runtime release_revision");
  if (revision !== DEVELOPMENT_REVISION && !REVISION_PATTERN.test(revision)) {
    throw new RangeError("web runtime release_revision must be development or a commit");
  }
  return revision;
};

/** Parse one strict descriptor without accepting authority URLs or secrets. */
export const parseWebRuntime = (value: unknown): WebRuntime => {
  const item = record(value, "web runtime");
  if (item.schema_version !== WEB_RUNTIME_SCHEMA) {
    throw new RangeError("unsupported web runtime schema");
  }
  if (item.mode === "static_inspection") {
    exact(item, STATIC_FIELDS, "static-inspection web runtime");
    return Object.freeze({
      schema_version: WEB_RUNTIME_SCHEMA,
      mode: item.mode,
      release_revision: releaseRevision(item.release_revision),
    });
  }
  if (item.mode !== "local_companion") {
    throw new RangeError("unsupported web runtime mode");
  }
  exact(item, COMPANION_FIELDS, "local-companion web runtime");
  if (item.authority_path !== REGIONAL_GROUND_AUTHORITY_ROOT) {
    throw new RangeError("local-companion authority_path must be fixed and relative");
  }
  return Object.freeze({
    schema_version: WEB_RUNTIME_SCHEMA,
    mode: item.mode,
    release_revision: releaseRevision(item.release_revision),
    authority_path: item.authority_path,
  });
};

/** Read exactly one embedded descriptor before the application is mounted. */
export const parseEmbeddedWebRuntime = (source: Document): WebRuntime => {
  const candidates = source.querySelectorAll(`#${WEB_RUNTIME_ELEMENT_ID}`);
  if (candidates.length !== 1) {
    throw new RangeError("exactly one embedded web runtime descriptor is required");
  }
  const candidate = candidates[0];
  if (!(candidate instanceof HTMLScriptElement) ||
      candidate.type !== "application/json") {
    throw new RangeError("embedded web runtime descriptor must be JSON script data");
  }
  try {
    return parseWebRuntime(JSON.parse(candidate.textContent ?? ""));
  } catch {
    throw new RangeError("embedded web runtime descriptor is invalid");
  }
};

/** Development uses its authenticated proxy; production trusts only embedded data. */
export const applicationWebRuntime = (source: Document): WebRuntime =>
  import.meta.env.DEV ? LOCAL_COMPANION_WEB_RUNTIME : parseEmbeddedWebRuntime(source);
