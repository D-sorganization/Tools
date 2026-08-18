/** Canonical browser download boundary for validated regional execution evidence. */

import {
  MAX_GROUND_REGIONAL_EXECUTION_WIRE_BYTES,
  stableGroundRegionalExecutionJson,
  type GroundRegionalExecutionResult,
} from "./groundRegionalExecution";

export const canonicalRegionalExecutionEvidenceJson = (
  result: GroundRegionalExecutionResult,
): string => {
  const text = stableGroundRegionalExecutionJson(result);
  if (new TextEncoder().encode(text).byteLength >
    MAX_GROUND_REGIONAL_EXECUTION_WIRE_BYTES) {
    throw new RangeError("regional execution evidence exceeds maximum wire size");
  }
  return text;
};

export const downloadRegionalExecutionEvidence = (
  result: GroundRegionalExecutionResult,
): void => {
  const text = canonicalRegionalExecutionEvidenceJson(result);
  const url = URL.createObjectURL(
    new Blob([text], { type: "application/json;charset=utf-8" }),
  );
  try {
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "regional-execution-evidence.json";
    anchor.click();
  } finally {
    URL.revokeObjectURL(url);
  }
};
