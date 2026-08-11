import { afterEach, describe, expect, it, vi } from "vitest";

import fixture from "./__fixtures__/ground_regional_execution_golden_v1.json";
import { parseGroundRegionalExecutionResult } from "./groundRegionalExecution";
import {
  canonicalRegionalExecutionEvidenceJson,
  downloadRegionalExecutionEvidence,
} from "./regionalExecutionFiles";

describe("regional execution evidence export", () => {
  const result = parseGroundRegionalExecutionResult(fixture.representable.result);

  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("retains the exact validated canonical envelope", () => {
    const text = canonicalRegionalExecutionEvidenceJson(result);

    expect(JSON.parse(text)).toEqual(fixture.representable.result);
    expect(parseGroundRegionalExecutionResult(JSON.parse(text))).toEqual(result);
  });

  it("downloads canonical bytes and always revokes its object URL", () => {
    const createUrl = vi.fn(() => "blob:regional-execution");
    const revokeUrl = vi.fn();
    vi.stubGlobal("URL", { createObjectURL: createUrl, revokeObjectURL: revokeUrl });
    const click = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});

    downloadRegionalExecutionEvidence(result);

    expect(createUrl).toHaveBeenCalledOnce();
    expect(click).toHaveBeenCalledOnce();
    expect(revokeUrl).toHaveBeenCalledWith("blob:regional-execution");
  });
});
