import { afterEach, describe, expect, it, vi } from "vitest";

import fixture from "./__fixtures__/regional_ground_execution_result_golden_v1.json";
import { parseRegionalGroundExecutionResult } from "./regionalGroundExecutionResult";
import {
  downloadRegionalGroundExecutionResult,
  downloadRegionalGroundExecutionRowsCsv,
} from "./regionalGroundExecutionResultFiles";

describe("regional-ground execution-result download", () => {
  afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals(); });

  it("downloads canonical JSON and always revokes its object URL", () => {
    const result = parseRegionalGroundExecutionResult(fixture.result);
    const createUrl = vi.fn(() => "blob:regional-ground-result");
    const revokeUrl = vi.fn();
    vi.stubGlobal("URL", { createObjectURL: createUrl, revokeObjectURL: revokeUrl });
    const click = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});

    downloadRegionalGroundExecutionResult(result);

    expect(click).toHaveBeenCalledOnce();
    expect(createUrl).toHaveBeenCalledOnce();
    expect(revokeUrl).toHaveBeenCalledWith("blob:regional-ground-result");
  });

  it("downloads every scalar row as spreadsheet-safe CSV", () => {
    const result = parseRegionalGroundExecutionResult(fixture.result);
    const createUrl = vi.fn(() => "blob:regional-ground-csv");
    const revokeUrl = vi.fn();
    vi.stubGlobal("URL", { createObjectURL: createUrl, revokeObjectURL: revokeUrl });
    const click = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});

    downloadRegionalGroundExecutionRowsCsv(result);

    expect(click).toHaveBeenCalledOnce();
    expect(createUrl).toHaveBeenCalledOnce();
    expect(revokeUrl).toHaveBeenCalledWith("blob:regional-ground-csv");
  });
});
