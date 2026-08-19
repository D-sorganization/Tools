import { describe, expect, it, vi } from "vitest";

import { runCapabilityOptimization } from "./capabilityRun";
import {
  buildCapabilityWorkflow,
  defaultCapabilityWorkflowInputs,
} from "./capabilityWorkflow";

describe("capability optimization run facade", () => {
  it("retains every observation and reports exact final progress", () => {
    const document = buildCapabilityWorkflow({
      ...defaultCapabilityWorkflowInputs(),
      candidateBudget: 1, ensembleSize: 2, alternativesCount: 1,
    });
    const progress = vi.fn();

    const output = runCapabilityOptimization(document, progress);

    expect(output.result.evaluationsAttempted).toBe(2);
    expect(output.ensemble.rows).toHaveLength(2);
    expect(progress).toHaveBeenLastCalledWith({ completed: 2, total: 2 });
  });
});
