import { afterEach, describe, expect, it, vi } from "vitest";

import fixture from "./__fixtures__/regional_ground_execution_job_golden_v1.json";
import {
  MAX_REGIONAL_GROUND_EXECUTION_JOB_BYTES,
  parseRegionalGroundExecutionJob,
} from "./regionalGroundExecutionJob";
import {
  downloadRegionalGroundExecutionJob,
  readRegionalGroundExecutionJobFile,
} from "./regionalGroundExecutionJobFiles";

afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals(); });

describe("regional-ground execution-job files", () => {
  const bytes = (text: string): ArrayBuffer =>
    Uint8Array.from(new TextEncoder().encode(text)).buffer;

  it("reads one bounded exact job", async () => {
    const source = JSON.stringify(fixture.job);
    const job = await readRegionalGroundExecutionJobFile({
      name: "study.json",
      size: new TextEncoder().encode(source).byteLength,
      arrayBuffer: async () => bytes(source),
    });

    expect(job.job_sha256).toBe(fixture.job_sha256);
  });

  it("rejects oversized metadata before reading", async () => {
    let read = false;
    await expect(readRegionalGroundExecutionJobFile({
      name: "oversized.json",
      size: MAX_REGIONAL_GROUND_EXECUTION_JOB_BYTES + 1,
      arrayBuffer: async () => { read = true; return bytes("{}"); },
    })).rejects.toThrow(/maximum wire size/i);
    expect(read).toBe(false);
  });

  it("rejects underreported actual bytes and malformed UTF-8", async () => {
    const oversized = " ".repeat(MAX_REGIONAL_GROUND_EXECUTION_JOB_BYTES + 1);
    await expect(readRegionalGroundExecutionJobFile({
      name: "underreported.json", size: 1, arrayBuffer: async () => bytes(oversized),
    })).rejects.toThrow(/maximum wire size/i);
    await expect(readRegionalGroundExecutionJobFile({
      name: "invalid-utf8.json", size: 2,
      arrayBuffer: async () => Uint8Array.from([0xc3, 0x28]).buffer,
    })).rejects.toThrow(/valid utf-8/i);
  });

  it("downloads canonical job bytes and revokes the object URL", () => {
    const job = parseRegionalGroundExecutionJob(fixture.job);
    let captured: Blob | undefined;
    const createUrl = vi.fn((blob: Blob) => {
      captured = blob;
      return "blob:regional-ground-job";
    });
    const revokeUrl = vi.fn();
    vi.stubGlobal("URL", { createObjectURL: createUrl, revokeObjectURL: revokeUrl });
    const click = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});

    downloadRegionalGroundExecutionJob(job);

    expect(click).toHaveBeenCalledOnce();
    expect(createUrl).toHaveBeenCalledOnce();
    expect(revokeUrl).toHaveBeenCalledWith("blob:regional-ground-job");
    if (captured === undefined) throw new Error("download did not create a Blob");
    expect(captured.type).toBe("application/json;charset=utf-8");
  });
});
