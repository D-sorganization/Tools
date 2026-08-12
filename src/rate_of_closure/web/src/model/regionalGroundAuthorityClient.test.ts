import { describe, expect, it, vi } from "vitest";

import jobFixture from "./__fixtures__/regional_ground_execution_job_golden_v1.json";
import statusFixture from "./__fixtures__/regional_ground_authority_job_status_golden_v1.json";
import resultFixture from "./__fixtures__/regional_ground_execution_result_golden_v1.json";
import { canonicalGroundJson } from "./flightGroundContract";
import {
  createRegionalGroundAuthorityClient,
  parseRegionalGroundAuthorityJobStatus,
  regionalGroundAuthorityJobStatusFromJson,
  RegionalGroundAuthorityRequestError,
  REGIONAL_GROUND_AUTHORITY_JOBS_PATH,
  REGIONAL_GROUND_JOB_PREPARATIONS_PATH,
  stableRegionalGroundAuthorityJobStatusJson,
} from "./regionalGroundAuthorityClient";
import { parseRegionalGroundExecutionJob } from "./regionalGroundExecutionJob";
import {
  REGIONAL_GROUND_JOB_PREPARATION_REQUEST_SCHEMA,
  type RegionalGroundJobPreparationRequest,
} from "./regionalGroundJobPreparationRequest";

const job = parseRegionalGroundExecutionJob(jobFixture.job);
const status = {
  schema_version: "rate-of-closure/regional-ground-authority-job-status/v1",
  job_id: job.job_id,
  job_sha256: job.job_sha256,
  status: "queued",
  completed: 0,
  total: job.execution_options.max_trials,
  result_available: false,
  failure: null,
};
const preparationRequest: RegionalGroundJobPreparationRequest = {
  schema_version: REGIONAL_GROUND_JOB_PREPARATION_REQUEST_SCHEMA,
  unit_system: "SI",
  job_id: job.job_id,
  launch: job.launch,
  variation_request: job.variation_request,
};

const jsonResponse = (value: unknown): Response => new Response(JSON.stringify(value), {
  status: 200,
  headers: { "content-type": "application/json" },
});

describe("regional-ground authority REST contracts", () => {
  it("prepares a strict job snapshot without submitting it", async () => {
    const fetcher = vi.fn().mockResolvedValue(jsonResponse(jobFixture.job));
    const client = createRegionalGroundAuthorityClient(fetcher);
    const controller = new AbortController();

    const result = await client.prepare(preparationRequest, controller.signal);

    expect(result.job_id).toBe(preparationRequest.job_id);
    expect(fetcher).toHaveBeenCalledTimes(1);
    expect(fetcher).toHaveBeenCalledWith(
      REGIONAL_GROUND_JOB_PREPARATIONS_PATH,
      expect.objectContaining({ method: "POST", signal: controller.signal }),
    );
    expect(JSON.parse(fetcher.mock.calls[0][1].body as string)).toEqual(preparationRequest);
  });

  it("fails closed on oversized, duplicate, invalid, or substituted prepared jobs", async () => {
    const jobText = JSON.stringify(jobFixture.job);
    const duplicate = jobText.replace(
      `"job_id":"${job.job_id}"`,
      `"job_id":"${job.job_id}","job_id":"${job.job_id}"`,
    );
    const responses = [
      new Response("{}", {
        status: 200,
        headers: { "content-length": "1048577" },
      }),
      new Response(duplicate, { status: 200 }),
      jsonResponse({ ...jobFixture.job, unexpected: true }),
      jsonResponse(jobFixture.job),
    ];
    const requests: RegionalGroundJobPreparationRequest[] = [
      preparationRequest,
      preparationRequest,
      preparationRequest,
      { ...preparationRequest, job_id: "different-job" },
    ];

    for (const [index, response] of responses.entries()) {
      const client = createRegionalGroundAuthorityClient(vi.fn().mockResolvedValue(response));
      await expect(client.prepare(requests[index])).rejects.toThrow();
    }
  });

  it.each([
    [400, {
      code: "invalid_preparation",
      detail: "Regional-ground job preparation request is invalid.",
    }, "invalid_preparation"],
    [503, {
      code: "preparation_unavailable",
      detail: "Qualified regional-ground job preparation is unavailable.",
    }, "preparation_unavailable"],
    [422, {
      code: "preparation_failed",
      detail: "Qualified regional-ground job preparation failed.",
    }, "preparation_failed"],
  ])("types preparation HTTP %i failures", async (httpStatus, body, expectedCode) => {
    const response = new Response(JSON.stringify(body), { status: httpStatus });
    const client = createRegionalGroundAuthorityClient(vi.fn().mockResolvedValue(response));

    const error = await client.prepare(preparationRequest).catch((reason: unknown) => reason);

    expect(error).toBeInstanceOf(RegionalGroundAuthorityRequestError);
    expect(error).toMatchObject({ httpStatus, code: expectedCode });
  });

  it("propagates preparation aborts without publishing a substituted job", async () => {
    const controller = new AbortController();
    const fetcher = vi.fn((_path: RequestInfo | URL, init?: RequestInit) =>
      new Promise<Response>((_resolve, reject) => {
        init?.signal?.addEventListener("abort", () => {
          reject(new DOMException("aborted", "AbortError"));
        }, { once: true });
      }));
    const client = createRegionalGroundAuthorityClient(fetcher);
    const pending = client.prepare(preparationRequest, controller.signal);

    controller.abort();

    await expect(pending).rejects.toMatchObject({ name: "AbortError" });
    expect(fetcher).toHaveBeenCalledTimes(1);
  });

  it("validates preparation requests before performing network I/O", async () => {
    const fetcher = vi.fn().mockResolvedValue(jsonResponse(jobFixture.job));
    const client = createRegionalGroundAuthorityClient(fetcher);
    const invalid = { ...preparationRequest, job_id: "../route-escape" };

    await expect(client.prepare(invalid as RegionalGroundJobPreparationRequest))
      .rejects.toThrow(/job_id/i);
    expect(fetcher).not.toHaveBeenCalled();
  });

  it("matches every Python-produced golden status byte-for-byte and semantically", () => {
    expect(statusFixture.fixture_schema).toBe(
      "rate-of-closure/regional-ground-authority-job-status-golden/v1",
    );
    expect(new Set(statusFixture.cases.map((item) => item.status))).toEqual(new Set([
      "queued", "running", "cancel_requested", "succeeded", "failed", "cancelled",
    ]));
    expect(new Set(statusFixture.cases.flatMap((item) =>
      item.failure === null ? [] : [item.failure.stage]))).toEqual(new Set([
      "authority_restart", "cancellation_callback", "preflight", "executor", "validation", "progress_callback",
      "publication", "runner", "result_validation",
    ]));

    for (const item of statusFixture.cases) {
      const golden = canonicalGroundJson(item);
      const parsed = regionalGroundAuthorityJobStatusFromJson(golden, job);
      expect(stableRegionalGroundAuthorityJobStatusJson(parsed, job)).toBe(golden);
    }
  });

  it("parses, freezes, and binds an exact status to its source job", () => {
    const parsed = parseRegionalGroundAuthorityJobStatus(status, job);

    expect(parsed).toEqual(status);
    expect(Object.isFrozen(parsed)).toBe(true);
  });

  it("rejects extra, substituted, Boolean, and semantically invalid status fields", () => {
    expect(() => parseRegionalGroundAuthorityJobStatus({
      ...status, unexpected: true,
    }, job)).toThrow(/fields/i);
    expect(() => parseRegionalGroundAuthorityJobStatus({
      ...status, job_sha256: "0".repeat(64),
    }, job)).toThrow(/job_sha256/i);
    expect(() => parseRegionalGroundAuthorityJobStatus({
      ...status, completed: false,
    }, job)).toThrow(/completed/i);
    expect(() => parseRegionalGroundAuthorityJobStatus({
      ...status, status: "succeeded", result_available: false,
    }, job)).toThrow(/succeeded.*result/i);
    expect(() => parseRegionalGroundAuthorityJobStatus({
      ...status, status: "failed", failure: null,
    }, job)).toThrow(/failure/i);
    expect(() => parseRegionalGroundAuthorityJobStatus({
      ...status,
      status: "failed",
      failure: { code: "internal_exception", stage: "runner" },
    }, job)).toThrow(/failure code/i);
  });

  it.each([
    ["nonfinite completed", { ...status, completed: Number.NaN }, /finite/i],
    ["unsafe total", { ...status, total: Number.MAX_SAFE_INTEGER + 1 }, /safe/i],
    ["mismatched job_id", { ...status, job_id: "other-job" }, /job_id/i],
    ["mismatched total", { ...status, total: 5 }, /total/i],
    ["numeric Boolean", { ...status, result_available: 1 }, /result_available/i],
  ])("rejects %s from the Python status boundary", (_name, changed, message) => {
    expect(() => parseRegionalGroundAuthorityJobStatus(changed, job)).toThrow(message);
  });

  it("rejects duplicate and oversized status JSON before publication", () => {
    const source = canonicalGroundJson(status);
    const duplicate = source.replace('"job_id":', '"job_id":"duplicate","job_id":');
    expect(() => regionalGroundAuthorityJobStatusFromJson(duplicate, job)).toThrow(/duplicate/i);
    expect(() => regionalGroundAuthorityJobStatusFromJson(
      source + " ".repeat(4_096), job,
    )).toThrow(/byte limit/i);
  });

  it("uses the reserved same-origin submit/status/cancel/result routes and strict bodies", async () => {
    const fetcher = vi.fn()
      .mockResolvedValueOnce(jsonResponse(status))
      .mockResolvedValueOnce(jsonResponse({ ...status, status: "running" }))
      .mockResolvedValueOnce(jsonResponse({ ...status, status: "cancel_requested" }))
      .mockResolvedValueOnce(jsonResponse(resultFixture.result));
    const client = createRegionalGroundAuthorityClient(fetcher);
    const controller = new AbortController();

    await client.submit(job, controller.signal);
    await client.status(job, controller.signal);
    await client.cancel(job, controller.signal);
    const result = await client.result(job, controller.signal);

    const encodedId = encodeURIComponent(job.job_id);
    expect(fetcher.mock.calls.map(([path, init]) => [path, init.method])).toEqual([
      [REGIONAL_GROUND_AUTHORITY_JOBS_PATH, "POST"],
      [`${REGIONAL_GROUND_AUTHORITY_JOBS_PATH}/${encodedId}`, "GET"],
      [`${REGIONAL_GROUND_AUTHORITY_JOBS_PATH}/${encodedId}/cancel`, "POST"],
      [`${REGIONAL_GROUND_AUTHORITY_JOBS_PATH}/${encodedId}/result`, "GET"],
    ]);
    expect(fetcher.mock.calls.every(([, init]) => init.signal === controller.signal)).toBe(true);
    expect(JSON.parse(fetcher.mock.calls[0][1].body as string)).toEqual(jobFixture.job);
    expect(result.job_id).toBe(job.job_id);
  });

  it("fails closed on oversized, duplicate, or job-mismatched responses", async () => {
    const duplicate = JSON.stringify(status).replace(
      `"job_id":"${job.job_id}"`,
      `"job_id":"${job.job_id}","job_id":"${job.job_id}"`,
    );
    const responses = [
      new Response("{}", { status: 200, headers: { "content-length": "5000" } }),
      new Response(duplicate, { status: 200 }),
      jsonResponse({ ...status, job_sha256: "0".repeat(64) }),
    ];

    for (const response of responses) {
      const client = createRegionalGroundAuthorityClient(vi.fn().mockResolvedValue(response));
      await expect(client.status(job)).rejects.toThrow();
    }
  });

  it.each([
    [401, { detail: "Local authority authentication required." }, "authentication_required"],
    [404, { code: "job_not_found", detail: "Job was not found." }, "job_not_found"],
    [503, {
      code: "execution_unavailable",
      detail: "Qualified regional-ground execution is unavailable.",
    }, "execution_unavailable"],
    [500, "not-json", "authority_error"],
  ])("types HTTP %i failures without publishing an invalid status", async (
    httpStatus,
    body,
    expectedCode,
  ) => {
    const response = new Response(
      typeof body === "string" ? body : JSON.stringify(body),
      { status: httpStatus },
    );
    const client = createRegionalGroundAuthorityClient(vi.fn().mockResolvedValue(response));

    const error = await client.status(job).catch((reason: unknown) => reason);

    expect(error).toBeInstanceOf(RegionalGroundAuthorityRequestError);
    expect(error).toMatchObject({ httpStatus, code: expectedCode });
  });

  it("propagates AbortError and never substitutes a status", async () => {
    const controller = new AbortController();
    const fetcher = vi.fn((_path: RequestInfo | URL, init?: RequestInit) =>
      new Promise<Response>((_resolve, reject) => {
        init?.signal?.addEventListener("abort", () => {
          reject(new DOMException("aborted", "AbortError"));
        }, { once: true });
      }));
    const client = createRegionalGroundAuthorityClient(fetcher);
    const request = client.status(job, controller.signal);

    controller.abort();

    await expect(request).rejects.toMatchObject({ name: "AbortError" });
    expect(fetcher).toHaveBeenCalledTimes(1);
  });

  it("validates a job before constructing a route or performing network I/O", async () => {
    const fetcher = vi.fn().mockResolvedValue(jsonResponse(status));
    const client = createRegionalGroundAuthorityClient(fetcher);
    const invalid = { ...job, job_id: "../route-escape" };

    await expect(client.status(invalid as typeof job)).rejects.toThrow(/job_id/i);
    await expect(client.cancel(invalid as typeof job)).rejects.toThrow(/job_id/i);
    await expect(client.result(invalid as typeof job)).rejects.toThrow(/job_id/i);
    expect(fetcher).not.toHaveBeenCalled();
  });
});
