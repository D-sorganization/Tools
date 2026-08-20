import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import jobFixture from "./__fixtures__/regional_ground_execution_job_golden_v1.json";
import resultFixture from "./__fixtures__/regional_ground_execution_result_golden_v1.json";
import { unavailableRegionalGroundAuthorityCapability } from "./regionalGroundAuthority";
import { parseRegionalGroundExecutionJob } from "./regionalGroundExecutionJob";
import { parseRegionalGroundExecutionResult } from "./regionalGroundExecutionResult";
import {
  RegionalGroundExecutionPresentation,
} from "./regionalGroundExecutionPresentation";
import {
  initialRegionalGroundExecutionPresentation,
  presentRegionalGroundCancelRequested,
  presentRegionalGroundCancelled,
  presentRegionalGroundFailure,
  presentRegionalGroundProgress,
  presentRegionalGroundResult,
} from "./regionalGroundExecutionPresentationModel";

const job = parseRegionalGroundExecutionJob(jobFixture.job);
const capability = unavailableRegionalGroundAuthorityCapability(
  "execution_profile_unqualified",
  "Exact flight and ground execution profile is not qualified.",
);

describe("regional-ground non-executing presentation", () => {
  it("matches the compact evidence and false-capability explanation", () => {
    const view = initialRegionalGroundExecutionPresentation(job, capability);
    expect(view.summary).toEqual({
      schema_version: job.schema_version,
      model_id: job.flight.model_id,
      model_version: job.flight.model_version,
      producer: job.provenance.producer,
      producer_version: job.provenance.producer_version,
      source_revision: job.provenance.source_revision,
      input_sha256: job.input_sha256,
    });
    expect(view.execution_enabled).toBe(false);
    expect(view.disabled_detail).toBe(capability.detail);
  });

  it("presents progress, cancel, typed failure, and exact result", () => {
    const base = initialRegionalGroundExecutionPresentation(job, capability);
    const running = presentRegionalGroundProgress(base, 2, 4);
    expect(presentRegionalGroundCancelRequested(running).status).toBe(
      "cancel_requested",
    );
    expect(presentRegionalGroundCancelled(running, 2, 4).status).toBe("cancelled");
    const failed = presentRegionalGroundFailure(running, "preflight", 2, 4);
    expect(failed.failure_stage).toBe("preflight");
    const result = parseRegionalGroundExecutionResult(resultFixture.result);
    const success = presentRegionalGroundResult(running, result, job);
    expect(success.result_schema_version).toBe(result.schema_version);
    expect(success.result_sha256).toBe(result.dataset_sha256);
  });

  it("renders disabled controls and evidence without event handlers", () => {
    const view = initialRegionalGroundExecutionPresentation(job, capability);
    render(<RegionalGroundExecutionPresentation presentation={view} />);
    expect(screen.getByRole("button", { name: "Run study" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Cancel study" })).toBeDisabled();
    expect(screen.getByText(capability.detail)).toBeVisible();
    expect(screen.getByText(job.flight.model_id)).toBeVisible();
  });

  it("rejects stale and mismatched controller counts", () => {
    const running = presentRegionalGroundProgress(
      initialRegionalGroundExecutionPresentation(job, capability),
      2,
      4,
    );
    expect(() => presentRegionalGroundProgress(running, 1, 4)).toThrow(
      "execution counts must be monotonic and match the job",
    );
    expect(() => presentRegionalGroundCancelled(running, 2, 5)).toThrow(
      "execution counts must be monotonic and match the job",
    );
  });
});
