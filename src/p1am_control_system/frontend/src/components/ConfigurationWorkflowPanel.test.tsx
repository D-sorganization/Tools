import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ConfigurationWorkflowPanel } from "./ConfigurationWorkflowPanel";
import * as api from "../api/endpoints";

vi.mock("../api/endpoints", () => ({
  getConfigurationRevisions: vi.fn(),
  getConfigurationDiff: vi.fn(),
  validateConfiguration: vi.fn(),
  reviewConfiguration: vi.fn(),
  approveConfiguration: vi.fn(),
  activateConfiguration: vi.fn(),
  rollbackConfiguration: vi.fn(),
}));

const revision = (state: string) => ({
  revision_id: "cfg-000001-aaaaaaaaaaaa",
  version: 1,
  state,
  payload: {},
  payload_sha256: "a".repeat(64),
  reason: "Synthetic change",
  created_by: "engineer",
  created_at: "2026-08-03T00:00:00Z",
  validated_by: null,
  reviewed_by: null,
  approved_by: null,
  activated_by: null,
  activated_at: null,
  activation_identity: null,
  source_revision_id: null,
});

describe("ConfigurationWorkflowPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(api.getConfigurationDiff).mockResolvedValue([]);
  });

  it("shows immutable revision identity and advances a draft to validation", async () => {
    vi.mocked(api.getConfigurationRevisions).mockResolvedValue([revision("draft") as never]);
    vi.mocked(api.validateConfiguration).mockResolvedValue(revision("validated") as never);
    render(<ConfigurationWorkflowPanel />);

    expect(await screen.findByText(/cfg-000001-aaaaaaaaaaaa/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Validate" }));

    await waitFor(() =>
      expect(api.validateConfiguration).toHaveBeenCalledWith("cfg-000001-aaaaaaaaaaaa"),
    );
  });

  it("requires an explicit review reason before approval", async () => {
    vi.mocked(api.getConfigurationRevisions).mockResolvedValue([
      revision("in_review") as never,
    ]);
    vi.mocked(api.approveConfiguration).mockResolvedValue(revision("approved") as never);
    render(<ConfigurationWorkflowPanel />);

    fireEvent.change(await screen.findByLabelText(/Review or rollback reason/), {
      target: { value: "Synthetic approval evidence" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Approve" }));

    await waitFor(() =>
      expect(api.approveConfiguration).toHaveBeenCalledWith(
        "cfg-000001-aaaaaaaaaaaa",
        "Synthetic approval evidence",
      ),
    );
  });
});
