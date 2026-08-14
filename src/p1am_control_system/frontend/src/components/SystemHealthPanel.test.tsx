import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { SystemHealthPanel } from "./SystemHealthPanel";
import * as api from "../api/endpoints";

vi.mock("../api/endpoints", () => ({
  getSystemHealth: vi.fn(),
  downloadRecoveryPackage: vi.fn(),
  restoreRecoveryPackage: vi.fn(),
  runRepresentativeScenario: vi.fn(),
}));

const health = {
  generated_at: "2026-08-03T00:00:00Z",
  overall: "degraded" as const,
  identity: {
    software_revision: "software-test-1",
    configuration_revision: "cfg-000001-proof",
    configuration_sha256: "a".repeat(64),
    configuration_state: "active",
  },
  checks: [{ name: "primary_transport", status: "degraded" as const, detail: "Disconnected" }],
};

describe("SystemHealthPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(api.getSystemHealth).mockResolvedValue(health);
  });

  it("shows deployment identity without conflating degraded transport", async () => {
    render(<SystemHealthPanel />);

    expect(await screen.findByText(/software-test-1/)).toBeInTheDocument();
    expect(screen.getByText(/primary_transport: degraded/)).toBeInTheDocument();
    expect(screen.getByText(/restore into a draft only/i)).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Run Synthetic Acceptance Scenario" }),
    ).toBeInTheDocument();
  });

  it("refuses restore without a package and checksum", async () => {
    render(<SystemHealthPanel />);
    fireEvent.click(screen.getByRole("button", { name: "Verify & Restore as Draft" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("Select a package");
    await waitFor(() => expect(api.restoreRecoveryPackage).not.toHaveBeenCalled());
  });
});
