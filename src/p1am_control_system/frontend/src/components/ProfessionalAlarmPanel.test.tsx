import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ProfessionalAlarmPanel } from "./ProfessionalAlarmPanel";
import * as api from "../api/endpoints";

vi.mock("../api/endpoints", () => ({
  getProfessionalAlarms: vi.fn(),
  acknowledgeProfessionalAlarm: vi.fn(),
  shelfProfessionalAlarm: vi.fn(),
  unshelveProfessionalAlarm: vi.fn(),
}));

const alarm = {
  tag: "TAG_0",
  priority: "high" as const,
  lifecycle: "unacknowledged" as const,
  condition: "high",
  acknowledged_by: null,
  shelved_by: null,
  shelf_reason: null,
  shelf_until: null,
  suppression_rule: null,
  first_out_sequence: 1,
  active_since: "2026-08-03T20:00:00Z",
  help_text: "Review signal quality and the generic process context.",
};

describe("ProfessionalAlarmPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(api.getProfessionalAlarms).mockResolvedValue([alarm]);
    vi.mocked(api.acknowledgeProfessionalAlarm).mockResolvedValue(alarm);
  });

  it("shows priority, lifecycle, first-out, and response guidance", async () => {
    render(<ProfessionalAlarmPanel />);

    expect(await screen.findByText("TAG_0")).toBeInTheDocument();
    expect(screen.getByText(/high · unacknowledged/i)).toBeInTheDocument();
    expect(screen.getByText(/first-out #1/i)).toBeInTheDocument();
    expect(screen.getByText(/Review signal quality/)).toBeInTheDocument();
  });

  it("acknowledges through the canonical API and refreshes", async () => {
    render(<ProfessionalAlarmPanel />);
    fireEvent.click(await screen.findByRole("button", { name: /acknowledge TAG_0/i }));

    await waitFor(() =>
      expect(api.acknowledgeProfessionalAlarm).toHaveBeenCalledWith("TAG_0"),
    );
    expect(api.getProfessionalAlarms).toHaveBeenCalledTimes(2);
  });
});
