import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import * as api from "../api/endpoints";
import { OperatorWorkspace } from "./OperatorWorkspace";

vi.mock("../api/endpoints", () => ({
  getOperatorOverview: vi.fn(),
  getProtectionSnapshot: vi.fn(),
  getRepresentativeAssetHealth: vi.fn(),
  getShiftEntries: vi.fn(),
  getProductStatus: vi.fn(),
}));

const overview = {
  overview_id: "SYNTHETIC.PROCESS",
  title: "Representative Process Overview",
  data_classification: "synthetic" as const,
  not_for_live_control: true as const,
  areas: [
    {
      area_id: "SYNTHETIC.FEED",
      label: "Feed Preparation",
      detail_route: "/operator/areas/SYNTHETIC.FEED",
      assets: [
        {
          asset_id: "SYNTHETIC.FEED.PUMP",
          label: "Feed Pump",
          asset_type: "pump" as const,
          primary_value: {
            value: 62,
            unit: "%",
            source_timestamp: "2026-08-03T20:00:00Z",
          },
          quality: "simulated" as const,
          mode: "automatic" as const,
          alarm_state: "normal" as const,
          interlock_state: "clear" as const,
          detail_route: "/operator/assets/SYNTHETIC.FEED.PUMP",
          trend_tags: ["SYNTHETIC.FEED.PUMP.PV"],
        },
      ],
    },
    {
      area_id: "SYNTHETIC.REACTOR",
      label: "Reaction",
      detail_route: "/operator/areas/SYNTHETIC.REACTOR",
      assets: [],
    },
    {
      area_id: "SYNTHETIC.SEPARATION",
      label: "Separation",
      detail_route: "/operator/areas/SYNTHETIC.SEPARATION",
      assets: [],
    },
  ],
};

const protections = {
  definitions: [
    {
      protection_id: "SYNTHETIC.REACTOR.HIGH_PRESSURE",
      category: "interlock" as const,
      consequences: ["SYNTHETIC.FEED stops"],
      bypassable: true,
    },
    {
      protection_id: "SYNTHETIC.REACTOR.INDEPENDENT_TRIP",
      category: "independent_protection" as const,
      consequences: ["Synthetic heater power removed"],
      bypassable: false,
    },
  ],
  trips: [
    {
      protection_id: "SYNTHETIC.REACTOR.HIGH_PRESSURE",
      group_id: "trip-1",
      category: "interlock" as const,
      consequences: ["SYNTHETIC.FEED stops"],
      occurred_at: "2026-08-03T20:00:00Z",
      first_out: true,
    },
  ],
  active_bypasses: [
    {
      protection_id: "SYNTHETIC.REACTOR.HIGH_PRESSURE",
      actor: "engineer",
      reason: "Synthetic FAT verification",
      requested_at: "2026-08-03T20:00:00Z",
      expires_at: "2026-08-03T21:00:00Z",
      banner_required: true as const,
      active: true as const,
    },
  ],
};

const assetHealth = {
  asset_id: "SYNTHETIC.FEED.PUMP",
  generated_at: "2026-08-03T20:00:00Z",
  counters: { runtime_seconds: 600, start_count: 1 },
  statistics: {
    sample_count: 2,
    minimum: 15,
    maximum: 15,
    mean: 15,
    standard_deviation: 0,
  },
  advisories: [
    {
      code: "calibration_due" as const,
      asset_id: "SYNTHETIC.FEED.PUMP",
      detected_at: "2026-08-03T20:00:00Z",
      detail: "Calibration due date has passed",
      classification: "maintenance_advisory" as const,
      authoritative_trip: false as const,
    },
  ],
  data_classification: "synthetic" as const,
};

const productStatus = {
  procedure_state: "idle" as const,
  procedure_events: [],
  connectors: [
    {
      connector_id: "SYNTHETIC.CONNECTOR.DEMO",
      version: "1.0",
      details: { state: "online" },
    },
  ],
  samples: {
    "SYNTHETIC.DEMO.PV": {
      value: 1,
      quality: "good" as const,
      diagnostic: "",
      connector_id: "SYNTHETIC.CONNECTOR.DEMO",
    },
  },
  notification_policy: {
    primary_recipient: "synthetic.primary",
    escalation_recipient: "synthetic.escalation",
  },
  notification_audit: [],
  availability: {
    recovery_time_objective_seconds: 300,
    recovery_point_objective_seconds: 30,
    clock_ordering_reliable: true,
    command_authority: null,
    transport_available: true,
    hmi_available: true,
    buffered_samples: 0,
    data_classification: "synthetic" as const,
  },
  data_classification: "synthetic" as const,
  not_for_live_control: true as const,
};

describe("OperatorWorkspace", () => {
  beforeEach(() => {
    vi.mocked(api.getOperatorOverview).mockResolvedValue(overview);
    vi.mocked(api.getProtectionSnapshot).mockResolvedValue(protections);
    vi.mocked(api.getRepresentativeAssetHealth).mockResolvedValue(assetHealth);
    vi.mocked(api.getShiftEntries).mockResolvedValue([]);
    vi.mocked(api.getProductStatus).mockResolvedValue(productStatus);
  });

  it("navigates from a multi-area overview to a consistent faceplate", async () => {
    render(<OperatorWorkspace />);

    expect(await screen.findByText("Feed Preparation")).toBeInTheDocument();
    expect(screen.getByText("Reaction")).toBeInTheDocument();
    expect(screen.getByText("Separation")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /Feed Pump/ }));

    expect(screen.getByRole("dialog", { name: "Feed Pump faceplate" })).toHaveTextContent(
      "Quality simulated",
    );
    expect(screen.getByRole("dialog")).toHaveTextContent("Mode automatic");
    expect(screen.getByRole("dialog")).toHaveTextContent("Alarm normal");
    expect(screen.getByRole("dialog")).toHaveTextContent("Interlock clear");
    expect(screen.getByRole("button", { name: "Open trend drill-down" })).toBeInTheDocument();
  });

  it("keeps protection categories and active bypass status unmistakable", async () => {
    render(<OperatorWorkspace />);

    expect(await screen.findByRole("alert")).toHaveTextContent("Synthetic FAT verification");
    expect(screen.getByText("FIRST OUT")).toBeInTheDocument();
    expect(screen.getByText("interlock", { selector: "strong" })).toBeInTheDocument();
    expect(screen.getByText("independent protection", { selector: "strong" })).toBeInTheDocument();
    expect(screen.getByText("Non-bypassable")).toBeInTheDocument();
    expect(screen.getByText(/calibration due date has passed/i)).toBeInTheDocument();
    expect(screen.getByText(/Saved synthetic investigations retain/i)).toBeInTheDocument();
    expect(screen.getByText(/Signed entries are append-only/i)).toBeInTheDocument();
    expect(screen.getByText(/Procedure state:/i)).toHaveTextContent("idle");
    expect(screen.getByText(/RTO 300s \/ RPO 30s/i)).toBeInTheDocument();
  });
});
