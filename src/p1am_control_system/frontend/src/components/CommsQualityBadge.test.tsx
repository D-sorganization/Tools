import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { CommsQualityBadge } from "./CommsQualityBadge";

describe("CommsQualityBadge", () => {
  it("does not call a live socket healthy when process data is stale", () => {
    render(
      <CommsQualityBadge
        transportConnected
        health={{
          quality: "stale",
          diagnostic_reason: "read_timeout",
          sequence: 42,
          server_timestamp: "2026-08-03T20:00:00+00:00",
          source: "synthetic.driver",
        }}
      />,
    );

    expect(screen.getByRole("status")).toHaveTextContent("DATA STALE");
    expect(screen.getByRole("status")).toHaveAccessibleDescription(
      /read_timeout.*sequence 42/i,
    );
  });

  it("labels simulated data explicitly", () => {
    render(
      <CommsQualityBadge
        transportConnected
        health={{
          quality: "simulated",
          diagnostic_reason: "synthetic_source",
          sequence: 1,
          server_timestamp: "2026-08-03T20:00:00+00:00",
          source: "synthetic.simulator",
        }}
      />,
    );

    expect(screen.getByRole("status")).toHaveTextContent("SIMULATED DATA");
  });

  it("shows transport offline before any quality claim", () => {
    render(<CommsQualityBadge transportConnected={false} health={undefined} />);

    expect(screen.getByRole("status")).toHaveTextContent("OFFLINE");
  });
});
