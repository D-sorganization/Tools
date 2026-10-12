import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { InterlocksPanel } from "./InterlocksPanel";
import type { InterlockConfig } from "../App";

const interlocks: InterlockConfig[] = Array.from({ length: 32 }, () => ({
  lolo_limit: null,
  low_limit: null,
  high_limit: null,
  hihi_limit: null,
}));

function renderPanel(props: {
  firstOutTagId?: number | null;
  plcTripLatch?: "tripped" | "clear" | "unknown";
  connected?: boolean;
}) {
  return render(
    <InterlocksPanel
      interlocks={interlocks}
      onChange={vi.fn()}
      onDeploy={vi.fn()}
      deploying={false}
      connected={true}
      {...props}
    />,
  );
}

describe("InterlocksPanel first-out indicator (F07)", () => {
  it("shows the first-out tag name when the PLC latch is tripped and the tag is known", () => {
    renderPanel({ plcTripLatch: "tripped", firstOutTagId: 7 });
    expect(screen.getByTestId("first-out")).toHaveTextContent("Tag 7");
  });

  it("shows tag 0 (falsy id) rather than treating it as unknown", () => {
    renderPanel({ plcTripLatch: "tripped", firstOutTagId: 0 });
    expect(screen.getByTestId("first-out")).toHaveTextContent("Tag 0");
  });

  it("PLC-only trip with unknown tag still shows the trip", () => {
    renderPanel({ plcTripLatch: "tripped", firstOutTagId: null });
    expect(screen.getByTestId("first-out")).toHaveTextContent(
      /PLC trip latched, first-out unknown/i,
    );
  });

  it("shows nothing alarming when the latch is clear", () => {
    renderPanel({ plcTripLatch: "clear", firstOutTagId: null });
    expect(screen.queryByTestId("first-out")).toBeNull();
    expect(screen.queryByTestId("trip-status-unknown")).toBeNull();
  });

  it("shows an explicit trip status unknown state when the latch is unknown and connected", () => {
    renderPanel({ plcTripLatch: "unknown", connected: true });
    expect(screen.getByTestId("trip-status-unknown")).toHaveTextContent(
      /trip status unknown/i,
    );
    expect(screen.queryByTestId("first-out")).toBeNull();
  });

  it("treats an absent latch field as unknown while connected", () => {
    renderPanel({ connected: true });
    expect(screen.getByTestId("trip-status-unknown")).toBeInTheDocument();
  });

  it("does not claim unknown trip status when not connected", () => {
    renderPanel({ plcTripLatch: "unknown", connected: false });
    expect(screen.queryByTestId("trip-status-unknown")).toBeNull();
  });
});
