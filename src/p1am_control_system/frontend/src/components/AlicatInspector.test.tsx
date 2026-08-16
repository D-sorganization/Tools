import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { AlicatInspector } from "./AlicatInspector";
import type { AlicatMFCState } from "../api/schemas";

/**
 * Alicat setpoint entry (#4013).
 *
 * The draft used to be re-seeded by an App effect keyed on `[inspectorView,
 * alicats]`. `alicats` gets a new reference on essentially every frame because
 * the equality check compares live analog fields (mass_flow, pressure,
 * temperature) that change every scan on real hardware — so the field fought
 * every keystroke and pressing Set could commit a partially-reverted value
 * (2 instead of 25), which `AlicatInspector` accepts because it is in range.
 *
 * The entry is now purely operator-owned, and the DEVICE's live setpoint is
 * shown as its own read-only readout so both numbers are visible at once
 * instead of fighting over one widget.
 */

const mfc: AlicatMFCState = {
  device_id: "MFC1",
  name: "Fuel MFC",
  gas: "N2",
  setpoint: 25,
  mass_flow: 24.87,
  volumetric_flow: 24.5,
  pressure: 14.7,
  temperature: 21.3,
  max_flow: 50,
  port_or_ip: "/dev/ttyUSB0",
  connection_state: "connected",
};

describe("AlicatInspector", () => {
  it("renders the operator's draft text verbatim, not the device value", () => {
    render(
      <AlicatInspector
        deviceId="MFC1"
        alicats={[mfc]}
        setpointValue="2"
        onSetpointValueChange={vi.fn()}
        onSetpoint={vi.fn()}
        onGasChange={vi.fn()}
        triggerNotification={vi.fn()}
      />,
    );
    expect(screen.getByLabelText(/Flow Setpoint Command/i)).toHaveValue(2);
  });

  it("shows the LIVE device setpoint as a separate read-only readout", () => {
    // Without this, the operator has no way to see what the device is actually
    // commanded to while they are mid-entry — the old design had exactly one
    // widget for two different numbers.
    render(
      <AlicatInspector
        deviceId="MFC1"
        alicats={[mfc]}
        setpointValue="2"
        onSetpointValueChange={vi.fn()}
        onSetpoint={vi.fn()}
        onGasChange={vi.fn()}
        triggerNotification={vi.fn()}
      />,
    );
    const readout = screen.getByTestId("alicat-device-setpoint");
    expect(readout).toHaveTextContent("25.00");
    // It must be a readout, not an input the draft can be confused with.
    expect(readout.querySelector("input")).toBeNull();
  });

  it("flags when the draft differs from the device so Set is a conscious act", () => {
    render(
      <AlicatInspector
        deviceId="MFC1"
        alicats={[mfc]}
        setpointValue="2"
        onSetpointValueChange={vi.fn()}
        onSetpoint={vi.fn()}
        onGasChange={vi.fn()}
        triggerNotification={vi.fn()}
      />,
    );
    expect(screen.getByTestId("alicat-setpoint-pending")).toBeInTheDocument();
  });

  it("does not flag a pending change when the draft matches the device", () => {
    render(
      <AlicatInspector
        deviceId="MFC1"
        alicats={[mfc]}
        setpointValue="25"
        onSetpointValueChange={vi.fn()}
        onSetpoint={vi.fn()}
        onGasChange={vi.fn()}
        triggerNotification={vi.fn()}
      />,
    );
    expect(screen.queryByTestId("alicat-setpoint-pending")).not.toBeInTheDocument();
  });
});
