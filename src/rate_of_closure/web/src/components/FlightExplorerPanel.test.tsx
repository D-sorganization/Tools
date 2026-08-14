import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { useState } from "react";
import { beforeAll, describe, expect, it, vi } from "vitest";

import { FlightExplorerPanel } from "./FlightExplorerPanel";
import { DEFAULT_TARGET, spatialTargetFromRegion } from "../model/targets";
import { directLaunch, exploreFlight } from "../model/flightExplorer";

function FlightExplorerHarness() {
  const [target, setTarget] = useState(() => spatialTargetFromRegion(DEFAULT_TARGET));
  return <FlightExplorerPanel spatialTarget={target} onSpatialTargetChange={setTarget} />;
}

beforeAll(() => {
  const ctx: unknown = new Proxy(function () {} as object, {
    get: (_target, prop) =>
      prop === "measureText" ? () => ({ width: 0 }) : () => ctx,
    set: () => true,
    apply: () => ctx,
  });
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(
    ctx as CanvasRenderingContext2D,
  );
  HTMLElement.prototype.scrollIntoView = vi.fn();
});

describe("FlightExplorerPanel input editing", () => {
  it("applies one canonical spatial target to its summary and flight plots", () => {
    render(<FlightExplorerHarness />);

    const downrange = screen.getByLabelText("Target downrange m");
    fireEvent.change(downrange, { target: { value: "180" } });
    fireEvent.click(screen.getByRole("button", { name: "Apply spatial target" }));

    expect(screen.getByRole("status", { name: "Current spatial target" }))
      .toHaveTextContent("180.0 m downrange");
    expect(screen.getByLabelText("Flight top-down view (lateral vs carry)"))
      .toHaveAttribute("aria-description", expect.stringContaining("180.0 m downrange"));
    expect(screen.getByLabelText("Flight side profile (height vs carry)"))
      .toHaveAttribute("aria-description", expect.stringContaining("180.0 m downrange"));
    expect(screen.getByRole("status", { name: "Spatial target assessment" }))
      .toHaveTextContent("Run Flight to evaluate this target");

    fireEvent.click(screen.getByRole("button", { name: "Run Flight" }));
    expect(screen.getByRole("status", { name: "Spatial target assessment" }))
      .toHaveTextContent(/Target (accepted|missed)/);
  });

  it("uses launch-monitor terminology and exposes a working definition", () => {
    render(<FlightExplorerHarness />);

    expect(screen.queryByText("Launch Azimuth")).not.toBeInTheDocument();
    expect(screen.getByLabelText("Launch Direction")).toBeInTheDocument();
    expect(screen.getByLabelText("Launch Direction Convention")).toHaveValue(
      "app_native",
    );
    const foresight = screen.getByRole("option", {
      name: "Foresight-Comparable (Sign Unavailable)",
    });
    expect(foresight).toBeDisabled();
    expect(foresight).toHaveAttribute(
      "title",
      expect.stringMatching(/public sign convention/i),
    );
    expect(screen.getByTestId("direction-sign-example")).toHaveTextContent(
      "0° = straight · + = right of the target line",
    );
    fireEvent.click(screen.getByLabelText("Explain Launch Direction"));
    expect(screen.getByText(/positive values start right of the target line/i)).toBeVisible();
  });

  it("accepts and preserves a negative spin-axis tilt", () => {
    render(<FlightExplorerHarness />);
    const input = screen.getByLabelText("Spin-Axis Tilt") as HTMLInputElement;

    fireEvent.focus(input);
    fireEvent.change(input, { target: { value: "-" } });
    expect(input.value).toBe("-");
    fireEvent.change(input, { target: { value: "-12.5" } });
    fireEvent.blur(input);

    expect(input.value).toBe("-12.5");
    fireEvent.click(screen.getByRole("button", { name: "Run Flight" }));
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("runs a visible common-input no-wind versus selected-wind comparison", () => {
    render(<FlightExplorerHarness />);

    expect(screen.getByLabelText(/Wind 10.0 miles per hour from 0.0 degrees/))
      .toBeInTheDocument();
    fireEvent.click(screen.getByRole("checkbox", {
      name: "Compare No Wind and Selected Wind",
    }));
    fireEvent.click(screen.getByRole("button", { name: "Run Flight" }));

    expect(screen.getByLabelText("Wind effect deltas")).toHaveTextContent(
      "Selected Wind Minus No Wind",
    );
    expect(screen.getByLabelText("Flight side profile (height vs carry)"))
      .toBeInTheDocument();
  });

  it("retains the atomic prior flight, context, and selection after a failed rerun", () => {
    const accepted = exploreFlight(directLaunch({
      ballSpeedMph: 167, launchAngleDeg: 10.9, launchDirectionDeg: 0,
      spinRpm: 2686, spinAxisTiltDeg: 0,
    }));
    const execute = vi.fn()
      .mockReturnValueOnce(accepted)
      .mockImplementationOnce(() => { throw new Error("planted executor failure"); });
    render(<FlightExplorerHarnessWithExecutor execute={execute} />);
    fireEvent.click(screen.getByRole("button", { name: "Run Flight" }));
    const side = screen.getByLabelText("Flight side profile (height vs carry)");
    side.focus();
    fireEvent.keyDown(side, { key: "Home" });
    expect(screen.getByRole("status", { name: "Selected flight sample" }))
      .toHaveTextContent("source sample 1/");
    expect(screen.getByLabelText("Ball flight playback position")).toHaveTextContent(/^0\.00/);
    fireEvent.keyDown(side, { key: "End" });
    expect(screen.getByLabelText("Ball flight playback position")).not.toHaveTextContent(/^0\.00/);
    fireEvent.keyDown(side, { key: "Home" });
    expect(screen.getByLabelText("Ball flight playback position")).toHaveTextContent(/^0\.00/);
    const displayed = screen.getByRole("status", { name: "Displayed flight context" }).textContent;

    fireEvent.click(screen.getByRole("button", { name: "Run Flight" }));
    expect(screen.getByRole("alert")).toHaveTextContent(/prior accepted flight remains displayed/i);
    expect(screen.getByRole("status", { name: "Displayed flight context" }).textContent).toBe(displayed);
    expect(screen.getByRole("status", { name: "Selected flight sample" }))
      .toHaveTextContent("source sample 1/");
    expect(execute).toHaveBeenCalledTimes(2);
  });

  it("uses the injected execution authority for both wind cohorts and retains prior evidence", () => {
    let calls = 0;
    const execute = vi.fn((launch: Parameters<typeof exploreFlight>[0]) => {
      calls += 1;
      if (calls > 2) throw new Error("planted paired executor failure");
      return exploreFlight(launch);
    });
    render(<FlightExplorerHarnessWithExecutor execute={execute} />);
    fireEvent.click(screen.getByRole("checkbox", {
      name: "Compare No Wind and Selected Wind",
    }));
    fireEvent.click(screen.getByRole("button", { name: "Run Flight" }));
    const displayed = screen.getByRole("status", {
      name: "Displayed flight context",
    }).textContent;
    expect(execute).toHaveBeenCalledTimes(2);

    fireEvent.click(screen.getByRole("button", { name: "Run Flight" }));
    expect(screen.getByRole("alert")).toHaveTextContent(
      /prior accepted flight remains displayed/i,
    );
    expect(screen.getByRole("status", { name: "Displayed flight context" }).textContent)
      .toBe(displayed);
    expect(execute).toHaveBeenCalledTimes(3);
  });

  it("shows an honest empty error when the paired wind executor fails initially", () => {
    const execute = vi.fn(() => {
      throw new Error(`\u0000${"initial paired failure ".repeat(60)}`);
    });
    render(<FlightExplorerHarnessWithExecutor execute={execute} />);
    fireEvent.click(screen.getByRole("checkbox", {
      name: "Compare No Wind and Selected Wind",
    }));
    fireEvent.click(screen.getByRole("button", { name: "Run Flight" }));
    const alert = screen.getByRole("alert");
    expect(alert).toHaveTextContent(/no accepted flight is available/i);
    expect(alert.textContent).not.toContain("\u0000");
    expect(alert.textContent?.length).toBeLessThanOrEqual(512);
    expect(screen.queryByRole("status", { name: "Displayed flight context" }))
      .not.toBeInTheDocument();
  });

  it("labels retained evidence as prior when current scientific inputs change", () => {
    render(<FlightExplorerHarness />);
    fireEvent.click(screen.getByRole("button", { name: "Run Flight" }));
    expect(screen.getByRole("status", { name: "Displayed flight context" }))
      .toHaveTextContent(/^Displayed flight:/);
    fireEvent.change(screen.getByLabelText("Launch Angle"), { target: { value: "12" } });
    fireEvent.blur(screen.getByLabelText("Launch Angle"));
    expect(screen.getByRole("status", { name: "Displayed flight context" }))
      .toHaveTextContent(/^Prior result — inputs changed:/);
  });

  it("keeps speed canonical across presentation-unit switches and reruns", () => {
    const execute = vi.fn((launch: Parameters<typeof exploreFlight>[0]) =>
      exploreFlight(launch));
    render(<FlightExplorerHarnessWithExecutor execute={execute} />);
    fireEvent.click(screen.getByRole("button", { name: "Run Flight" }));
    const status = screen.getByRole("status", {
      name: "Displayed flight context",
    });
    const displayed = status.textContent;
    fireEvent.change(screen.getByLabelText("Ball speed unit"), {
      target: { value: "m/s" },
    });
    expect(status.textContent).toBe(displayed);
    expect(status).toHaveTextContent(/^Displayed flight:/);
    expect(execute).toHaveBeenCalledTimes(1);
    const firstSpeed = execute.mock.calls[0][0].ballSpeedMps;
    fireEvent.click(screen.getByRole("button", { name: "Run Flight" }));
    expect(execute.mock.calls[1][0].ballSpeedMps).toBe(firstSpeed);
  });

  it("reveals the accepted visual once after a pointer run without moving focus", async () => {
    render(<FlightExplorerHarness />);
    const run = screen.getByRole("button", { name: "Run Flight" });
    run.focus();
    fireEvent.click(run, { detail: 1 });
    await waitFor(() => expect(HTMLElement.prototype.scrollIntoView).toHaveBeenCalledOnce());
    expect(vi.mocked(HTMLElement.prototype.scrollIntoView).mock.instances[0])
      .toHaveAccessibleName("Flight side profile (height vs carry)");
    expect(document.activeElement).toBe(run);
  });
});

function FlightExplorerHarnessWithExecutor({ execute }: {
  execute: typeof exploreFlight;
}) {
  const [target, setTarget] = useState(() => spatialTargetFromRegion(DEFAULT_TARGET));
  return <FlightExplorerPanel spatialTarget={target} onSpatialTargetChange={setTarget}
    executeFlight={execute} />;
}
