// @vitest-environment jsdom

import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { PuttingPanel } from "./PuttingPanel";
import { SCRUB_STEPS } from "../model/playbackTransport";
import { evaluatePuttWithTrajectory } from "../model/puttingScenario";
import { greenSurfaceToJson } from "../model/puttingGreenWire";
import { planarSurface } from "../model/puttingGreen";
import udGreenFixture from "../model/__fixtures__/ud_green_topography.json";

type Authority = typeof evaluatePuttWithTrajectory;

describe("PuttingPanel model bounds", () => {
  it.each([
    ["Clubhead speed m/s", "0.2", "6"],
    ["Distance to hole m", "0.1", "40"],
    ["Aim °", "-45", "45"],
    ["Face angle °", "-20", "20"],
    ["Putter path °", "-20", "20"],
    ["Attack angle °", "-10", "10"],
    ["Shaft lean °", "-10", "10"],
    ["Strike toward toe mm", "-40", "40"],
    ["Strike up the face mm", "-20", "20"],
    ["Green speed (stimp) ft", "3", "16"],
    ["Slope grade %", "0", "10"],
    ["Downhill direction °", "-360", "360"],
  ])("exposes Python-parity bounds for %s", (name, min, max) => {
    render(<PuttingPanel />);

    expect(screen.getByRole("textbox", { name })).toHaveAttribute("min", min);
    expect(screen.getByRole("textbox", { name })).toHaveAttribute("max", max);
  });

  it("selects a putter and a hole-capture model", () => {
    render(<PuttingPanel />);
    const putter = screen.getByRole("combobox", { name: "Putter" });
    expect((putter as HTMLSelectElement).options.length).toBeGreaterThan(1);
    const capture = screen.getByRole("combobox", { name: "Hole capture" });
    expect((capture as HTMLSelectElement).value).toBe("effective_radius");
    fireEvent.change(capture, { target: { value: "speed_threshold" } });
    expect((capture as HTMLSelectElement).value).toBe("speed_threshold");
  });
});

describe("PuttingPanel P1-P5 result surface", () => {
  it("reports the launched start line, the break apex and the capture margin", () => {
    render(<PuttingPanel distanceUnit="m" />);
    for (const label of [
      /Start Line/,
      /Ball Speed/,
      /Launch Sidespin/,
      /Face Twist At Impact/,
      /Apex Break/,
      /Entry Angle/,
      /Closest Approach/,
      /Capture Margin/,
    ]) {
      expect(screen.getByRole("button", { name: label })).not.toHaveTextContent(
        "—",
      );
    }
  });

  it("moves the start line right when the face is opened", () => {
    render(<PuttingPanel distanceUnit="m" />);
    const square = screen
      .getByRole("button", { name: /Start Line/ })
      .textContent;
    const face = screen.getByRole("textbox", { name: "Face angle °" });
    fireEvent.change(face, { target: { value: "2" } });
    fireEvent.blur(face);
    const opened = screen.getByRole("button", { name: /Start Line/ });
    expect(opened.textContent).not.toBe(square);
    expect(opened).toHaveTextContent("right");
  });

  it("rings the break apex only once the green actually breaks", () => {
    render(<PuttingPanel distanceUnit="m" />);
    // A flat green with a square stroke has no lateral excursion at all.
    expect(screen.queryByTestId("putting-apex-break")).toBeNull();
    const grade = screen.getByRole("textbox", { name: "Slope grade %" });
    fireEvent.change(grade, { target: { value: "2" } });
    fireEvent.blur(grade);
    expect(screen.getByTestId("putting-apex-break")).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /Apex Break/ }),
    ).not.toHaveTextContent("—");
  });

  it("opens the capture mouth only for an approach the hole can swallow", () => {
    render(<PuttingPanel distanceUnit="m" />);
    // 1.8 m/s arrives above the geometric capture bound: the published
    // effective radius is zero, so no mouth is drawn.
    expect(screen.queryByTestId("putting-capture-mouth")).toBeNull();
    const speed = screen.getByRole("textbox", { name: "Clubhead speed m/s" });
    fireEvent.change(speed, { target: { value: "1.55" } });
    fireEvent.blur(speed);
    expect(screen.getByTestId("putting-capture-mouth")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Holed/ })).toHaveTextContent(
      "HOLED",
    );
  });
});

describe("PuttingPanel accepted-study authority", () => {
  it("preserves exact selection for presentation units and clears on scientific replacement", () => {
    const { rerender } = render(<PuttingPanel distanceUnit="m" />);
    const path = screen.getByRole("img", { name: /interactive putt path/i });
    fireEvent.keyDown(path, { key: "Home" });
    expect(screen.getByRole("status", { name: "Selected putt sample" })).toHaveTextContent("Source sample 0");

    rerender(<PuttingPanel distanceUnit="yd" />);
    expect(screen.getByRole("status", { name: "Selected putt sample" })).toHaveTextContent("Source sample 0");

    fireEvent.change(screen.getByRole("textbox", { name: "Slope grade %" }), {
      target: { value: "1" },
    });
    fireEvent.blur(screen.getByRole("textbox", { name: "Slope grade %" }));
    expect(screen.getByRole("status", { name: "Selected putt sample" })).toHaveTextContent(
      "No trajectory sample selected",
    );
  });

  it("retains the exact accepted inspector when a scientific replacement fails", () => {
    const executeStudy: Authority = (scenario) => {
      if (scenario.surface.kind === "planar" && scenario.surface.gradePercent === 1) {
        throw new Error("solver authority unavailable");
      }
      return evaluatePuttWithTrajectory(scenario);
    };
    render(<PuttingPanel executeStudy={executeStudy} />);
    const path = screen.getByRole("img", { name: /interactive putt path/i });
    fireEvent.keyDown(path, { key: "Home" });
    const marker = screen.getAllByTestId("putting-selected-sample")[0];
    const acceptedGeometry = path.innerHTML;
    const acceptedContext = screen.getByLabelText(
      "Displayed putting result context",
    ).textContent;

    fireEvent.change(screen.getByRole("textbox", { name: "Slope grade %" }), {
      target: { value: "1" },
    });
    fireEvent.blur(screen.getByRole("textbox", { name: "Slope grade %" }));

    expect(screen.getByRole("alert")).toHaveTextContent(
      "solver authority unavailable",
    );
    expect(screen.getByRole("status", { name: "Selected putt sample" })).toHaveTextContent("Source sample 0");
    expect(screen.getAllByTestId("putting-selected-sample")[0]).toBe(marker);
    expect(path.innerHTML).toBe(acceptedGeometry);
    expect(
      screen.getByLabelText("Displayed putting result context"),
    ).toHaveTextContent(acceptedContext ?? "");
  });

  it("labels a first computation failure without fabricating prior evidence", () => {
    const executeStudy: Authority = () => {
      throw new Error("solver authority unavailable");
    };
    render(<PuttingPanel executeStudy={executeStudy} />);

    expect(screen.getByRole("alert")).toHaveTextContent(
      "No accepted putt is available",
    );
    expect(
      screen.queryByRole("img", { name: /interactive putt path/i }),
    ).toBeNull();
    expect(
      screen.queryByText(/prior accepted putt remains displayed/i),
    ).toBeNull();
  });

  it("snapshots mutable results and clears selection for a same-object replacement", () => {
    const cached = evaluatePuttWithTrajectory({
      scenarioId: "cached",
      putter: { name: "Blade Putter", headMassKg: 0.35, loftDeg: 3, cor: 0.78 },
      stroke: {
        clubheadSpeedMps: 1.8,
        shaftLeanDeg: 0,
        aimDeg: 0,
        faceAngleDeg: 0,
        pathAngleDeg: 0,
        attackAngleDeg: 0,
        strikeOffsetToeMm: 0,
        strikeOffsetHighMm: 0,
      },
      surface: { kind: "planar", gradePercent: 0, aspectDeg: 90 },
      stimpFt: 10,
      holeDistanceM: 3,
      provenance: {
        putterSource: "minimal",
        putterName: "Blade Putter",
        strokeSource: "declared",
        captureModel: "effective_radius",
        putterMeshSha256: null,
        putterLibraryName: null,
        strokeSourceId: null,
        kernel: "RK4-2ms-v1",
      },
    });
    const executeStudy: Authority = () => cached;
    const { rerender } = render(<PuttingPanel executeStudy={executeStudy} />);
    const path = screen.getByRole("img", { name: /interactive putt path/i });
    fireEvent.keyDown(path, { key: "Home" });
    const rollout = screen.getByRole("button", { name: /Roll-Out Distance/ });
    const displayed = rollout.textContent;
    const originalDistance = cached.result.totalDistanceM;
    cached.result.totalDistanceM = 999;
    cached.document.totalDistanceM = 999;
    rerender(<PuttingPanel executeStudy={executeStudy} distanceUnit="yd" />);
    expect(
      screen.getByRole("button", { name: /Roll-Out Distance/ }),
    ).toHaveTextContent(displayed ?? "");
    cached.result.totalDistanceM = originalDistance;
    cached.document.totalDistanceM = originalDistance;
    fireEvent.change(screen.getByRole("textbox", { name: "Slope grade %" }), {
      target: { value: "1" },
    });
    fireEvent.blur(screen.getByRole("textbox", { name: "Slope grade %" }));
    expect(screen.getByRole("status", { name: "Selected putt sample" })).toHaveTextContent(
      "No trajectory sample selected",
    );
  });
});

describe("PuttingPanel playback on the shared transport (#4800 P8)", () => {
  it("exposes the shared transport with Putt wording and Strike/Finish jumps", () => {
    render(<PuttingPanel />);

    expect(screen.getByRole("button", { name: "Play Putt" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Restart Putt" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Jump to Strike" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Jump to Finish" })).toBeEnabled();
    expect(screen.getByRole("slider", { name: "Putt Time" })).toHaveAttribute(
      "max",
      String(SCRUB_STEPS),
    );
    expect(screen.getByLabelText("Playback Speed")).toHaveValue("1");
    expect(screen.getByLabelText("Putt playback position")).toHaveTextContent(
      /^0\.00 \/ \d+\.\d\d s$/,
    );
  });

  it("replays the retained samples: scrubbing moves the reported frame", () => {
    render(<PuttingPanel />);

    const frame = screen.getByRole("status", { name: "Putt playback frame" });
    expect(frame).toHaveTextContent(/^t 0\.000 s of \d+\.\d{3} s;/);
    const start = frame.textContent;
    fireEvent.change(screen.getByRole("slider", { name: "Putt Time" }), {
      target: { value: String(SCRUB_STEPS) },
    });
    expect(frame.textContent).not.toBe(start);
    expect(frame).toHaveTextContent(/elevation -?\d+\.\d{3} m; (holed|in play)\.$/);
  });

  it("jumps to the recorded strike and finish samples", () => {
    render(<PuttingPanel />);

    fireEvent.click(screen.getByRole("button", { name: "Jump to Finish" }));
    const finish = screen.getByLabelText("Putt playback position").textContent;
    const [current, total] = (finish ?? "").replace(" s", "").split(" / ");
    expect(current).toBe(total);
    expect(Number(total)).toBeGreaterThan(0);
    fireEvent.click(screen.getByRole("button", { name: "Jump to Strike" }));
    expect(screen.getByLabelText("Putt playback position")).toHaveTextContent(
      /^0\.00 \//,
    );
  });

  it("draws the playback ball on the accepted green view", () => {
    const view = render(<PuttingPanel />);

    const ball = view.container.querySelector(
      '[data-testid="putt-playback-ball"]',
    );
    expect(ball).not.toBeNull();
    const startX = ball?.getAttribute("cx");
    fireEvent.click(screen.getByRole("button", { name: "Jump to Finish" }));
    expect(
      view.container
        .querySelector('[data-testid="putt-playback-ball"]')
        ?.getAttribute("cx"),
    ).not.toBe(startX);
  });

  it("returns to the strike when a new putt is accepted", () => {
    render(<PuttingPanel />);

    fireEvent.click(screen.getByRole("button", { name: "Jump to Finish" }));
    expect(screen.getByLabelText("Putt playback position")).not.toHaveTextContent(
      /^0\.00 \//,
    );
    fireEvent.change(screen.getByRole("textbox", { name: "Slope grade %" }), {
      target: { value: "2" },
    });
    fireEvent.blur(screen.getByRole("textbox", { name: "Slope grade %" }));
    expect(screen.getByLabelText("Putt playback position")).toHaveTextContent(
      /^0\.00 \//,
    );
  });
});

describe("PuttingPanel green import (ADR-0045 F2)", () => {
  it("imports a swing_sim.green_surface/1 document and putts on it", async () => {
    render(<PuttingPanel distanceUnit="m" />);
    const before = screen.getByRole("button", { name: /Roll-Out Distance/ })
      .textContent;
    const text = greenSurfaceToJson(planarSurface(4.0, 90.0));
    const input = screen.getByLabelText("Import Green Document File");
    fireEvent.change(input, {
      target: { files: [new File([text], "custom.json")] },
    });
    await waitFor(() =>
      expect(screen.getByLabelText("Green surface source")).toHaveTextContent(
        "swing_sim.green_surface/1",
      ),
    );
    expect(screen.getByLabelText("Green surface source")).toHaveTextContent(
      "custom.json",
    );
    // The imported heightfield replaces the planar green outright: the
    // grade/aspect spins are disabled rather than silently ignored.
    expect(screen.getByRole("textbox", { name: "Slope grade %" })).toBeDisabled();
    expect(
      screen.getByRole("textbox", { name: "Downhill direction °" }),
    ).toBeDisabled();
    expect(screen.queryByRole("alert")).toBeNull();
    expect(
      screen.getByRole("button", { name: /Roll-Out Distance/ }).textContent,
    ).not.toBe(before);
  });

  it("imports an UpstreamDrift putting_green topography through the P9 adapter and putts on it", async () => {
    render(<PuttingPanel distanceUnit="m" />);
    const text = JSON.stringify(udGreenFixture);
    const input = screen.getByLabelText("Import Green Document File");
    fireEvent.change(input, {
      target: { files: [new File([text], "ud_green.json")] },
    });
    await waitFor(() =>
      expect(screen.getByLabelText("Green surface source")).toHaveTextContent(
        "upstreamdrift",
      ),
    );
    expect(screen.getByLabelText("Green surface source")).toHaveTextContent(
      "ud_green.json",
    );
    expect(screen.queryByRole("alert")).toBeNull();
    for (const label of [/Roll-Out Distance/, /Ball Speed/, /Apex Break/]) {
      expect(
        screen.getByRole("button", { name: label }),
      ).not.toHaveTextContent("—");
    }
  });

  it("refuses a weighted-slope UD document with the adapter's named reason and keeps the previous green", async () => {
    render(<PuttingPanel distanceUnit="m" />);
    const text = JSON.stringify({
      contours: [
        { x: 0.0, y: 0.0, elevation: 0.0 },
        { x: 1.0, y: 0.0, elevation: 0.0 },
        { x: 0.0, y: 1.0, elevation: 0.0 },
        { x: 1.0, y: 1.0, elevation: 0.0 },
      ],
      slopes: [
        {
          center: [0.5, 0.5],
          radius: 1.0,
          direction: [1.0, 0.0],
          magnitude: 0.02,
        },
      ],
    });
    const input = screen.getByLabelText("Import Green Document File");
    fireEvent.change(input, {
      target: { files: [new File([text], "ud_slopes.json")] },
    });
    await waitFor(() =>
      expect(screen.getByRole("alert")).toHaveTextContent(/slope/),
    );
    expect(screen.getByRole("alert")).toHaveTextContent("ud_slopes.json");
    // Fail-closed: the previously integrated planar green is untouched.
    expect(
      screen.getByLabelText("Green surface source"),
    ).toHaveTextContent("planar grade/aspect");
    expect(screen.getByRole("textbox", { name: "Slope grade %" })).toBeEnabled();
  });

  it("refuses an unrecognized document format", async () => {
    render(<PuttingPanel distanceUnit="m" />);
    const input = screen.getByLabelText("Import Green Document File");
    fireEvent.change(input, {
      target: {
        files: [
          new File(
            ['{"format": "swing_sim.green_surface/9"}'],
            "bad.json",
          ),
        ],
      },
    });
    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent("bad.json"));
    expect(
      screen.getByLabelText("Green surface source"),
    ).toHaveTextContent("planar grade/aspect");
  });

  it("Use planar green reverts an imported heightfield", async () => {
    render(<PuttingPanel distanceUnit="m" />);
    const text = greenSurfaceToJson(planarSurface(4.0, 90.0));
    const input = screen.getByLabelText("Import Green Document File");
    fireEvent.change(input, {
      target: { files: [new File([text], "custom.json")] },
    });
    await waitFor(() =>
      expect(screen.getByRole("textbox", { name: "Slope grade %" })).toBeDisabled(),
    );
    fireEvent.click(screen.getByRole("button", { name: "Use Planar Green" }));
    expect(screen.getByLabelText("Green surface source")).toHaveTextContent(
      "planar grade/aspect",
    );
    expect(screen.getByRole("textbox", { name: "Slope grade %" })).toBeEnabled();
  });
});
