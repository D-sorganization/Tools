import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { BallSetupDiagram } from "./BallSetupDiagram";

afterEach(cleanup);

describe("BallSetupDiagram", () => {
  it("renders a representative tee only in Tee mode", () => {
    const view = render(
      <BallSetupDiagram setup={{ supportMode: "tee", teeHeightM: 0.0381 }} />,
    );
    expect(screen.getByLabelText("Tee ball support side elevation")).toBeInTheDocument();
    expect(screen.getByLabelText("Representative Tee")).toBeInTheDocument();
    view.rerender(
      <BallSetupDiagram setup={{ supportMode: "ground", teeHeightM: 0 }} />,
    );
    expect(screen.getByLabelText("Ground ball support side elevation")).toBeInTheDocument();
    expect(screen.queryByLabelText("Representative Tee")).not.toBeInTheDocument();
  });
});
