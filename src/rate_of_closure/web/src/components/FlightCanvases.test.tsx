import { render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { FlightCanvases } from "./FlightCanvases";

describe("FlightCanvases responsive layout", () => {
  beforeEach(() => {
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null);
  });

  afterEach(() => vi.restoreAllMocks());

  it.each([
    ["Flight side profile (height vs carry)", "860", "260", "860 / 260"],
    ["Flight top-down view (lateral vs carry)", "860", "220", "860 / 220"],
  ])(
    "preserves the intrinsic aspect ratio of %s at responsive widths",
    (label, width, height, aspectRatio) => {
      render(<FlightCanvases points={[]} />);

      const canvas = screen.getByLabelText(label);
      expect(canvas).toHaveAttribute("width", width);
      expect(canvas).toHaveAttribute("height", height);
      expect(canvas).toHaveStyle({
        width: "100%",
        height: "auto",
        aspectRatio,
      });
    },
  );
});
