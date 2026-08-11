import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { CLUBHEAD_STL_HEADER } from "../model/clubStlExport";
import { ClubPanel } from "./ClubPanel";

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("ClubPanel STL export", () => {
  it("downloads the current selected specification with explicit contract guidance", () => {
    const createObjectURL = vi.fn<(blob: Blob) => string>(() => "blob:clubhead");
    const revokeObjectURL = vi.fn();
    vi.stubGlobal("URL", { createObjectURL, revokeObjectURL });
    const click = vi
      .spyOn(HTMLAnchorElement.prototype, "click")
      .mockImplementation(() => undefined);
    render(
      <ClubPanel
        onDriveScenario={() => undefined}
        onGenerate={() => undefined}
      />,
    );

    const button = screen.getByRole("button", {
      name: /download selected clubhead stl/i,
    });
    expect(button).toHaveAttribute("title", expect.stringContaining("units=mm"));
    expect(button).toHaveAttribute(
      "title",
      expect.stringContaining("x target, y up, z toe"),
    );
    fireEvent.click(button);

    expect(createObjectURL).toHaveBeenCalledOnce();
    expect(click).toHaveBeenCalledOnce();
    expect(revokeObjectURL).toHaveBeenCalledWith("blob:clubhead");
    expect(screen.getByRole("status")).toHaveTextContent(
      /driver 10.5.*driver-10-5\.stl/i,
    );
    const blob = createObjectURL.mock.calls[0][0];
    expect(blob.type).toBe("model/stl");
    expect(CLUBHEAD_STL_HEADER).toContain("mesh=parametric");
  });

  it("reports a browser download failure without crashing the panel", () => {
    vi.stubGlobal("URL", {
      createObjectURL: () => {
        throw new Error("downloads disabled");
      },
      revokeObjectURL: vi.fn(),
    });
    render(
      <ClubPanel
        onDriveScenario={() => undefined}
        onGenerate={() => undefined}
      />,
    );

    fireEvent.click(
      screen.getByRole("button", {
        name: /download selected clubhead stl/i,
      }),
    );

    expect(screen.getByRole("status")).toHaveTextContent(
      /stl download failed: downloads disabled/i,
    );
  });
});
