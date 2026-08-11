import { fireEvent, render, screen, waitFor } from "@testing-library/react";
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

  it("downloads a capability-honest engineering sidecar", async () => {
    const createObjectURL = vi.fn<(blob: Blob) => string>(() => "blob:sidecar");
    const revokeObjectURL = vi.fn();
    const digestBytes = Uint8Array.from(
      "3ea68a083099ce3780418e9eff0900e7178b835608261bf7d89825bddef243c8"
        .match(/.{2}/gu) ?? [],
      (value) => Number.parseInt(value, 16),
    );
    vi.stubGlobal("URL", { createObjectURL, revokeObjectURL });
    vi.stubGlobal("crypto", {
      subtle: { digest: vi.fn(async () => digestBytes.buffer) },
    });
    const click = vi
      .spyOn(HTMLAnchorElement.prototype, "click")
      .mockImplementation(() => undefined);
    render(
      <ClubPanel
        onDriveScenario={() => undefined}
        onGenerate={() => undefined}
      />,
    );

    fireEvent.click(
      screen.getByRole("button", {
        name: /download engineering sidecar json/i,
      }),
    );

    await waitFor(() => expect(createObjectURL).toHaveBeenCalledOnce());
    expect(click).toHaveBeenCalledOnce();
    expect(revokeObjectURL).toHaveBeenCalledWith("blob:sidecar");
    expect(screen.getByRole("status")).toHaveTextContent(
      /engineering json downloaded.*driver-10-5\.engineering\.json/i,
    );
    const blob = createObjectURL.mock.calls[0][0];
    expect(blob.type).toBe("application/json");
  });

  it("reports an unavailable digest capability without starting a download", async () => {
    const createObjectURL = vi.fn();
    vi.stubGlobal("URL", {
      createObjectURL,
      revokeObjectURL: vi.fn(),
    });
    vi.stubGlobal("crypto", {});
    render(
      <ClubPanel
        onDriveScenario={() => undefined}
        onGenerate={() => undefined}
      />,
    );

    fireEvent.click(
      screen.getByRole("button", {
        name: /download engineering sidecar json/i,
      }),
    );

    expect(await screen.findByRole("status")).toHaveTextContent(
      /engineering json download failed: browser sha-256 capability is unavailable/i,
    );
    expect(createObjectURL).not.toHaveBeenCalled();
  });
});
