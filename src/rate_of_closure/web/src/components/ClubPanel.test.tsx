import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import bindingFixture from "../../../../../tests/rate_of_closure/fixtures/club_assembly_binding_driver_10_5.json";

import { CLUBHEAD_STL_HEADER } from "../model/clubStlExport";
import { ClubPanel } from "./ClubPanel";

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("ClubPanel STL export", () => {
  it("imports an exact assembly binding and fails closed after selection changes", async () => {
    const digestBytes = (hex: string) =>
      Uint8Array.from(hex.match(/.{2}/gu) ?? [], (value) =>
        Number.parseInt(value, 16),
      ).buffer;
    vi.stubGlobal("crypto", {
      subtle: {
        digest: vi.fn(async (_algorithm: string, input: BufferSource) => {
          const text = new TextDecoder().decode(input as ArrayBuffer);
          return digestBytes(
            text.includes("driver-qualified-2026-08")
              ? bindingFixture.assembly_identity.sha256
              : bindingFixture.selected_spec_identity.sha256,
          );
        }),
      },
    });
    render(
      <ClubPanel
        onDriveScenario={() => undefined}
        onGenerate={() => undefined}
      />,
    );
    const input = screen.getByLabelText(/import assembly binding json/i);
    const fixtureFile = () =>
      new File([JSON.stringify(bindingFixture)], "driver.club-assembly.json", {
        type: "application/json",
      });

    fireEvent.change(input, { target: { files: [fixtureFile()] } });
    expect(await screen.findByRole("status")).toHaveTextContent(
      /assembly binding loaded: driver-qualified-2026-08.*qualified_analysis/i,
    );
    expect(screen.getByText(/bound: driver-qualified-2026-08/i)).toBeVisible();

    fireEvent.change(screen.getByRole("combobox", { name: "Club" }), {
      target: { value: "7-Iron" },
    });
    expect(screen.getByRole("status")).toHaveTextContent(
      /assembly binding cleared.*specification changed/i,
    );
    fireEvent.change(input, { target: { files: [fixtureFile()] } });
    await waitFor(() =>
      expect(screen.getByRole("status")).toHaveTextContent(
        /assembly binding import failed.*selected clubspec identity/i,
      ),
    );
    expect(screen.getByText(/no binding.*complete cg/i)).toBeVisible();
  });

  it("downloads the current selected specification with explicit contract guidance", () => {
    const createObjectURL = vi.fn<(blob: Blob) => string>(
      () => "blob:clubhead",
    );
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
    expect(button).toHaveAttribute(
      "title",
      expect.stringContaining("units=mm"),
    );
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
      "3ea68a083099ce3780418e9eff0900e7178b835608261bf7d89825bddef243c8".match(
        /.{2}/gu,
      ) ?? [],
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
