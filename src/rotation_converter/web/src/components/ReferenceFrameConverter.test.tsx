import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { ReferenceFrameConverter } from "./ReferenceFrameConverter";

type MockResponsePayload = {
  operation: string;
  results: Record<string, unknown>;
  explanation_markdown: string;
  explanation_latex: string;
};

const DEFAULT_RESPONSE: MockResponsePayload = {
  operation: "twist_frame_conversion",
  results: {},
  explanation_markdown: "ok",
  explanation_latex: "ok",
};

const mockedFetch = vi.fn<
  (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>
>();

function mockSuccessfulResponse(payload: MockResponsePayload = DEFAULT_RESPONSE): void {
  mockedFetch.mockResolvedValue({
    ok: true,
    json: async () => payload,
  } as Response);
}

function assertPayloadContainsOnlyExpectedFields(
  expectedOperation: string,
  expectedFields: string[],
): void {
  expect(mockedFetch).toHaveBeenCalledTimes(1);
  const init = mockedFetch.mock.calls[0][1] as RequestInit;
  const parsed = JSON.parse(String(init.body)) as Record<string, unknown>;
  expect(parsed.operation).toBe(expectedOperation);
  const keys = Object.keys(parsed).sort();
  expect(keys).toEqual(["operation", ...expectedFields].sort());
}

describe("ReferenceFrameConverter", () => {
  afterEach(() => {
    mockedFetch.mockReset();
    vi.unstubAllGlobals();
  });

  it("sends twist payload for twist_frame_conversion", async () => {
    mockSuccessfulResponse();
    vi.stubGlobal("fetch", mockedFetch);
    render(<ReferenceFrameConverter />);

    fireEvent.click(screen.getByRole("button", { name: "Compute" }));

    await waitFor(() => {
      assertPayloadContainsOnlyExpectedFields("twist_frame_conversion", [
        "transform",
        "twist",
      ]);
    });
  });

  it("sends rotation + translation payload for homogeneous_transform", async () => {
    mockSuccessfulResponse({
      ...DEFAULT_RESPONSE,
      operation: "homogeneous_transform",
    });
    vi.stubGlobal("fetch", mockedFetch);
    render(<ReferenceFrameConverter />);

    fireEvent.change(screen.getByRole("combobox"), {
      target: { value: "homogeneous_transform" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Compute" }));

    await waitFor(() => {
      assertPayloadContainsOnlyExpectedFields("homogeneous_transform", [
        "rotation_matrix",
        "translation",
      ]);
    });
  });

  it("sends so3_vector payload for so3_so3_maps", async () => {
    mockSuccessfulResponse({
      ...DEFAULT_RESPONSE,
      operation: "so3_so3_maps",
    });
    vi.stubGlobal("fetch", mockedFetch);
    render(<ReferenceFrameConverter />);

    fireEvent.change(screen.getByRole("combobox"), {
      target: { value: "so3_so3_maps" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Compute" }));

    await waitFor(() => {
      assertPayloadContainsOnlyExpectedFields("so3_so3_maps", ["so3_vector"]);
    });
  });

  it("gives every dense numeric input a descriptive accessible name", () => {
    render(<ReferenceFrameConverter />);

    expect(screen.getByRole("combobox", { name: "Operation" })).toBeInTheDocument();
    expect(
      screen.getByRole("spinbutton", {
        name: "Transform matrix row 1, column 1",
      }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("spinbutton", { name: "Twist angular velocity x" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("spinbutton", { name: "Twist linear velocity z" }),
    ).toBeInTheDocument();

    fireEvent.change(screen.getByRole("combobox", { name: "Operation" }), {
      target: { value: "homogeneous_transform" },
    });
    expect(
      screen.getByRole("spinbutton", {
        name: "Rotation matrix row 3, column 3",
      }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("spinbutton", { name: "Translation component y" }),
    ).toBeInTheDocument();

    fireEvent.change(screen.getByRole("combobox", { name: "Operation" }), {
      target: { value: "so3_so3_maps" },
    });
    expect(
      screen.getByRole("spinbutton", { name: "Rotation vector component z" }),
    ).toBeInTheDocument();
  });
});
