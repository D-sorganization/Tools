import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";

import App from "../App";

beforeAll(() => {
  const context: unknown = new Proxy(function () {} as object, {
    get: (_target, property) =>
      property === "measureText" ? () => ({ width: 0 }) : () => context,
    set: () => true,
    apply: () => context,
  });
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(
    context as CanvasRenderingContext2D,
  );
});

afterEach(() => {
  cleanup();
  if (typeof window.localStorage.clear === "function") window.localStorage.clear();
});

describe("application view compositor routing", () => {
  it("routes direct commands to real hosts and expands to distinct grid views", async () => {
    render(<App />);
    fireEvent.click(screen.getByRole("button", { name: /^Impact$/ }));

    expect(await screen.findByRole("region", {
      name: "Impact synchronized viewport",
    })).toBeInTheDocument();
    fireEvent.change(screen.getByRole("combobox", { name: "Viewport layout" }), {
      target: { value: "grid" },
    });

    await waitFor(() => {
      expect(screen.getByRole("region", { name: "Swing synchronized viewport" }))
        .toBeInTheDocument();
      expect(screen.getByRole("region", { name: "Flight synchronized viewport" }))
        .toBeInTheDocument();
    });
    expect(screen.getAllByRole("region", { name: /synchronized viewport$/ })).toHaveLength(3);
  });
});
