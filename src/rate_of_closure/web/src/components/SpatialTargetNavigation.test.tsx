import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import App from "../App";

describe("canonical spatial target navigation", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn(() => new Promise<Response>(() => {})));
    const stored = new Map<string, string>();
    Object.defineProperty(window, "localStorage", {
      configurable: true,
      value: {
        clear: () => stored.clear(),
        getItem: (key: string) => stored.get(key) ?? null,
        removeItem: (key: string) => stored.delete(key),
        setItem: (key: string, value: string) => stored.set(key, value),
      },
    });
    const context: unknown = new Proxy(function () {} as object, {
      get: () => () => context,
      set: () => true,
      apply: () => context,
    });
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(
      context as CanvasRenderingContext2D,
    );
  });

  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("preserves one edited target across Flight Explorer unmount and Simulation", () => {
    render(<App />);
    fireEvent.click(screen.getByRole("tab", { name: "Flight Explorer" }));
    fireEvent.change(screen.getByLabelText("Target downrange m"), {
      target: { value: "180" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Apply spatial target" }));

    fireEvent.click(screen.getByRole("tab", { name: "Simulation" }));
    const displayTabs = screen.getByRole("tablist", {
      name: "Display views (scale-separated)",
    });
    fireEvent.click(within(displayTabs).getByRole("tab", { name: "Flight" }));
    expect(screen.getByLabelText("Target downrange m")).toHaveValue("180");

    fireEvent.click(screen.getByRole("tab", { name: "Flight Explorer" }));
    expect(screen.getByLabelText("Target downrange m")).toHaveValue("180");
  });
});
