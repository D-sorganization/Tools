import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
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

  it("keeps every established simulation display reachable beside Multi View", async () => {
    render(<App />);
    fireEvent.change(screen.getByRole("combobox", { name: "Club" }), {
      target: { value: "Sand Wedge" },
    });
    fireEvent.click(screen.getByRole("button", { name: /^Impact$/ }));

    const displayTabs = await screen.findByRole("tablist", {
      name: "Display views (scale-separated)",
    });
    const tabs = within(displayTabs);
    expect(tabs.getByRole("tab", { name: "Multi View" }))
      .toHaveAttribute("aria-selected", "true");

    fireEvent.click(tabs.getByRole("tab", { name: "Swing" }));
    expect(await screen.findByRole("complementary", {
      name: "Impact Kinematics Engineering Readout",
    })).toBeInTheDocument();
    expect(screen.getByRole("region", { name: "Interactive Impact Scene" }))
      .toBeInTheDocument();
    expect(screen.getByRole("complementary", {
      name: "Wedge Ground-Clearance Engineering Readout",
    })).toBeInTheDocument();

    fireEvent.click(tabs.getByRole("tab", { name: "Kinetics" }));
    expect(screen.getByText(/Kinetics need the pendulum joint states/))
      .toBeInTheDocument();

    fireEvent.click(tabs.getByRole("tab", { name: "Flight" }));
    expect(screen.getByLabelText("Flight side profile (height vs carry)"))
      .toBeInTheDocument();
    expect(screen.getByLabelText("Flight top-down view (lateral vs carry)"))
      .toBeInTheDocument();
  });
});
