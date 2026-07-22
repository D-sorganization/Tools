import { describe, it, expect, beforeEach } from "vitest";
import { render, fireEvent } from "@testing-library/react";

import { PanelStack, type PanelItem } from "./PanelStack";
import { loadPanelLayout } from "../lib/panelLayout";

const ids = ["a", "b", "c"];
const panels: PanelItem[] = ids.map((id) => ({
  id,
  title: id,
  node: <div data-testid={`body-${id}`}>content-{id}</div>,
}));

/** The rendered top-to-bottom panel order, read from each body's testid. */
function renderedOrder(container: HTMLElement): string[] {
  return Array.from(container.querySelectorAll(".panel-stack-item")).map((s) =>
    (s.querySelector("[data-testid^='body-']")?.getAttribute("data-testid") ?? "")
      .replace("body-", ""),
  );
}

describe("PanelStack", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it("renders panels in the declared order with a grip + resize gutter each", () => {
    const { container } = render(<PanelStack regionId="t" panels={panels} />);
    expect(renderedOrder(container)).toEqual(["a", "b", "c"]);
    expect(container.querySelectorAll(".panel-grip").length).toBe(3);
    expect(container.querySelectorAll(".panel-resize-gutter").length).toBe(3);
  });

  it("reorders on drag-and-drop and persists the new order", () => {
    const { container } = render(<PanelStack regionId="t" panels={panels} />);
    const grips = container.querySelectorAll(".panel-grip");
    const sections = container.querySelectorAll(".panel-stack-item");
    fireEvent.dragStart(grips[2]); // grab "c"
    fireEvent.drop(sections[0]); // drop it before "a"
    expect(renderedOrder(container)).toEqual(["c", "a", "b"]);
    expect(loadPanelLayout("t", ids).order).toEqual(["c", "a", "b"]);
  });

  it("applies a persisted body height to the panel", () => {
    localStorage.setItem(
      "p1am.panelLayout.t.v1",
      JSON.stringify({ order: ids, heights: { b: 321 } }),
    );
    const { getByTestId } = render(<PanelStack regionId="t" panels={panels} />);
    const body = getByTestId("body-b").parentElement as HTMLElement;
    expect(body.style.height).toBe("321px");
  });

  it("resizes a panel via the gutter and commits the height", () => {
    const { container, getByTestId } = render(
      <PanelStack regionId="t" panels={panels} />,
    );
    const body = getByTestId("body-a").parentElement as HTMLElement;
    body.getBoundingClientRect = () =>
      ({ height: 200, width: 0, top: 0, left: 0, right: 0, bottom: 200, x: 0, y: 0, toJSON: () => ({}) }) as DOMRect;
    const gutter = container.querySelector(".panel-resize-gutter") as HTMLElement;
    fireEvent.pointerDown(gutter, { clientY: 500 });
    fireEvent.pointerMove(gutter, { clientY: 640 }); // +140 -> 340
    fireEvent.pointerUp(gutter, { clientY: 640 });
    expect(loadPanelLayout("t", ids).heights.a).toBe(340);
  });
});
