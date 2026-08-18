import { describe, expect, it, vi } from "vitest";

import {
  ancestorClippedIntersection,
  revealMeaningfulVisual,
  scheduleMeaningfulVisualReveal,
} from "./variationVisualProminence";

const rect = (x: number, y: number, width: number, height: number): DOMRect =>
  new DOMRect(x, y, width, height);
const scrollSpy = (element: HTMLElement) => {
  element.scrollIntoView = vi.fn();
  return vi.mocked(element.scrollIntoView);
};

describe("Variation accepted-visual prominence", () => {
  it("clips through every overflow ancestor and the viewport", () => {
    const outer = document.createElement("div");
    const inner = document.createElement("div");
    const visual = document.createElement("div");
    outer.style.overflow = "hidden";
    inner.style.overflowY = "auto";
    outer.append(inner);
    inner.append(visual);
    vi.spyOn(outer, "getBoundingClientRect").mockReturnValue(rect(0, 0, 300, 220));
    vi.spyOn(inner, "getBoundingClientRect").mockReturnValue(rect(20, 10, 260, 180));
    vi.spyOn(visual, "getBoundingClientRect").mockReturnValue(rect(30, -40, 240, 240));

    expect(ancestorClippedIntersection(visual, { width: 390, height: 844 }))
      .toEqual({ x: 30, y: 10, width: 240, height: 180 });
  });

  it("reveals once only below the manifest-owned threshold", () => {
    const visual = document.createElement("div");
    vi.spyOn(visual, "getBoundingClientRect").mockReturnValue(rect(0, 900, 300, 300));
    const scroll = scrollSpy(visual);

    expect(revealMeaningfulVisual(visual, {
      viewport: { width: 390, height: 844 }, reducedMotion: false,
    })).toBe(true);
    expect(scroll).toHaveBeenCalledWith({ behavior: "smooth", block: "nearest", inline: "nearest" });

    scroll.mockClear();
    vi.spyOn(visual, "getBoundingClientRect").mockReturnValue(rect(20, 20, 240, 240));
    expect(revealMeaningfulVisual(visual, {
      viewport: { width: 390, height: 844 }, reducedMotion: false,
    })).toBe(false);
    expect(scroll).not.toHaveBeenCalled();
  });

  it("uses instant motion when the user requests reduced motion", () => {
    const visual = document.createElement("div");
    vi.spyOn(visual, "getBoundingClientRect").mockReturnValue(rect(0, 900, 300, 300));
    const scroll = scrollSpy(visual);

    revealMeaningfulVisual(visual, {
      viewport: { width: 1440, height: 900 }, reducedMotion: true,
    });

    expect(scroll).toHaveBeenCalledWith({ behavior: "auto", block: "nearest", inline: "nearest" });
  });

  it("defaults safely when an embedding window has no media-query API", () => {
    const visual = document.createElement("div");
    vi.spyOn(visual, "getBoundingClientRect").mockReturnValue(rect(0, 900, 300, 300));
    const scroll = scrollSpy(visual);
    const owner = {
      innerWidth: 390,
      innerHeight: 844,
      requestAnimationFrame: (callback: FrameRequestCallback) => {
        callback(0);
        return 1;
      },
    };

    expect(scheduleMeaningfulVisualReveal(() => visual, owner)).toBe(1);
    expect(scroll).toHaveBeenCalledWith({ behavior: "smooth", block: "nearest", inline: "nearest" });
  });

  it("does not fail an embedding DOM without scroll-into-view support", () => {
    const visual = document.createElement("div");
    vi.spyOn(visual, "getBoundingClientRect").mockReturnValue(rect(0, 900, 300, 300));

    expect(revealMeaningfulVisual(visual, {
      viewport: { width: 390, height: 844 }, reducedMotion: false,
    })).toBe(false);
  });

  it.each([
    [390, 844, 120, 179], [390, 844, 119, 180],
    [1440, 900, 240, 239], [1440, 900, 239, 240],
  ])("rejects a clipped sliver at %ix%i", (width, height, visualWidth, visualHeight) => {
    const visual = document.createElement("div");
    vi.spyOn(visual, "getBoundingClientRect")
      .mockReturnValue(rect(0, 0, visualWidth, visualHeight));
    const scroll = scrollSpy(visual);

    expect(revealMeaningfulVisual(visual, {
      viewport: { width, height }, reducedMotion: false,
    })).toBe(true);
    expect(scroll).toHaveBeenCalledOnce();
  });
});
