import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { generatedHeadFor } from "../model/clubHeadGeneration";
import { DEFAULT_CLUB_CAMERA, type ClubCamera } from "../model/clubCamera";
import {
  generatedMeshSource,
  proceduralMeshSource,
  type ClubMeshSource,
} from "../model/clubMeshSource";
import { getClub } from "../model/club";
import { DEFAULT_SCENARIO } from "../model/impact";
import * as impactModel from "../model/impact";
import { MAX_STL_BYTES } from "../model/mesh";
import { writeBinaryStlFixture } from "../model/meshTestSupport";
import { ClubCanvas } from "./ClubCanvas";

class DeferredFileReader {
  static instances: DeferredFileReader[] = [];
  result: string | ArrayBuffer | null = null;
  onload: ((event: ProgressEvent<FileReader>) => void) | null = null;
  onerror: ((event: ProgressEvent<FileReader>) => void) | null = null;
  onabort: ((event: ProgressEvent<FileReader>) => void) | null = null;
  aborted = false;

  constructor() { DeferredFileReader.instances.push(this); }
  readAsArrayBuffer(): void { /* completion is controlled by each test */ }
  abort(): void { this.aborted = true; this.onabort?.({} as ProgressEvent<FileReader>); }
  finish(buffer: ArrayBuffer): void {
    this.result = buffer;
    this.onload?.({} as ProgressEvent<FileReader>);
  }
  fail(): void { this.onerror?.({} as ProgressEvent<FileReader>); }
}

function Harness({ generated = false }: { generated?: boolean }) {
  const head = generatedHeadFor(getClub("Driver 10.5°"));
  const [source, setSource] = useState<ClubMeshSource>(() => generated
    ? generatedMeshSource(head, head.label, 0) : proceduralMeshSource());
  const [camera, setCamera] = useState<ClubCamera>(DEFAULT_CLUB_CAMERA);
  return <>
    <button onClick={() => setSource((prior) =>
      generatedMeshSource(head, head.label, prior.generation + 1))}>Generate parent head</button>
    <button onClick={() => setSource((prior) =>
      proceduralMeshSource(prior.generation))}>Replace same generation</button>
    <ClubCanvas scenario={DEFAULT_SCENARIO} source={source}
      onSourceChange={setSource} camera={camera} onCameraChange={setCamera} />
  </>;
}

beforeEach(() => {
  DeferredFileReader.instances = [];
  vi.stubGlobal("FileReader", DeferredFileReader);
  class TestPointerEvent extends MouseEvent {
    readonly pointerId: number;
    readonly isPrimary: boolean;
    constructor(type: string, init: PointerEventInit = {}) {
      super(type, init);
      this.pointerId = init.pointerId ?? 0;
      this.isPrimary = init.isPrimary ?? false;
    }
  }
  vi.stubGlobal("PointerEvent", TestPointerEvent);
  window.matchMedia = vi.fn().mockReturnValue({ matches: true });
  const gradient = { addColorStop: vi.fn() };
  const context = new Proxy({
    canvas: document.createElement("canvas"),
    createRadialGradient: () => gradient,
    measureText: () => ({ width: 10 }),
  }, { get: (target, key) => Reflect.get(target, key) ?? vi.fn() });
  vi.spyOn(HTMLCanvasElement.prototype, "getContext")
    .mockReturnValue(context as unknown as CanvasRenderingContext2D);
});

afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals(); });

function choose(file: File): void {
  fireEvent.change(document.querySelector("input[type=file]")!, {
    target: { files: [file] },
  });
}

describe("ClubCanvas camera and atomic sources", () => {
  it("solves once while presentation camera and controls redraw", async () => {
    const user = userEvent.setup();
    const solve = vi.spyOn(impactModel, "solve");
    render(<Harness />);
    const canvas = screen.getByRole("img", { name: /Interactive 3D clubhead camera/ });
    canvas.focus();
    await user.keyboard("{ArrowLeft}+{Home}");
    await user.click(screen.getByRole("checkbox", { name: "Show reference marker" }));
    await user.selectOptions(screen.getByRole("combobox"), "Head Fixed in Place");
    expect(solve).toHaveBeenCalledTimes(1);
  });

  it("preserves canvas aspect while uniformly capping a high-DPR backing store", () => {
    Object.defineProperty(window, "devicePixelRatio", { configurable: true, value: 2.5 });
    vi.spyOn(HTMLCanvasElement.prototype, "getBoundingClientRect").mockReturnValue({
      width: 1600, height: 1088, x: 0, y: 0, top: 0, left: 0,
      right: 1600, bottom: 1088, toJSON: () => ({}),
    });
    render(<Harness />);
    const canvas = screen.getByRole("img", { name: /Interactive 3D clubhead camera/ });
    expect(canvas).toHaveAttribute("width", "2048");
    expect(Number(canvas.getAttribute("height")) / 2048).toBeCloseTo(0.68, 2);
  });

  it("stops playback and exposes Play after a fatal timer render", () => {
    vi.useFakeTimers();
    window.matchMedia = vi.fn().mockReturnValue({ matches: false });
    render(<Harness />);
    expect(screen.getByRole("button", { name: "Pause" })).toBeVisible();
    vi.mocked(HTMLCanvasElement.prototype.getContext).mockReturnValue(null);
    act(() => vi.advanceTimersByTime(40));
    expect(screen.getByRole("alert")).toHaveTextContent("rendering context is unavailable");
    expect(screen.getByRole("button", { name: "Play" })).toBeVisible();
    vi.useRealTimers();
  });

  it("stops live playback when reduced-motion preference becomes active", () => {
    let changed: (() => void) | undefined;
    const media = {
      matches: false,
      addEventListener: (_name: string, callback: () => void) => { changed = callback; },
      removeEventListener: vi.fn(),
    };
    window.matchMedia = vi.fn().mockReturnValue(media);
    render(<Harness />);
    expect(screen.getByRole("button", { name: "Pause" })).toBeVisible();
    media.matches = true;
    act(() => changed?.());
    expect(screen.getByRole("button", { name: "Play" })).toBeVisible();
  });

  it("operates one focusable camera by keyboard, pointer, wheel, and reset", async () => {
    const user = userEvent.setup();
    render(<Harness />);
    const canvas = screen.getByRole("img", { name: /Interactive 3D clubhead camera/ });
    canvas.focus();
    fireEvent.keyDown(canvas, { key: "ArrowLeft" });
    expect(screen.getByRole("status")).toHaveTextContent("azimuth 145°");
    fireEvent.pointerDown(canvas, {
      pointerId: 2, clientX: 20, clientY: 20, isPrimary: true, button: 0,
    });
    fireEvent.pointerMove(canvas, {
      pointerId: 2, clientX: 30, clientY: 10, isPrimary: true,
    });
    expect(screen.getByRole("status")).toHaveTextContent("elevation 26°");
    fireEvent.wheel(canvas, { deltaY: -1 });
    expect(screen.getByRole("status")).toHaveTextContent("zoom 1.10×");
    fireEvent.wheel(canvas, { deltaY: 0 });
    expect(screen.getByRole("status")).toHaveTextContent("zoom 1.10×");
    fireEvent.keyDown(canvas, { key: "Home" });
    expect(screen.getByRole("status")).toHaveTextContent("azimuth 150°");
    expect(document.activeElement).toBe(canvas);
    const reset = screen.getByRole("button", { name: "Reset View" });
    await user.click(reset);
    expect(document.activeElement).toBe(reset);
  });

  it("does not schedule recurring redraws under reduced motion while paused", () => {
    const interval = vi.spyOn(window, "setInterval");
    render(<Harness />);
    expect(interval).not.toHaveBeenCalled();
  });

  it("ignores foreign pointers and safely ends only the captured drag", () => {
    render(<Harness />);
    const canvas = screen.getByRole("img", { name: /Interactive 3D clubhead camera/ });
    fireEvent.pointerDown(canvas, {
      pointerId: 2, clientX: 20, clientY: 20, isPrimary: true, button: 0,
    });
    fireEvent.pointerMove(canvas, {
      pointerId: 3, clientX: 100, clientY: 100, isPrimary: false,
    });
    expect(screen.getByRole("status")).toHaveTextContent("azimuth 150°");
    fireEvent.pointerUp(canvas, { pointerId: 3 });
    fireEvent.pointerMove(canvas, {
      pointerId: 2, clientX: 30, clientY: 20, isPrimary: true,
    });
    expect(screen.getByRole("status")).toHaveTextContent("azimuth 146°");
    fireEvent.pointerCancel(canvas, { pointerId: 2 });
    fireEvent.pointerMove(canvas, {
      pointerId: 2, clientX: 40, clientY: 20, isPrimary: true,
    });
    expect(screen.getByRole("status")).toHaveTextContent("azimuth 146°");
  });

  it("contains a missing rendering context as a bounded alert", () => {
    vi.mocked(HTMLCanvasElement.prototype.getContext).mockReturnValue(null);
    render(<Harness />);
    expect(screen.getByRole("alert")).toHaveTextContent("rendering context is unavailable");
  });

  it("rejects suffix and size before constructing a reader", () => {
    render(<Harness />);
    choose(new File([new Uint8Array(8)], "head.obj"));
    expect(screen.getByRole("alert")).toHaveTextContent(".stl suffix");
    choose(new File([new Uint8Array(MAX_STL_BYTES + 1)], "head.stl"));
    expect(screen.getByRole("alert")).toHaveTextContent("2 MiB");
    expect(DeferredFileReader.instances).toHaveLength(0);
  });

  it.each([
    ["suffix", 8, "replacement.obj"],
    ["size", MAX_STL_BYTES + 1, "replacement.stl"],
  ] as const)("invalid replacement %s supersedes a pending valid import", async (
    _case, bytes, name,
  ) => {
    render(<Harness />);
    choose(new File([new Uint8Array(1)], "valid.stl"));
    const pending = DeferredFileReader.instances[0];
    choose(new File([new Uint8Array(bytes)], name));
    expect(pending.aborted).toBe(true);
    pending.finish(writeBinaryStlFixture());
    await act(async () => Promise.resolve());
    expect(screen.getByRole("status")).toHaveTextContent("Procedural head");
    expect(screen.getByRole("alert")).toHaveTextContent("STL load failed");
  });

  it("accepts only the latest reader and suppresses stale success and error", async () => {
    render(<Harness />);
    choose(new File([new Uint8Array(1)], "a.stl"));
    choose(new File([new Uint8Array(1)], "b.stl"));
    const [first, second] = DeferredFileReader.instances;
    await act(async () => { second.finish(writeBinaryStlFixture()); });
    await waitFor(() => expect(screen.getByRole("status")).toHaveTextContent("Imported b.stl"));
    await act(async () => { first.finish(writeBinaryStlFixture()); first.fail(); });
    expect(screen.getByRole("status")).toHaveTextContent("Imported b.stl");
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("procedural and generated replacement abort pending reads without resurrection", () => {
    render(<Harness generated />);
    choose(new File([new Uint8Array(1)], "a.stl"));
    const first = DeferredFileReader.instances[0];
    fireEvent.click(screen.getByRole("button", { name: "Procedural Head" }));
    expect(first.aborted).toBe(true);
    expect(screen.getByRole("status")).toHaveTextContent("Procedural head");
    choose(new File([new Uint8Array(1)], "b.stl"));
    const second = DeferredFileReader.instances[1];
    fireEvent.click(screen.getByRole("button", { name: "Generate parent head" }));
    expect(second.aborted).toBe(true);
    expect(screen.getByRole("status")).toHaveTextContent("Generated representative");
  });

  it("aborts the active reader on unmount", () => {
    const view = render(<Harness />);
    choose(new File([new Uint8Array(1)], "a.stl"));
    const reader = DeferredFileReader.instances[0];
    view.unmount();
    expect(reader.aborted).toBe(true);
  });

  it("rejects a stale reader after an exact-source replacement at one generation", () => {
    render(<Harness generated />);
    choose(new File([new Uint8Array(1)], "pending.stl"));
    const reader = DeferredFileReader.instances[0];
    fireEvent.click(screen.getByRole("button", { name: "Replace same generation" }));
    expect(reader.aborted).toBe(true);
    reader.finish(writeBinaryStlFixture());
    expect(screen.getByRole("status")).toHaveTextContent("Procedural head");
  });
});
