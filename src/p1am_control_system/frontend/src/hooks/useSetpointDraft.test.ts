import { describe, it, expect } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useSetpointDraft } from "./useSetpointDraft";

/**
 * Operator-owned setpoint draft (#4013, #4020).
 *
 * The Alicat entry was re-seeded by an effect keyed on `[inspectorView,
 * alicats]`; `alicats` changes reference on essentially every frame on real
 * hardware, so the entry was overwritten between keystrokes and Set could
 * commit a partially-reverted value. The power-supply entry had the opposite
 * bug — never seeded, so it read "0" while the supply ran at 30 A.
 *
 * This hook is the shared latch: seed from the device while untouched, hand
 * ownership to the operator on the first edit, and only re-seed when the entry
 * is re-scoped to a different device.
 */

describe("useSetpointDraft", () => {
  it("seeds from the device value on first sight", () => {
    const { result } = renderHook(
      ({ v }) => useSetpointDraft(v, { scope: "MFC1", digits: 2 }),
      { initialProps: { v: 25 as number | undefined } },
    );
    expect(result.current.text).toBe("25.00");
  });

  it("stays empty until a device value exists", () => {
    const { result } = renderHook(() =>
      useSetpointDraft(undefined, { scope: "MFC1", digits: 2 }),
    );
    expect(result.current.text).toBe("");
    expect(result.current.touched).toBe(false);
  });

  it("re-seeds from later frames while the operator has not typed", () => {
    const { result, rerender } = renderHook(
      ({ v }) => useSetpointDraft(v, { scope: "PS", digits: 2 }),
      { initialProps: { v: undefined as number | undefined } },
    );
    expect(result.current.text).toBe("");
    rerender({ v: 30 });
    expect(result.current.text).toBe("30.00");
  });

  it("NEVER overwrites the entry once the operator has typed", () => {
    const { result, rerender } = renderHook(
      ({ v }) => useSetpointDraft(v, { scope: "MFC1", digits: 2 }),
      { initialProps: { v: 25 as number | undefined } },
    );

    act(() => result.current.setText("2"));
    expect(result.current.text).toBe("2");

    // Ten frames land mid-entry, each with a slightly different live value —
    // exactly what real hardware does. The draft must not budge.
    for (const v of [25.01, 24.98, 25.03, 25, 24.99]) {
      rerender({ v });
      expect(result.current.text).toBe("2");
    }
    expect(result.current.touched).toBe(true);
  });

  it("re-seeds and drops ownership when the entry is scoped to another device", () => {
    const { result, rerender } = renderHook(
      ({ v, scope }) => useSetpointDraft(v, { scope, digits: 2 }),
      { initialProps: { v: 25 as number | undefined, scope: "MFC1" } },
    );
    act(() => result.current.setText("2"));
    expect(result.current.text).toBe("2");

    // Selecting a different MFC is a new entry, not the same one being fought
    // over: it must show the new device's setpoint.
    rerender({ v: 7.5, scope: "MFC2" });
    expect(result.current.text).toBe("7.50");
    expect(result.current.touched).toBe(false);
  });

  it("reseed() returns the field to the device value on demand", () => {
    const { result } = renderHook(() =>
      useSetpointDraft(30, { scope: "PS", digits: 2 }),
    );
    act(() => result.current.setText("999"));
    expect(result.current.text).toBe("999");

    act(() => result.current.reseed());
    expect(result.current.text).toBe("30.00");
    expect(result.current.touched).toBe(false);
  });

  it("keeps setText referentially stable so memoized inputs do not churn", () => {
    const { result, rerender } = renderHook(
      ({ v }) => useSetpointDraft(v, { scope: "PS", digits: 2 }),
      { initialProps: { v: 30 as number | undefined } },
    );
    const first = result.current.setText;
    rerender({ v: 30.4 });
    expect(result.current.setText).toBe(first);
  });
});
