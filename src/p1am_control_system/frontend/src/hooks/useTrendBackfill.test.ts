import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { naiveUtcIso, parseHistorianTs, useTrendBackfill } from "./useTrendBackfill";

// Mock the fetch seam so we can inspect the historian request URL. vi.hoisted
// lets the mock factory (hoisted above imports) reference the spy safely.
const { fetchMock } = vi.hoisted(() => ({ fetchMock: vi.fn() }));
vi.mock("../lib/fetchWithTimeout", () => ({
  fetchWithTimeout: (url: string) => fetchMock(url),
}));

describe("historian timestamp helpers", () => {
  const ms = Date.UTC(2026, 5, 29, 21, 0, 0); // 2026-06-29T21:00:00Z

  it("naiveUtcIso drops the zone suffix", () => {
    expect(naiveUtcIso(ms)).toBe("2026-06-29T21:00:00.000");
  });

  it("parseHistorianTs treats a zoneless stamp as UTC", () => {
    expect(parseHistorianTs("2026-06-29T21:00:00.000")).toBe(ms);
  });

  it("parseHistorianTs respects an explicit zone", () => {
    expect(parseHistorianTs("2026-06-29T21:00:00.000Z")).toBe(ms);
    expect(parseHistorianTs("2026-06-29T21:00:00.000+00:00")).toBe(ms);
  });

  it("round-trips naive UTC", () => {
    const t = Date.UTC(2026, 0, 2, 3, 4, 5);
    expect(parseHistorianTs(naiveUtcIso(t))).toBe(t);
  });
});

describe("useTrendBackfill request URL", () => {
  beforeEach(() => {
    fetchMock.mockReset();
    fetchMock.mockResolvedValue({
      ok: true,
      json: async () => ({ timestamps: [], values: [] }),
    });
  });

  it("requests a bounded max_points so a long window returns a light span", async () => {
    renderHook(() => useTrendBackfill(3, 6 * 3600, 1, 4000));
    await waitFor(() => expect(fetchMock).toHaveBeenCalled());
    const url = fetchMock.mock.calls[0][0] as string;
    expect(url).toContain("tag_id=3");
    expect(url).toContain("max_points=4000");
    expect(url).toContain("start_time=");
    expect(url).toContain("end_time=");
  });

  it("omits max_points when no bound is given (server default applies)", async () => {
    renderHook(() => useTrendBackfill(3, 3600));
    await waitFor(() => expect(fetchMock).toHaveBeenCalled());
    expect(fetchMock.mock.calls[0][0] as string).not.toContain("max_points");
  });

  it("does not fetch for an invalid tag id", () => {
    renderHook(() => useTrendBackfill(-1, 3600, 1, 4000));
    expect(fetchMock).not.toHaveBeenCalled();
  });
});
