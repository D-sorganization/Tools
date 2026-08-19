import { describe, expect, it, vi } from "vitest";

import goldenRequest from "./__fixtures__/regional_ground_variation_request_golden_v1.json";
import {
  MAX_REGIONAL_GROUND_VARIATION_REQUEST_BYTES,
  readRegionalGroundVariationRequestFile,
} from "./regionalGroundVariationRequestFiles";

describe("regional-ground variation request browser files", () => {
  const bytes = (text: string): ArrayBuffer =>
    Uint8Array.from(new TextEncoder().encode(text)).buffer;

  it("rejects oversized input before allocating text", async () => {
    const arrayBuffer = vi.fn().mockResolvedValue(bytes("{}"));

    await expect(readRegionalGroundVariationRequestFile({
      name: "oversized.json",
      size: MAX_REGIONAL_GROUND_VARIATION_REQUEST_BYTES + 1,
      arrayBuffer,
    })).rejects.toThrow(/maximum wire size/i);
    expect(arrayBuffer).not.toHaveBeenCalled();
  });

  it("parses a bounded Python-canonical request", async () => {
    const text = JSON.stringify(goldenRequest);
    const request = await readRegionalGroundVariationRequestFile({
      name: "request.json",
      size: new TextEncoder().encode(text).byteLength,
      arrayBuffer: vi.fn().mockResolvedValue(bytes(text)),
    });

    expect(request.seriesId).toBe("driver");
  });

  it("does not trust an underreported browser file size", async () => {
    const text = " ".repeat(MAX_REGIONAL_GROUND_VARIATION_REQUEST_BYTES + 1);

    await expect(readRegionalGroundVariationRequestFile({
      name: "dishonest.json", size: 1,
      arrayBuffer: vi.fn().mockResolvedValue(bytes(text)),
    })).rejects.toThrow(/maximum wire size/i);
  });

  it("rejects malformed UTF-8 bytes before parsing", async () => {
    await expect(readRegionalGroundVariationRequestFile({
      name: "invalid-utf8.json", size: 2,
      arrayBuffer: vi.fn().mockResolvedValue(Uint8Array.from([0xc3, 0x28]).buffer),
    })).rejects.toThrow(/valid utf-8/i);
  });
});
