/** Browser file adapter for combined regional-ground variation requests. */

import {
  MAX_REGIONAL_GROUND_VARIATION_REQUEST_BYTES,
  regionalGroundVariationRequestFromJson,
  stableRegionalGroundVariationRequestJson,
} from "./regionalGroundVariationRequestWire";
import type { RegionalGroundVariationRequestTs } from "./regionalGroundVariationWorkspace";

export { MAX_REGIONAL_GROUND_VARIATION_REQUEST_BYTES };

export interface RegionalGroundVariationRequestFile {
  readonly name: string;
  readonly size: number;
  arrayBuffer(): Promise<ArrayBuffer>;
}

const decodeUtf8 = (buffer: ArrayBuffer): string => {
  try {
    return new TextDecoder("utf-8", { fatal: true }).decode(buffer);
  } catch {
    throw new RangeError("regional-ground variation request must be valid UTF-8");
  }
};

/** Read one bounded browser-selected file and completely validate it. */
export const readRegionalGroundVariationRequestFile = async (
  file: RegionalGroundVariationRequestFile,
): Promise<RegionalGroundVariationRequestTs> => {
  if (!Number.isSafeInteger(file.size) || file.size < 0) {
    throw new RangeError("regional-ground variation request file size must be nonnegative");
  }
  if (file.size > MAX_REGIONAL_GROUND_VARIATION_REQUEST_BYTES) {
    throw new RangeError("regional-ground variation request exceeds maximum wire size");
  }
  const buffer = await file.arrayBuffer();
  if (buffer.byteLength > MAX_REGIONAL_GROUND_VARIATION_REQUEST_BYTES) {
    throw new RangeError("regional-ground variation request exceeds maximum wire size");
  }
  return regionalGroundVariationRequestFromJson(decodeUtf8(buffer));
};

/** Start a canonical browser download; destination and replacement remain browser-owned. */
export const downloadRegionalGroundVariationRequest = (
  request: RegionalGroundVariationRequestTs,
): void => {
  const text = stableRegionalGroundVariationRequestJson(request);
  const url = URL.createObjectURL(
    new Blob([text], { type: "application/json;charset=utf-8" }),
  );
  try {
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "regional-ground-variation-request.json";
    anchor.click();
  } finally {
    URL.revokeObjectURL(url);
  }
};
