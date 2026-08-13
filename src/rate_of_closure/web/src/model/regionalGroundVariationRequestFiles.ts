/** Browser file adapter for combined regional-ground variation requests. */

import {
  MAX_REGIONAL_GROUND_VARIATION_REQUEST_BYTES,
  regionalGroundVariationRequestFromJson,
  stableRegionalGroundVariationRequestJson,
} from "./regionalGroundVariationRequestWire";
import type { RegionalGroundVariationRequestTs } from "./regionalGroundVariationWorkspace";
import {
  readBoundedUtf8File,
  type BoundedUtf8File,
} from "./boundedUtf8File";

export { MAX_REGIONAL_GROUND_VARIATION_REQUEST_BYTES };

export type RegionalGroundVariationRequestFile = BoundedUtf8File;

/** Read one bounded browser-selected file and completely validate it. */
export const readRegionalGroundVariationRequestFile = async (
  file: RegionalGroundVariationRequestFile,
): Promise<RegionalGroundVariationRequestTs> => {
  const text = await readBoundedUtf8File(
    file,
    MAX_REGIONAL_GROUND_VARIATION_REQUEST_BYTES,
    "regional-ground variation request",
  );
  return regionalGroundVariationRequestFromJson(text);
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
