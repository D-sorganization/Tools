/** Shared browser boundary for bounded, fatal UTF-8 file decoding. */

export interface BoundedUtf8File {
  readonly name: string;
  readonly size: number;
  arrayBuffer(): Promise<ArrayBuffer>;
}

/** Read bounded bytes without trusting browser-supplied metadata. */
export const readBoundedUtf8File = async (
  file: BoundedUtf8File,
  maximumBytes: number,
  documentName: string,
): Promise<string> => {
  if (!Number.isSafeInteger(maximumBytes) || maximumBytes < 1) {
    throw new RangeError("maximumBytes must be a positive safe integer");
  }
  if (!Number.isSafeInteger(file.size) || file.size < 0) {
    throw new RangeError(documentName + " file size must be nonnegative");
  }
  if (file.size > maximumBytes) {
    throw new RangeError(documentName + " exceeds maximum wire size");
  }
  const buffer = await file.arrayBuffer();
  if (!(buffer instanceof ArrayBuffer)) {
    throw new TypeError(documentName + " file reader must return an ArrayBuffer");
  }
  if (buffer.byteLength > maximumBytes) {
    throw new RangeError(documentName + " exceeds maximum wire size");
  }
  try {
    return new TextDecoder("utf-8", { fatal: true }).decode(buffer);
  } catch {
    throw new RangeError(documentName + " must be valid UTF-8");
  }
};
