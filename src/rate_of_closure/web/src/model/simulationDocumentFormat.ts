export interface SimulationDocumentFormat {
  readonly version: number;
  readonly web: boolean;
}

export const CURRENT_SIMULATION_DOCUMENT_VERSION = 5;
const SUPPORTED_NATIVE_VERSIONS = new Set([1, 2, 5]);
const SUPPORTED_WEB_VERSIONS = new Set([1, 2, 3, 4, 5]);

/** Parse and bound the native/web run-document envelope shared by all importers. */
export function simulationDocumentFormat(
  data: Record<string, unknown>,
): SimulationDocumentFormat | null {
  if (data.format === undefined) return null;
  const text = String(data.format);
  const match = text.match(/^rate_of_closure\.simulation_run(?:\.web)?\/(\d+)$/);
  if (!match) throw new Error(`Unsupported simulation format: ${text}.`);
  const format = { version: Number(match[1]), web: text.includes(".web/") };
  const supported = format.web ? SUPPORTED_WEB_VERSIONS : SUPPORTED_NATIVE_VERSIONS;
  if (!supported.has(format.version)) {
    throw new Error(`Unsupported simulation schema version ${format.version}.`);
  }
  return format;
}
