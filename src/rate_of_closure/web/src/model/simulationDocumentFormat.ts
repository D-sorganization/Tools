export interface SimulationDocumentFormat {
  readonly version: number;
  readonly web: boolean;
}

export const CURRENT_SIMULATION_DOCUMENT_VERSION = 5;

/** Parse and bound the native/web run-document envelope shared by all importers. */
export function simulationDocumentFormat(
  data: Record<string, unknown>,
): SimulationDocumentFormat | null {
  if (data.format === undefined) return null;
  const text = String(data.format);
  const match = text.match(/^rate_of_closure\.simulation_run(?:\.web)?\/(\d+)$/);
  if (!match) throw new Error(`Unsupported simulation format: ${text}.`);
  const format = { version: Number(match[1]), web: text.includes(".web/") };
  if (format.version < 1 || format.version > CURRENT_SIMULATION_DOCUMENT_VERSION) {
    throw new Error(`Unsupported simulation schema version ${format.version}.`);
  }
  return format;
}
