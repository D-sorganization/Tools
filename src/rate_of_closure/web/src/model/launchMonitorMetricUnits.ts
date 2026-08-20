const COLUMN_UNITS: Array<[RegExp, string]> = [
  [/(angle|path|direction|loft|azimuth|elevation)/i, "deg"],
  [/(spin|rpm)/i, "rpm"],
  [/(speed.*mph|mph)/i, "mph"],
  [/(speed.*mps|speed.*m_s|velocity.*m_s)/i, "m/s"],
  [/(yard|_yd$|carry_distance|total_distance|offline)/i, "yd"],
  [/(_m$|meters?|metres?|observed_lateral_m)/i, "m"],
  [/(_s$|seconds?|flight_time)/i, "s"],
  [/(percent|percentage|rate$)/i, "%"],
];

export const metricUnit = (column: string): string =>
  COLUMN_UNITS.find(([pattern]) => pattern.test(column))?.[1] ?? "unit unknown";

export const metricLabel = (column: string): string =>
  `${column.replace(/_/g, " ")} (${metricUnit(column)})`;
