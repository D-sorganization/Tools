/**
 * Temperature-source domain types + labels for the heater controls.
 *
 * The heater reads temperature from one of four sources: a type-K or type-R
 * thermocouple, each available on two acquisition paths — straight into the
 * P1-04THM card, or through a 4-20 mA signal conditioner on an analog input.
 * These are shared by the control component and its tests, so they live in
 * `lib/` (keeps the component file exporting only components — fast-refresh
 * clean — and gives one source of truth for how the sources are named).
 */

/** Thermocouple type selectable for the heater control. */
export type TcType = "K" | "R";

/** Acquisition path a thermocouple's reading arrives through: straight into the
 *  P1-04THM card ("thm") or via a 4-20 mA signal conditioner on an analog input
 *  ("analog"). Orthogonal to {@link TcType}, so the operator picks one of four
 *  sources (K/R x card/analog). */
export type TcPath = "thm" | "analog";

/** Human label for a (type, path) source, e.g. "Analog Type R" (single source
 *  of truth for how the four sources are named across the HMI). */
export function sourceLabel(tcType: TcType, tcPath: TcPath): string {
  return `${tcPath === "analog" ? "Analog" : "TC Card"} Type ${tcType}`;
}

/** The four selectable temperature sources (type x path), in HMI display order:
 *  the two TC-card sources first, then the two signal-conditioned analog ones. */
export const TEMP_SOURCES: ReadonlyArray<{ type: TcType; path: TcPath }> = [
  { type: "K", path: "thm" },
  { type: "R", path: "thm" },
  { type: "K", path: "analog" },
  { type: "R", path: "analog" },
];
