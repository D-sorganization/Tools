import { describe, expect, it } from "vitest";
import { recallSetpointText } from "./TemperatureControl";

/**
 * Unit tests for the pure setpoint-recall helper. Rendering the full
 * TemperatureControl pulls in the WebSocket-fed trend and the config fetch,
 * so the pre-fill decision is extracted into recallSetpointText() and tested
 * directly here (light + green, per the task).
 */
describe("recallSetpointText", () => {
  it("pre-fills the recalled last setpoint when untouched", () => {
    expect(recallSetpointText(250, false)).toBe("250.0");
    expect(recallSetpointText(37.25, false)).toBe("37.3");
    expect(recallSetpointText(0, false)).toBe("0.0");
  });

  it("does not overwrite once the operator has typed this session", () => {
    expect(recallSetpointText(250, true)).toBeNull();
  });

  it("returns null for a missing or non-finite recalled value", () => {
    expect(recallSetpointText(null, false)).toBeNull();
    expect(recallSetpointText(undefined, false)).toBeNull();
    expect(recallSetpointText(Number.NaN, false)).toBeNull();
    expect(recallSetpointText(Infinity, false)).toBeNull();
  });

  it("guards its precondition (DbC)", () => {
    // @ts-expect-error deliberately wrong type for the guard test
    expect(() => recallSetpointText(100, "yes")).toThrow(TypeError);
  });
});
