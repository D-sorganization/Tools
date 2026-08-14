"use strict";

const assert = require("node:assert/strict");
const toolsCore = require("../pkg/tools_core.js");
const {
  applyOverrides,
  assertConformanceCase,
  loadConformanceCases,
  materializeCase,
  parseStrictJson,
  resolveTemplatePath,
} = require("./ground_reference_conformance_support.cjs");

const pointerDocument = { "a/b": 1, "a~b": 2, items: [3, 4] };
applyOverrides(pointerDocument, {
  "/a~1b": 10,
  "/a~0b": 20,
  "/items/1": 40,
});
assert.deepEqual(pointerDocument, { "a/b": 10, "a~b": 20, items: [3, 40] });
for (const jsonPointer of ["/items/-1", "/items/01", "/items/2", "/items/~2"]) {
  assert.throws(() => applyOverrides({ items: [1, 2] }, { [jsonPointer]: 9 }));
}
assert.throws(() => parseStrictJson('{"outer":{"value":1,"value":2}}'));
assert.throws(() => resolveTemplatePath("../outside.json"));

const { template, cases } = loadConformanceCases();
for (const testCase of cases) {
  if (!testCase.platforms.includes("wasm")) {
    throw new Error(`${testCase.case_id} omits the WASM platform`);
  }
  const { request, execution } = materializeCase(template, testCase);
  const result = JSON.parse(
    toolsCore.runFlightToGroundReferenceV1(
      JSON.stringify(request),
      JSON.stringify(execution),
      undefined,
    ),
  );
  assertConformanceCase(result, request, testCase);
}
