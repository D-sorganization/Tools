"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const fixtureDirectory = path.resolve(
  __dirname,
  "../../../src/rate_of_closure/web/src/model/__fixtures__",
);
const corpusPath = path.join(
  fixtureDirectory,
  "ground_reference_conformance_v1.json",
);
const templateFixture = "ground_reference_pipeline_golden_v1.json";

function resolveTemplatePath(reference) {
  assert.equal(reference, templateFixture, "untrusted conformance template");
  return path.join(fixtureDirectory, templateFixture);
}

function decodePointerToken(token) {
  let decoded = "";
  for (let index = 0; index < token.length; index += 1) {
    if (token[index] !== "~") {
      decoded += token[index];
      continue;
    }
    const escape = token[index + 1];
    assert.ok(
      escape === "0" || escape === "1",
      `invalid pointer escape: ${token}`,
    );
    decoded += escape === "0" ? "~" : "/";
    index += 1;
  }
  return decoded;
}

function pointerTokens(jsonPointer) {
  if (jsonPointer === "") return [];
  assert.ok(
    jsonPointer.startsWith("/"),
    `invalid JSON pointer: ${jsonPointer}`,
  );
  return jsonPointer.slice(1).split("/").map(decodePointerToken);
}

function arrayIndex(token, length) {
  assert.match(
    token,
    /^(0|[1-9][0-9]*)$/,
    `noncanonical array index: ${token}`,
  );
  const index = Number(token);
  assert.ok(
    Number.isSafeInteger(index) && index < length,
    `array index out of range: ${token}`,
  );
  return index;
}

function childAt(current, token) {
  if (Array.isArray(current)) return current[arrayIndex(token, current.length)];
  assert.ok(
    current !== null && typeof current === "object",
    `pointer traverses scalar: ${token}`,
  );
  assert.ok(
    Object.hasOwn(current, token),
    `pointer key does not exist: ${token}`,
  );
  return current[token];
}

function pointer(document, jsonPointer) {
  return pointerTokens(jsonPointer).reduce(childAt, document);
}

function applyOverrides(document, overrides) {
  for (const [jsonPointer, replacement] of Object.entries(overrides)) {
    const tokens = pointerTokens(jsonPointer);
    assert.ok(
      tokens.length > 0,
      "an override cannot replace the document root",
    );
    const leaf = tokens.at(-1);
    const parent = tokens.slice(0, -1).reduce(childAt, document);
    const target = Array.isArray(parent)
      ? arrayIndex(leaf, parent.length)
      : leaf;
    assert.ok(
      parent !== null &&
        typeof parent === "object" &&
        Object.hasOwn(parent, target),
      `override is not a leaf: ${jsonPointer}`,
    );
    parent[target] = structuredClone(replacement);
  }
}

function scanJsonString(text, state) {
  const start = state.index;
  assert.equal(text[state.index], '"');
  state.index += 1;
  while (state.index < text.length) {
    if (text[state.index] === "\\") {
      state.index += 2;
    } else if (text[state.index] === '"') {
      state.index += 1;
      return JSON.parse(text.slice(start, state.index));
    } else {
      state.index += 1;
    }
  }
  throw new Error("unterminated JSON string");
}

function skipWhitespace(text, state) {
  while (/\s/.test(text[state.index] ?? "")) state.index += 1;
}

function scanJsonArray(text, state) {
  state.index += 1;
  skipWhitespace(text, state);
  if (text[state.index] === "]") {
    state.index += 1;
    return;
  }
  while (state.index < text.length) {
    scanJsonValue(text, state);
    skipWhitespace(text, state);
    if (text[state.index] === "]") {
      state.index += 1;
      return;
    }
    assert.equal(text[state.index], ",");
    state.index += 1;
  }
}

function scanJsonObject(text, state) {
  state.index += 1;
  const keys = new Set();
  skipWhitespace(text, state);
  if (text[state.index] === "}") {
    state.index += 1;
    return;
  }
  while (state.index < text.length) {
    const key = scanJsonString(text, state);
    assert.ok(!keys.has(key), `duplicate JSON key: ${key}`);
    keys.add(key);
    skipWhitespace(text, state);
    assert.equal(text[state.index], ":");
    state.index += 1;
    scanJsonValue(text, state);
    skipWhitespace(text, state);
    if (text[state.index] === "}") {
      state.index += 1;
      return;
    }
    assert.equal(text[state.index], ",");
    state.index += 1;
    skipWhitespace(text, state);
  }
}

function scanJsonValue(text, state) {
  skipWhitespace(text, state);
  if (text[state.index] === "{") return scanJsonObject(text, state);
  if (text[state.index] === "[") return scanJsonArray(text, state);
  if (text[state.index] === '"') return void scanJsonString(text, state);
  while (state.index < text.length && !/[\s,\]}]/.test(text[state.index])) {
    state.index += 1;
  }
}

function parseStrictJson(text) {
  const parsed = JSON.parse(text);
  const state = { index: 0 };
  scanJsonValue(text, state);
  skipWhitespace(text, state);
  assert.equal(state.index, text.length, "unconsumed JSON text");
  return parsed;
}

function dot(left, right) {
  return left.reduce((total, value, index) => total + value * right[index], 0);
}

function cross(left, right) {
  return [
    left[1] * right[2] - left[2] * right[1],
    left[2] * right[0] - left[0] * right[2],
    left[0] * right[1] - left[1] * right[0],
  ];
}

function assertClose(actual, expected, check) {
  const error = Math.abs(actual - expected);
  const scale = Math.max(Math.abs(actual), Math.abs(expected));
  const tolerance = Math.max(
    check.absolute_tolerance,
    check.relative_tolerance * scale,
  );
  assert.ok(error <= tolerance, `${check.description}: error=${error}`);
}

function assertRollingConstraint(result, request, check) {
  const event = result.events[check.event_index];
  const normal = request.surface.normal_unit;
  const arm = normal.map((value) => -request.ball_radius_m * value);
  const spinVelocity = cross(event.angular_velocity_after_rad_s, arm);
  const contact = event.velocity_after_m_s.map(
    (value, index) =>
      value + spinVelocity[index] - request.surface.surface_velocity_m_s[index],
  );
  const normalSpeed = dot(contact, normal);
  const tangent = contact.map(
    (value, index) => value - normalSpeed * normal[index],
  );
  assert.ok(Math.sqrt(dot(tangent, tangent)) <= check.absolute_tolerance);
}

function impactEnergy(event, request, suffix) {
  const velocity = event[`velocity_${suffix}_m_s`];
  const spin = event[`angular_velocity_${suffix}_rad_s`];
  const mass = request.ball_mass_kg;
  const inertia =
    request.rotational_inertia_factor * mass * request.ball_radius_m ** 2;
  return 0.5 * mass * dot(velocity, velocity) + 0.5 * inertia * dot(spin, spin);
}

function assertCheck(result, request, check) {
  if (check.kind === "value_equal") {
    assert.deepEqual(pointer(result, check.path), check.expected);
  } else if (check.kind === "scalar_close") {
    assertClose(pointer(result, check.path), check.expected, check);
  } else if (check.kind === "vector_close") {
    const actual = pointer(result, check.path);
    assert.equal(actual.length, check.expected.length);
    actual.forEach((value, index) =>
      assertClose(value, check.expected[index], check),
    );
  } else if (check.kind === "terminal_vector_close") {
    const actual = result.trajectory.at(-1)[check.field];
    assert.equal(actual.length, check.expected.length);
    actual.forEach((value, index) =>
      assertClose(value, check.expected[index], check),
    );
  } else if (check.kind === "event_types_equal") {
    assert.deepEqual(
      result.events.map((event) => event.event_type),
      check.expected,
    );
  } else if (check.kind === "restitution_ratio") {
    const event = result.events[check.event_index];
    const normal = request.surface.normal_unit;
    const before = dot(event.velocity_before_m_s, normal);
    const after = dot(event.velocity_after_m_s, normal);
    assertClose(after / -before, check.expected, check);
  } else if (check.kind === "rolling_constraint") {
    assertRollingConstraint(result, request, check);
  } else if (check.kind === "impact_energy_nonincrease") {
    const event = result.events[check.event_index];
    assert.ok(
      impactEnergy(event, request, "after") <=
        impactEnergy(event, request, "before") + check.absolute_tolerance_j,
    );
  } else {
    throw new Error(`unsupported conformance check: ${check.kind}`);
  }
}

function loadConformanceCases() {
  const corpus = parseStrictJson(fs.readFileSync(corpusPath, "utf8"));
  assert.equal(corpus.schema_version, "ground-reference-conformance/v1");
  const template = parseStrictJson(
    fs.readFileSync(resolveTemplatePath(corpus.template_fixture), "utf8"),
  );
  const identifiers = corpus.cases.map((entry) => entry.case_id);
  assert.equal(new Set(identifiers).size, identifiers.length);
  return { template, cases: corpus.cases };
}

function materializeCase(template, testCase) {
  const request = structuredClone(template.request);
  applyOverrides(request, testCase.request_overrides);
  return {
    request,
    execution: {
      ...structuredClone(template.execution),
      schema_version: template.execution_schema_version,
    },
  };
}

function assertConformanceCase(result, request, testCase) {
  testCase.checks.forEach((check) => assertCheck(result, request, check));
}

module.exports = {
  applyOverrides,
  assertConformanceCase,
  loadConformanceCases,
  materializeCase,
  parseStrictJson,
  resolveTemplatePath,
};
