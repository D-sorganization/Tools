"use strict";

const crypto = require("node:crypto");
const fs = require("node:fs");
const path = require("node:path");

const toolsCore = require("../pkg/tools_core.js");
const fixturePath = path.resolve(
  __dirname,
  "../../../src/rate_of_closure/web/src/model/__fixtures__/ground_reference_pipeline_golden_v1.json",
);
const fixture = JSON.parse(fs.readFileSync(fixturePath, "utf8"));
const execution = {
  ...fixture.execution,
  schema_version: fixture.execution_schema_version,
};
const requestJson = JSON.stringify(fixture.request);
const executionJson = JSON.stringify(execution);

function expectRuntimeFailure(request, executionControl, expected) {
  let caught;
  try {
    toolsCore.runFlightToGroundReferenceV1(
      JSON.stringify(request),
      JSON.stringify(executionControl),
      undefined,
    );
  } catch (error) {
    caught = error;
  }
  if (caught === undefined) {
    throw new Error(`WASM accepted ${expected.native_reason}`);
  }
  const payload = JSON.parse(String(caught));
  for (const [key, value] of Object.entries(expected)) {
    if (payload[key] !== value) {
      throw caught;
    }
  }
}

const actual = toolsCore.runFlightToGroundReferenceV1(
  requestJson,
  executionJson,
  undefined,
);
const digest = crypto.createHash("sha256").update(actual).digest("hex");
if (digest !== fixture.result_sha256) {
  throw new Error(`golden digest mismatch: ${digest}`);
}
if (toolsCore.runFlightToGroundReferenceV1(requestJson) !== actual) {
  throw new Error("default WASM execution controls diverged");
}
for (let index = 0; index < 100; index += 1) {
  const repeated = toolsCore.runFlightToGroundReferenceV1(
    requestJson,
    executionJson,
    undefined,
  );
  if (repeated !== actual) {
    throw new Error(`WASM determinism failed at run ${index}`);
  }
}

const immediateCapture = structuredClone(fixture.request);
const rollingSpeed = 1.0;
const rollingSpin = -rollingSpeed / immediateCapture.ball_radius_m;
immediateCapture.request_id = "compiled-wasm-immediate-capture";
for (const stateName of ["last_separated_state", "first_penetrating_state"]) {
  immediateCapture[stateName].velocity_m_s = [rollingSpeed, -0.04, 0];
  immediateCapture[stateName].angular_velocity_rad_s = [0, 0, rollingSpin];
}
const immediateResult = JSON.parse(
  toolsCore.runFlightToGroundReferenceV1(
    JSON.stringify(immediateCapture),
    executionJson,
    undefined,
  ),
);
if (immediateResult.trajectory[0].phase !== "impact") {
  throw new Error("WASM immediate capture lost the first impact point");
}

const pathological = structuredClone(fixture.request);
pathological.output_interval_s = 1e-11;
let outputLimitCallbackCalls = 0;
try {
  toolsCore.runFlightToGroundReferenceV1(
    JSON.stringify(pathological),
    executionJson,
    () => {
      outputLimitCallbackCalls += 1;
      return false;
    },
  );
  throw new Error("WASM accepted an unbounded output schedule");
} catch (error) {
  const payload = JSON.parse(String(error));
  if (
    payload.native_reason !== "output_point_limit" ||
    outputLimitCallbackCalls !== 0
  ) {
    throw error;
  }
}

for (const restitution of [fixture.request.surface.normal_restitution, 0]) {
  const unrepresentable = structuredClone(fixture.request);
  unrepresentable.last_separated_state.time_s = 9_000_000_000_000_000;
  unrepresentable.first_penetrating_state.time_s = 9_000_000_000_000_002;
  unrepresentable.surface.normal_restitution = restitution;
  unrepresentable.max_time_s = 0.1;
  unrepresentable.output_interval_s = 0.001;
  let resolutionCallbackCalls = 0;
  try {
    toolsCore.runFlightToGroundReferenceV1(
      JSON.stringify(unrepresentable),
      executionJson,
      () => {
        resolutionCallbackCalls += 1;
        return false;
      },
    );
    throw new Error("WASM emitted an unrepresentable time grid");
  } catch (error) {
    const payload = JSON.parse(String(error));
    if (
      payload.native_reason !== "time_resolution" ||
      payload.phase !== "bounce" ||
      resolutionCallbackCalls !== 0
    ) {
      throw error;
    }
  }
}

const capped = structuredClone(fixture.request);
capped.max_time_s = 200_001;
capped.output_interval_s = 1;
const untrustedExecution = structuredClone(execution);
untrustedExecution.skid_roll_settings.max_steps = 9_007_199_254_740_991;
let capCallbackCalls = 0;
try {
  toolsCore.runFlightToGroundReferenceV1(
    JSON.stringify(capped),
    JSON.stringify(untrustedExecution),
    () => {
      capCallbackCalls += 1;
      return false;
    },
  );
  throw new Error("WASM accepted an expanded absolute output cap");
} catch (error) {
  const payload = JSON.parse(String(error));
  if (
    payload.native_reason !== "output_point_limit" ||
    payload.phase !== "bounce" ||
    capCallbackCalls !== 0
  ) {
    throw error;
  }
}

const boundedSteps = structuredClone(fixture.request);
boundedSteps.max_time_s = 1;
boundedSteps.output_interval_s = 1;
const oversizedSteps = structuredClone(execution);
oversizedSteps.skid_roll_settings.integration_step_s = 1e-11;
oversizedSteps.skid_roll_settings.max_steps = 1_000_001;
let stepCallbackCalls = 0;
try {
  toolsCore.runFlightToGroundReferenceV1(
    JSON.stringify(boundedSteps),
    JSON.stringify(oversizedSteps),
    () => {
      stepCallbackCalls += 1;
      return false;
    },
  );
  throw new Error("WASM accepted an oversized integration-step budget");
} catch (error) {
  const payload = JSON.parse(String(error));
  if (
    payload.native_reason !== "integration_step_limit" ||
    payload.phase !== "skid_roll" ||
    stepCallbackCalls !== 0
  ) {
    throw error;
  }
}

const oversizedEvents = structuredClone(fixture.request);
oversizedEvents.max_events = 10_001;
expectRuntimeFailure(oversizedEvents, execution, {
  code: "execution_failure",
  phase: "bounce",
  native_reason: "event_count_limit",
});

const bounceOverflow = structuredClone(fixture.request);
bounceOverflow.ball_radius_m = 1_000_000;
Object.assign(bounceOverflow.surface, {
  normal_restitution: 1,
  static_friction: 5,
  kinetic_friction: 5,
});
for (const [stateName, height] of [
  ["last_separated_state", 1_000_001],
  ["first_penetrating_state", 999_999],
]) {
  const state = bounceOverflow[stateName];
  state.position_m[1] = height;
  state.velocity_m_s = [0, -9_000_000_000_000_000, 0];
  state.angular_velocity_rad_s = [0, 0, 9_000_000_000_000_000];
}
expectRuntimeFailure(bounceOverflow, execution, {
  code: "numerical_failure",
  phase: "bounce",
  native_reason: "numeric_range",
});

const surfaceOverflow = structuredClone(fixture.request);
surfaceOverflow.surface.normal_restitution = 0;
for (const stateName of ["last_separated_state", "first_penetrating_state"]) {
  surfaceOverflow[stateName].position_m[0] = 9_000_000_000_000_000;
  surfaceOverflow[stateName].velocity_m_s[0] = 9_000_000_000_000_000;
}
surfaceOverflow.max_time_s = 0.01;
surfaceOverflow.output_interval_s = 0.001;
expectRuntimeFailure(surfaceOverflow, execution, {
  code: "numerical_failure",
  phase: "skid_roll",
  native_reason: "numeric_range",
});

const compositionOverflow = structuredClone(fixture.request);
for (const stateName of ["last_separated_state", "first_penetrating_state"]) {
  compositionOverflow[stateName].position_m[0] = 6_500_000_000_000_000;
  compositionOverflow[stateName].position_m[2] = 6_500_000_000_000_000;
}
expectRuntimeFailure(compositionOverflow, execution, {
  code: "numerical_failure",
  phase: "composition",
  native_reason: "numeric_range",
});

for (const restitution of [fixture.request.surface.normal_restitution, 0]) {
  const representable = structuredClone(fixture.request);
  representable.last_separated_state.time_s = 1_000_000_000_000;
  representable.first_penetrating_state.time_s = 1_000_000_000_002;
  representable.surface.normal_restitution = restitution;
  representable.output_interval_s = 0.00125;
  const result = JSON.parse(
    toolsCore.runFlightToGroundReferenceV1(
      JSON.stringify(representable),
      executionJson,
      undefined,
    ),
  );
  for (let index = 1; index < result.trajectory.length; index += 1) {
    if (
      result.trajectory[index - 1].time_s >= result.trajectory[index].time_s
    ) {
      throw new Error("WASM representable large epoch is not monotonic");
    }
  }
}

const eventLimited = structuredClone(immediateCapture);
eventLimited.max_events = 1;
const eventResult = JSON.parse(
  toolsCore.runFlightToGroundReferenceV1(
    JSON.stringify(eventLimited),
    executionJson,
    undefined,
  ),
);
if (
  eventResult.status !== "partial" ||
  eventResult.termination.reason !== "event_limit" ||
  eventResult.termination.completed !== false ||
  eventResult.events.length !== 1 ||
  eventResult.trajectory.at(-1).time_s !== eventResult.termination.time_s
) {
  throw new Error("WASM immediate event limit is incoherent");
}

try {
  toolsCore.runFlightToGroundReferenceV1(requestJson, executionJson, () => 1);
  throw new Error("WASM accepted a non-boolean cancellation result");
} catch (error) {
  if (String(error) !== "is_cancelled_result") {
    throw error;
  }
}

let cancelled = false;
try {
  toolsCore.runFlightToGroundReferenceV1(
    requestJson,
    executionJson,
    () => true,
  );
} catch (error) {
  const payload = JSON.parse(String(error));
  cancelled = payload.code === "cancelled" && payload.phase === "bounce";
}
if (!cancelled) {
  throw new Error("WASM typed cancellation was not observed");
}

const sentinel = new Error("callback-sentinel");
try {
  toolsCore.runFlightToGroundReferenceV1(requestJson, executionJson, () => {
    throw sentinel;
  });
  throw new Error("WASM callback exception was swallowed");
} catch (error) {
  if (error !== sentinel) {
    throw error;
  }
}
