#include <iostream>
#include <cassert>
#include <cmath>
#include <limits>

// Every check in this suite is an assert(). Building with NDEBUG would compile
// them all away and leave a binary that exits 0 without testing anything --
// exactly the silent-pass failure mode this suite exists to prevent.
#ifdef NDEBUG
#error "test_dcs must be built with assertions enabled (do not define NDEBUG)"
#endif

#include "MockHardware.h"
#include "SignalBroker.h"
#include "PIDController.h"
#include "SafetyInterlock.h"
#include "StorageManager.h"
#include "CommsWatchdog.h"

// Helper to check approximate equality for floats
bool FloatEquals(float a, float b, float epsilon = 0.001f) {
  return std::fabs(a - b) < epsilon;
}

void TestSignalBroker() {
  std::cout << "Running TestSignalBroker..." << std::endl;
  MockHardware hw;
  SignalBroker broker;

  // Initial state checks
  assert(FloatEquals(broker.GetTag(0), 0.0f));

  // Map Thermocouple 0 to Tag 2
  broker.SetInputRouting(0, 2);
  assert(broker.GetInputRouting(0) == 2);

  // Map Analog Input 0 to Tag 5
  broker.SetInputRouting(4, 5);
  assert(broker.GetInputRouting(4) == 5);

  // Set physical values in mock hardware. The expected percentage is derived
  // from the firmware's own full-scale constant rather than hardcoded, so a
  // change to the range updates this expectation instead of silently breaking
  // it (this assertion was stale at 35.0% from an earlier 1000 C full scale).
  const float kTempC = 350.0f;
  const float kExpectedPct =
      kTempC * (100.0f / SignalBroker::kThermocoupleFullScaleC);
  hw.SetThermocouple(0, kTempC);
  hw.SetAnalogInput(0, 62.5f);  // Scaled raw

  broker.ReadHardwareInputs(hw);

  // Assert tag values were updated and scaled
  assert(FloatEquals(broker.GetTag(2), kExpectedPct));
  assert(FloatEquals(broker.GetTag(5), 62.5f));

  // Output routing
  broker.SetOutputRouting(0, 10);  // AO0 driven by Tag 10
  assert(broker.GetOutputRouting(0) == 10);

  broker.SetTag(10, 75.8f);
  broker.WriteHardwareOutputs(hw);

  assert(FloatEquals(hw.GetAnalogOutput(0), 75.8f));
  std::cout << "TestSignalBroker PASSED!" << std::endl;
}

// Issue #4032: the broker used to clamp every tag into [0, 100] percent of
// span. A high limit entered above that ceiling (e.g. an operator typing a
// 900 degC limit straight into the percent-domain register) could never be
// exceeded -- the clamp capped the reading below it and the trip silently
// never fired. The clamp is gone: finite values pass through SetTag/GetTag
// unchanged, and only the physical AO write saturates at the DAC's [0, 100].
void TestEngineeringUnitTagsAndLimits() {
  std::cout << "Running TestEngineeringUnitTagsAndLimits..." << std::endl;
  MockHardware hw;
  SignalBroker broker;
  SafetyInterlock interlock;

  // (a) A high limit above the old clamp ceiling CAN trip when exceeded.
  //     Before the fix GetTag capped the reading at 100.0, so val > 101 was
  //     never true and the interlock was silently disabled.
  interlock.SetHighLimit(5, 101.0f);
  broker.SetTag(5, 101.5f);
  interlock.Evaluate(broker, hw);
  assert(interlock.IsTripped());
  assert(interlock.GetTripTagId() == 5);
  interlock.Reset();

  // (b) A genuine reading above 100 % of span is not truncated. 1470 degC on
  //     the type-K full scale is 105 % -- a runaway must be visible as such,
  //     not flattened onto a valid 100 % (top-of-range).
  broker.SetTag(6, 140.0f);
  assert(FloatEquals(broker.GetTag(6), 140.0f));
  broker.SetTag(7, -10.0f);  // -140 degC: sub-zero reads as negative percent
  assert(FloatEquals(broker.GetTag(7), -10.0f));

  hw.SetThermocouple(2, 1470.0f);
  broker.SetInputRouting(2, 8);
  broker.ReadHardwareInputs(hw);
  const float kRunawayPct = 1470.0f * (100.0f / SignalBroker::kThermocoupleFullScaleC);
  assert(FloatEquals(broker.GetTag(8), kRunawayPct));
  assert(broker.GetTag(8) > 100.0f);

  // The DAC contract is unchanged: a routed over-range tag drives the analog
  // output at full scale, while the tag itself still reads back untruncated.
  broker.SetOutputRouting(0, 8);
  broker.WriteHardwareOutputs(hw);
  assert(FloatEquals(hw.GetAnalogOutput(0), 100.0f));
  assert(FloatEquals(broker.GetTag(8), kRunawayPct));
  broker.SetOutputRouting(0, SignalBroker::kUnmappedTag);

  // (d) Non-finite stays bad-quality: never coerced, never stored as 0.0.
  const float nan = std::numeric_limits<float>::quiet_NaN();
  broker.SetTag(9, nan);
  assert(std::isnan(broker.GetTag(9)));
  assert(!broker.IsTagValid(9));

  std::cout << "TestEngineeringUnitTagsAndLimits PASSED!" << std::endl;
}


void TestPIDController() {
  std::cout << "Running TestPIDController..." << std::endl;
  SignalBroker broker;
  PIDController pid;

  pid.SetPvTagId(3);
  pid.SetCvTagId(4);
  pid.SetSetpoint(50.0f);
  pid.SetKp(1.5f);
  pid.SetKi(0.5f);
  pid.SetKd(0.2f);

  // PV starts at 40.0, error = 10.0
  broker.SetTag(3, 40.0f);

  // 1. Proportional step only
  // P-term = 1.5 * 10.0 = 15.0
  // I-term = 0.5 * (10.0 * 0.1s) = 0.5
  // D-term = 0.2 * ((10.0 - 10.0) / 0.1s) = 0.0
  // Output = 15.5
  pid.Compute(broker, 0.1f);
  assert(FloatEquals(broker.GetTag(4), 15.5f));

  // 2. Compute again with same PV
  // I-term += 0.5 * (10.0 * 0.1s) = 1.0 total
  // Output = 15.0 (P) + 1.0 (I) + 0.0 (D) = 16.0
  pid.Compute(broker, 0.1f);
  assert(FloatEquals(broker.GetTag(4), 16.0f));

  // Test anti-windup clamping
  // Let's run with a large error for many iterations
  pid.SetKi(5.0f);
  for (int i = 0; i < 100; ++i) {
    pid.Compute(broker, 0.1f);
  }
  // Max output must be clamped to 100.0f
  assert(FloatEquals(broker.GetTag(4), 100.0f));

  std::cout << "TestPIDController PASSED!" << std::endl;
}

void TestSafetyInterlock() {
  std::cout << "Running TestSafetyInterlock..." << std::endl;
  MockHardware hw;
  SignalBroker broker;
  SafetyInterlock interlock;

  // Map Tag 10 to AO0
  broker.SetOutputRouting(0, 10);
  broker.SetTag(10, 80.0f);

  // Set limits for Tag 5
  interlock.SetHighLimit(5, 75.0f);
  interlock.SetLowLimit(5, 10.0f);

  // Tag 5 is 50.0 (OK)
  broker.SetTag(5, 50.0f);
  interlock.Evaluate(broker, hw);

  assert(!interlock.IsTripped());
  assert(!hw.GetInhibitActive());
  assert(FloatEquals(hw.GetAnalogOutput(0), 80.0f));

  // Trigger high limit trip: Set Tag 5 to 80.0%
  broker.SetTag(5, 80.0f);
  interlock.Evaluate(broker, hw);

  assert(interlock.IsTripped());
  assert(hw.GetInhibitActive());
  // Tag 10 output must be forced to 0
  assert(FloatEquals(broker.GetTag(10), 0.0f));
  // Hardware output must be forced to 0
  assert(FloatEquals(hw.GetAnalogOutput(0), 0.0f));

  std::cout << "TestSafetyInterlock PASSED!" << std::endl;
}

void TestStorageManager() {
  std::cout << "Running TestStorageManager..." << std::endl;
  StorageManager storage;
  SignalBroker broker;
  PIDController pids[4];

  // All four interlock tiers are persisted (kMagic was bumped to 0xDC52 when
  // InterlockConfigData grew from 2 limits to 4). Round-trip every tier so a
  // future struct change cannot silently drop lolo/hihi again.
  float lolo_limits[SignalBroker::kNumTags];
  float low_limits[SignalBroker::kNumTags];
  float high_limits[SignalBroker::kNumTags];
  float hihi_limits[SignalBroker::kNumTags];

  // Initialize
  broker.Reset();
  for (int i = 0; i < SignalBroker::kNumTags; ++i) {
    lolo_limits[i] = -10.0f;
    low_limits[i] = 0.0f;
    high_limits[i] = 100.0f;
    hihi_limits[i] = 110.0f;
  }

  // Setup routing config
  broker.SetInputRouting(0, 5);
  broker.SetOutputRouting(0, 6);

  // Setup one PID
  pids[0].SetPvTagId(5);
  pids[0].SetCvTagId(6);
  pids[0].SetSetpoint(25.0f);
  pids[0].SetKp(2.0f);
  pids[0].SetKi(0.5f);
  pids[0].SetKd(0.1f);

  // Setup specific limits across all four tiers
  lolo_limits[5] = 3.25f;
  low_limits[5] = 12.5f;
  high_limits[5] = 85.5f;
  hihi_limits[5] = 97.75f;

  // Save
  bool save_ok = storage.Save(broker, pids, lolo_limits, low_limits, high_limits,
                              hihi_limits);
  assert(save_ok);

  // Mess up original state
  broker.Reset();
  for (int i = 0; i < 4; ++i) {
    pids[i].Reset();
  }
  for (int i = 0; i < SignalBroker::kNumTags; ++i) {
    lolo_limits[i] = 0.0f;
    low_limits[i] = 0.0f;
    high_limits[i] = 0.0f;
    hihi_limits[i] = 0.0f;
  }

  // Load
  bool load_ok = storage.Load(broker, pids, lolo_limits, low_limits, high_limits,
                              hihi_limits);
  assert(load_ok);

  // Verify
  assert(broker.GetInputRouting(0) == 5);
  assert(broker.GetOutputRouting(0) == 6);
  assert(pids[0].GetPvTagId() == 5);
  assert(pids[0].GetCvTagId() == 6);
  assert(FloatEquals(pids[0].GetSetpoint(), 25.0f));
  assert(FloatEquals(pids[0].GetKp(), 2.0f));
  assert(FloatEquals(pids[0].GetKi(), 0.5f));
  assert(FloatEquals(pids[0].GetKd(), 0.1f));
  assert(FloatEquals(lolo_limits[5], 3.25f));
  assert(FloatEquals(low_limits[5], 12.5f));
  assert(FloatEquals(high_limits[5], 85.5f));
  assert(FloatEquals(hihi_limits[5], 97.75f));
  // An untouched tag keeps the value it was saved with, not a zero fill.
  assert(FloatEquals(hihi_limits[7], 110.0f));

  // Cleanup
  storage.Clear();
  std::cout << "TestStorageManager PASSED!" << std::endl;
}

void TestSoftFailRuntimeContracts() {
  std::cout << "Running TestSoftFailRuntimeContracts..." << std::endl;
  MockHardware hw;
  SignalBroker broker;
  PIDController pid;
  SafetyInterlock interlock;
  const float nan = std::numeric_limits<float>::quiet_NaN();

  // Invalid SCADA/flash routes must fail closed instead of aborting firmware.
  broker.SetInputRouting(0, 999);
  broker.SetOutputRouting(0, -12);
  assert(broker.GetInputRouting(0) == SignalBroker::kUnmappedTag);
  assert(broker.GetOutputRouting(0) == SignalBroker::kUnmappedTag);
  assert(broker.GetInputRouting(-1) == SignalBroker::kUnmappedTag);
  assert(broker.GetOutputRouting(99) == SignalBroker::kUnmappedTag);

  // Bad hardware samples are kept as NaN (bad quality), never coerced to a
  // 0.0 that reads as a valid cold measurement (#4032). Finite values are
  // stored unchanged: the broker does not clamp (#4032), so a sub-zero degC
  // reading is a negative percent and an over-range runaway reads above 100.
  broker.SetTag(1, nan);
  broker.SetTag(2, -10.0f);
  broker.SetTag(3, 140.0f);
  broker.SetTag(999, 50.0f);
  assert(std::isnan(broker.GetTag(1)));
  assert(!broker.IsTagValid(1));
  assert(FloatEquals(broker.GetTag(2), -10.0f));
  assert(broker.IsTagValid(2));
  assert(FloatEquals(broker.GetTag(3), 140.0f));
  assert(FloatEquals(broker.GetTag(999), 0.0f));
  assert(!broker.IsTagValid(999));

  hw.SetThermocouple(0, nan);
  broker.SetInputRouting(0, 4);
  broker.ReadHardwareInputs(hw);
  assert(std::isnan(broker.GetTag(4)));

  // A bad-quality source tag never reaches the DAC: the output is driven to
  // the safe 0.0 % (MockHardware asserts the DAC contract itself).
  broker.SetOutputRouting(0, 4);
  broker.SetTag(4, nan);
  broker.WriteHardwareOutputs(hw);
  assert(FloatEquals(hw.GetAnalogOutput(0), 0.0f));
  broker.SetOutputRouting(0, SignalBroker::kUnmappedTag);

  // Invalid PID configuration or scan timing skips/normalizes safely.
  pid.SetPvTagId(999);
  pid.SetCvTagId(-8);
  assert(pid.GetPvTagId() == SignalBroker::kUnmappedTag);
  assert(pid.GetCvTagId() == SignalBroker::kUnmappedTag);
  pid.SetPvTagId(5);
  pid.SetCvTagId(6);
  pid.SetSetpoint(nan);
  pid.SetKp(nan);
  pid.SetKi(nan);
  pid.SetKd(nan);
  assert(FloatEquals(pid.GetSetpoint(), 0.0f));
  assert(FloatEquals(pid.GetKp(), 0.0f));
  assert(FloatEquals(pid.GetKi(), 0.0f));
  assert(FloatEquals(pid.GetKd(), 0.0f));
  broker.SetTag(6, 42.0f);
  pid.Compute(broker, 0.0f);
  assert(FloatEquals(broker.GetTag(6), 42.0f));
  pid.Compute(broker, nan);
  assert(FloatEquals(broker.GetTag(6), 42.0f));

  // Invalid interlock limits should not force the controller into an abort path.
  interlock.SetHighLimit(999, 1.0f);
  interlock.SetLowLimit(-1, 1.0f);
  interlock.SetHighLimit(0, nan);
  interlock.SetLowLimit(0, nan);
  assert(FloatEquals(interlock.GetHighLimit(0), SafetyInterlock::kDisabledHighLimit));
  assert(FloatEquals(interlock.GetLowLimit(0), SafetyInterlock::kDisabledLowLimit));
  assert(!interlock.IsInterlocked(0));

  // A NaN PV must de-energize the loop's CV, not drive it with NaN math.
  pid.SetPvTagId(4);
  pid.SetCvTagId(6);
  pid.SetSetpoint(50.0f);
  pid.SetKp(1.0f);
  broker.SetTag(4, nan);
  broker.SetTag(6, 42.0f);
  pid.Compute(broker, 0.1f);
  assert(FloatEquals(broker.GetTag(6), 0.0f));

  std::cout << "TestSoftFailRuntimeContracts PASSED!" << std::endl;
}

void TestInterlockDefaultsDoNotTrip() {
  std::cout << "Running TestInterlockDefaultsDoNotTrip..." << std::endl;
  MockHardware hw;
  SignalBroker broker;
  SafetyInterlock interlock;

  // Power-on state: every tag at 0.0 %, every limit at the disabled sentinel.
  // This is the "simulated boot with default config" of issue #4001/#4911:
  // unrouted tags sitting at 0.0 must not be able to trip the plant.
  broker.SetOutputRouting(0, 10);
  broker.SetTag(10, 80.0f);
  for (int i = 0; i < SignalBroker::kNumTags; ++i) {
    assert(!interlock.IsInterlocked(i));
  }
  interlock.Evaluate(broker, hw);
  assert(!interlock.IsTripped());
  assert(interlock.GetTripTagId() == SafetyInterlock::kNoTripTag);
  assert(FloatEquals(hw.GetAnalogOutput(0), 80.0f));

  // A room-temperature type-K channel (25 C -> 1.8 %) with only a high-side
  // limit configured is inside its band.
  interlock.SetHighLimit(0, 95.0f);
  assert(interlock.IsInterlocked(0));
  broker.SetTag(0, 25.0f * (100.0f / SignalBroker::kThermocoupleFullScaleC));
  interlock.Evaluate(broker, hw);
  assert(!interlock.IsTripped());

  // NaN on a NON-interlocked tag is ignored ...
  const float nan = std::numeric_limits<float>::quiet_NaN();
  broker.SetTag(7, nan);
  interlock.Evaluate(broker, hw);
  assert(!interlock.IsTripped());

  // ... but NaN on an interlocked tag is a sensor fault and trips (fail safe),
  // and the trip names the tag.
  broker.SetTag(0, nan);
  interlock.Evaluate(broker, hw);
  assert(interlock.IsTripped());
  assert(interlock.GetTripTagId() == 0);
  assert(hw.GetInhibitActive());
  assert(FloatEquals(hw.GetAnalogOutput(0), 0.0f));

  std::cout << "TestInterlockDefaultsDoNotTrip PASSED!" << std::endl;
}

void TestInterlockResetLatch() {
  std::cout << "Running TestInterlockResetLatch..." << std::endl;
  MockHardware hw;
  SignalBroker broker;
  SafetyInterlock interlock;

  broker.SetOutputRouting(0, 10);
  interlock.SetLowLimit(5, 10.0f);
  interlock.SetHighLimit(5, 75.0f);

  // Trip on a high violation of tag 5.
  broker.SetTag(10, 80.0f);
  broker.SetTag(5, 80.0f);
  interlock.Evaluate(broker, hw);
  assert(interlock.IsTripped());
  assert(interlock.GetTripTagId() == 5);
  assert(FloatEquals(hw.GetAnalogOutput(0), 0.0f));
  assert(hw.GetInhibitActive());

  // 1. Reset requested while the cause is still present: REFUSED. The latch
  //    and the forced-safe outputs are untouched.
  assert(!interlock.ClearTrip(broker));
  assert(interlock.IsTripped());
  assert(interlock.GetTripTagId() == 5);
  broker.SetTag(10, 80.0f);
  interlock.Evaluate(broker, hw);
  assert(FloatEquals(hw.GetAnalogOutput(0), 0.0f));
  assert(hw.GetInhibitActive());

  // 2. Cause clears WITHOUT a reset: the trip stays latched. A plant that
  //    un-trips itself the moment the reading dips back into band is exactly
  //    the behaviour an interlock exists to prevent.
  broker.SetTag(5, 50.0f);
  broker.SetTag(10, 80.0f);
  interlock.Evaluate(broker, hw);
  assert(interlock.IsTripped());
  assert(FloatEquals(hw.GetAnalogOutput(0), 0.0f));
  assert(hw.GetInhibitActive());

  // 3. Cause clear AND reset asserted: the latch clears and the next scan
  //    restores normal output routing.
  assert(interlock.ClearTrip(broker));
  assert(!interlock.IsTripped());
  assert(interlock.GetTripTagId() == SafetyInterlock::kNoTripTag);
  broker.SetTag(10, 80.0f);
  interlock.Evaluate(broker, hw);
  assert(!interlock.IsTripped());
  assert(FloatEquals(hw.GetAnalogOutput(0), 80.0f));
  assert(!hw.GetInhibitActive());

  // 4. Reset with nothing latched is a harmless no-op that reports "clear".
  assert(interlock.ClearTrip(broker));
  assert(!interlock.IsTripped());

  // 5. The interlock re-trips immediately if the violation returns.
  broker.SetTag(5, 3.0f);  // below the low limit
  interlock.Evaluate(broker, hw);
  assert(interlock.IsTripped());
  assert(interlock.GetTripTagId() == 5);

  std::cout << "TestInterlockResetLatch PASSED!" << std::endl;
}


void TestPidResetsIntegralOnSetpointZeroed() {
  std::cout << "Running TestPidResetsIntegralOnSetpointZeroed..." << std::endl;
  SignalBroker broker;
  PIDController pid;

  pid.SetPvTagId(3);
  pid.SetCvTagId(4);
  pid.SetSetpoint(100.0f);
  pid.SetKp(0.0f);
  pid.SetKi(1.0f);  // integral-only, so the wind-up is unambiguous
  pid.SetKd(0.0f);

  // Wind the integral to its clamp against a large sustained error.
  broker.SetTag(3, 0.0f);
  for (int i = 0; i < 200; ++i) {
    pid.Compute(broker, 0.1f);
  }
  assert(FloatEquals(broker.GetTag(4), 100.0f));

  // Commanding the setpoint to zero must drop the output on the NEXT scan,
  // not decay towards it over many seconds. This is the issue #4002 condition:
  // an E-stop's only effect reaching the plant is zeroing these setpoints.
  pid.SetSetpoint(0.0f);
  pid.Compute(broker, 0.1f);
  assert(FloatEquals(broker.GetTag(4), 0.0f));

  std::cout << "TestPidResetsIntegralOnSetpointZeroed PASSED!" << std::endl;
}

void TestPidKeepsIntegralAcrossNonZeroSetpointChange() {
  std::cout << "Running TestPidKeepsIntegralAcrossNonZeroSetpointChange..."
            << std::endl;
  SignalBroker broker;
  PIDController pid;

  pid.SetPvTagId(3);
  pid.SetCvTagId(4);
  pid.SetSetpoint(100.0f);
  pid.SetKp(0.0f);
  pid.SetKi(1.0f);
  pid.SetKd(0.0f);

  broker.SetTag(3, 0.0f);
  for (int i = 0; i < 200; ++i) {
    pid.Compute(broker, 0.1f);
  }
  assert(FloatEquals(broker.GetTag(4), 100.0f));

  // A change between two NON-ZERO setpoints must preserve the accumulated
  // integral. SyncModbusToDCS calls SetSetpoint on every scan whenever the host
  // register differs, so resetting on any change would clear the integrator
  // once per scan for the whole of a host-driven ramp -- leaving the loop
  // running P+D only, with a steady-state offset it can never close and no
  // indication that integral action had been silently disabled.
  pid.SetSetpoint(50.0f);
  pid.Compute(broker, 0.1f);

  // Error is still positive (PV = 0, SP = 50), so an intact integral keeps the
  // output at its clamp. A reset would have dropped it to ki*error*dt = 5.0.
  assert(FloatEquals(broker.GetTag(4), 100.0f));

  std::cout << "TestPidKeepsIntegralAcrossNonZeroSetpointChange PASSED!"
            << std::endl;
}

void TestPidDoesNotIntegrateWhileTripped() {
  std::cout << "Running TestPidDoesNotIntegrateWhileTripped..." << std::endl;
  SignalBroker broker;
  PIDController pid;

  pid.SetPvTagId(3);
  pid.SetCvTagId(4);
  pid.SetSetpoint(100.0f);
  pid.SetKp(0.0f);
  pid.SetKi(1.0f);
  pid.SetKd(0.0f);
  broker.SetTag(3, 0.0f);

  pid.Hold();  // interlock tripped: freeze the loop and shed accumulated state
  for (int i = 0; i < 200; ++i) {
    pid.Compute(broker, 0.1f);
  }
  assert(FloatEquals(broker.GetTag(4), 0.0f));

  // Releasing the hold starts from a clean integral, so the first scan after
  // recovery contributes one step -- not 200 scans' worth.
  pid.Release();
  pid.Compute(broker, 0.1f);
  assert(FloatEquals(broker.GetTag(4), 10.0f));  // ki * error * dt = 1*100*0.1

  std::cout << "TestPidDoesNotIntegrateWhileTripped PASSED!" << std::endl;
}

void TestCommsWatchdog() {
  std::cout << "Running TestCommsWatchdog..." << std::endl;
  CommsWatchdog watchdog(2000);  // 2 s against a nominal 100 ms scan

  watchdog.Begin(1000);
  assert(!watchdog.IsExpired(1000));
  assert(!watchdog.IsExpired(2999));

  // Exactly at the timeout counts as expired -- fail safe, not fail late.
  assert(watchdog.IsExpired(3000));
  assert(watchdog.IsExpired(50000));

  // Traffic re-arms it.
  watchdog.RecordActivity(50000);
  assert(!watchdog.IsExpired(51999));
  assert(watchdog.IsExpired(52000));

  // millis() wraps at ~49.7 days. Unsigned subtraction must carry the
  // watchdog across the wrap rather than disarming it for another 49 days.
  const unsigned long kNearWrap = 0xFFFFFF00UL;
  watchdog.RecordActivity(kNearWrap);
  assert(!watchdog.IsExpired(kNearWrap + 1999UL));
  assert(watchdog.IsExpired(kNearWrap + 2000UL));  // wraps past zero

  std::cout << "TestCommsWatchdog PASSED!" << std::endl;
}

int main() {
  std::cout << "=== DCS CORE FIRMWARE TDD TEST RUNNER ===" << std::endl;
  TestSignalBroker();
  TestEngineeringUnitTagsAndLimits();
  TestPIDController();
  TestSafetyInterlock();
  TestStorageManager();
  TestSoftFailRuntimeContracts();
  TestInterlockDefaultsDoNotTrip();
  TestInterlockResetLatch();
  TestPidResetsIntegralOnSetpointZeroed();
  TestPidKeepsIntegralAcrossNonZeroSetpointChange();
  TestPidDoesNotIntegrateWhileTripped();
  TestCommsWatchdog();
  std::cout << "All C++ Core Firmware Tests Passed Successfully!" << std::endl;
  return 0;
}
