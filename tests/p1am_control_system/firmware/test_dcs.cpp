#include <iostream>
#include <cassert>
#include <cmath>
#include "MockHardware.h"
#include "SignalBroker.h"
#include "PIDController.h"
#include "SafetyInterlock.h"
#include "StorageManager.h"

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

  // Set physical values in mock hardware
  hw.SetThermocouple(0, 350.0f);  // Should scale to 35.0%
  hw.SetAnalogInput(0, 62.5f);    // Scaled raw

  broker.ReadHardwareInputs(hw);

  // Assert tag values were updated and scaled
  assert(FloatEquals(broker.GetTag(2), 35.0f));
  assert(FloatEquals(broker.GetTag(5), 62.5f));

  // Output routing
  broker.SetOutputRouting(0, 10);  // AO0 driven by Tag 10
  assert(broker.GetOutputRouting(0) == 10);

  broker.SetTag(10, 75.8f);
  broker.WriteHardwareOutputs(hw);

  assert(FloatEquals(hw.GetAnalogOutput(0), 75.8f));
  std::cout << "TestSignalBroker PASSED!" << std::endl;
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

  float high_limits[SignalBroker::kNumTags];
  float low_limits[SignalBroker::kNumTags];

  // Initialize
  broker.Reset();
  for (int i = 0; i < SignalBroker::kNumTags; ++i) {
    high_limits[i] = 100.0f;
    low_limits[i] = 0.0f;
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

  // Setup specific limit
  high_limits[5] = 85.5f;
  low_limits[5] = 12.5f;

  // Save
  bool save_ok = storage.Save(broker, pids, high_limits, low_limits);
  assert(save_ok);

  // Mess up original state
  broker.Reset();
  for (int i = 0; i < 4; ++i) {
    pids[i].Reset();
  }
  for (int i = 0; i < SignalBroker::kNumTags; ++i) {
    high_limits[i] = 0.0f;
    low_limits[i] = 0.0f;
  }

  // Load
  bool load_ok = storage.Load(broker, pids, high_limits, low_limits);
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
  assert(FloatEquals(high_limits[5], 85.5f));
  assert(FloatEquals(low_limits[5], 12.5f));

  // Cleanup
  storage.Clear();
  std::cout << "TestStorageManager PASSED!" << std::endl;
}

int main() {
  std::cout << "=== DCS CORE FIRMWARE TDD TEST RUNNER ===" << std::endl;
  TestSignalBroker();
  TestPIDController();
  TestSafetyInterlock();
  TestStorageManager();
  std::cout << "All C++ Core Firmware Tests Passed Successfully!" << std::endl;
  return 0;
}
