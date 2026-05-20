#include "StorageManager.h"
#include <cstring>

#ifdef ARDUINO
#include <Arduino.h>
#include <FlashStorage.h>
// Setup flash storage for ConfigStruct
FlashStorage(dcs_flash_storage, ConfigStruct);
#else
#include <cstdio>
#include <fstream>
const char* kConfigFileName = "config.bin";
#endif

StorageManager::StorageManager() {}

bool StorageManager::Save(const SignalBroker& broker,
                          const PIDController* pids,
                          const float* interlock_high,
                          const float* interlock_low) {
  ConfigStruct config;
  config.magic = kMagic;

  // 1. Save input and output routing
  for (int i = 0; i < SignalBroker::kNumInputs; ++i) {
    config.input_routing[i] = broker.GetInputRouting(i);
  }
  for (int i = 0; i < SignalBroker::kNumOutputs; ++i) {
    config.output_routing[i] = broker.GetOutputRouting(i);
  }

  // 2. Save PID controller configs
  for (int i = 0; i < 4; ++i) {
    config.pids[i].pv_tag_id = pids[i].GetPvTagId();
    config.pids[i].cv_tag_id = pids[i].GetCvTagId();
    config.pids[i].setpoint = pids[i].GetSetpoint();
    config.pids[i].kp = pids[i].GetKp();
    config.pids[i].ki = pids[i].GetKi();
    config.pids[i].kd = pids[i].GetKd();
  }

  // 3. Save Interlock limits
  for (int i = 0; i < SignalBroker::kNumTags; ++i) {
    config.interlocks[i].high_limit = interlock_high[i];
    config.interlocks[i].low_limit = interlock_low[i];
  }

#ifdef ARDUINO
  dcs_flash_storage.write(config);
  return true;
#else
  std::ofstream out(kConfigFileName, std::ios::binary);
  if (!out) {
    return false;
  }
  out.write(reinterpret_cast<const char*>(&config), sizeof(ConfigStruct));
  return out.good();
#endif
}

bool StorageManager::Load(SignalBroker& broker,
                          PIDController* pids,
                          float* interlock_high,
                          float* interlock_low) {
  ConfigStruct config;

#ifdef ARDUINO
  config = dcs_flash_storage.read();
#else
  std::ifstream in(kConfigFileName, std::ios::binary);
  if (!in) {
    return false;
  }
  in.read(reinterpret_cast<char*>(&config), sizeof(ConfigStruct));
  if (!in.good()) {
    return false;
  }
#endif

  if (config.magic != kMagic) {
    return false;  // No valid config stored
  }

  // 1. Load routing matrices
  broker.Reset();
  for (int i = 0; i < SignalBroker::kNumInputs; ++i) {
    broker.SetInputRouting(i, config.input_routing[i]);
  }
  for (int i = 0; i < SignalBroker::kNumOutputs; ++i) {
    broker.SetOutputRouting(i, config.output_routing[i]);
  }

  // 2. Load PID configurations
  for (int i = 0; i < 4; ++i) {
    pids[i].Reset();
    pids[i].SetPvTagId(config.pids[i].pv_tag_id);
    pids[i].SetCvTagId(config.pids[i].cv_tag_id);
    pids[i].SetSetpoint(config.pids[i].setpoint);
    pids[i].SetKp(config.pids[i].kp);
    pids[i].SetKi(config.pids[i].ki);
    pids[i].SetKd(config.pids[i].kd);
  }

  // 3. Load interlocks
  for (int i = 0; i < SignalBroker::kNumTags; ++i) {
    interlock_high[i] = config.interlocks[i].high_limit;
    interlock_low[i] = config.interlocks[i].low_limit;
  }

  return true;
}

void StorageManager::Clear() {
#ifdef ARDUINO
  ConfigStruct empty_config;
  empty_config.magic = 0;
  dcs_flash_storage.write(empty_config);
#else
  std::remove(kConfigFileName);
#endif
}
