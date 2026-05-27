#ifndef P1AM_CONTROL_SYSTEM_FIRMWARE_STORAGE_MANAGER_H_
#define P1AM_CONTROL_SYSTEM_FIRMWARE_STORAGE_MANAGER_H_

#include "SignalBroker.h"
#include "PIDController.h"

struct PIDConfigData {
  int pv_tag_id;
  int cv_tag_id;
  float setpoint;
  float kp;
  float ki;
  float kd;
};

struct InterlockConfigData {
  float lolo_limit;
  float low_limit;
  float high_limit;
  float hihi_limit;
};

struct ConfigStruct {
  int magic;  // Signature validation (kMagic).
  int input_routing[SignalBroker::kNumInputs];
  int output_routing[SignalBroker::kNumOutputs];
  PIDConfigData pids[4];
  InterlockConfigData interlocks[SignalBroker::kNumTags];
};

class StorageManager {
 public:
  // Bumped from 0xDC51 when InterlockConfigData grew from 2 to 4 limits.
  // Configs written by older firmware are silently rejected and the unit
  // boots with defaults instead of garbling the new wider struct.
  static const int kMagic = 0xDC52;

  StorageManager();

  // Save active configuration to non-volatile storage.
  // Precondition: pids contains 4 controllers; each interlock_* buffer
  // contains kNumTags floats laid out as lolo / low / high / hihi.
  bool Save(const SignalBroker& broker,
            const PIDController* pids,
            const float* interlock_lolo,
            const float* interlock_low,
            const float* interlock_high,
            const float* interlock_hihi);

  // Load configuration from non-volatile storage.
  // Returns true if a valid config was loaded, false if no config existed.
  bool Load(SignalBroker& broker,
            PIDController* pids,
            float* interlock_lolo,
            float* interlock_low,
            float* interlock_high,
            float* interlock_hihi);

  // Clear stored configuration.
  void Clear();
};

#endif  // P1AM_CONTROL_SYSTEM_FIRMWARE_STORAGE_MANAGER_H_
