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
  float high_limit;
  float low_limit;
};

struct ConfigStruct {
  int magic;  // Signature validation (e.g., 0xDCS1)
  int input_routing[SignalBroker::kNumInputs];
  int output_routing[SignalBroker::kNumOutputs];
  PIDConfigData pids[4];
  InterlockConfigData interlocks[SignalBroker::kNumTags];
};

class StorageManager {
 public:
  static const int kMagic = 0xDCS1;

  StorageManager();

  // Save active configuration to non-volatile storage.
  // Precondition: pids contains 4 controllers, interlock_high and interlock_low contain 32 limits
  bool Save(const SignalBroker& broker,
            const PIDController* pids,
            const float* interlock_high,
            const float* interlock_low);

  // Load configuration from non-volatile storage.
  // Returns true if a valid config was loaded, false if no config existed.
  bool Load(SignalBroker& broker,
            PIDController* pids,
            float* interlock_high,
            float* interlock_low);

  // Clear stored configuration.
  void Clear();
};

#endif  // P1AM_CONTROL_SYSTEM_FIRMWARE_STORAGE_MANAGER_H_
