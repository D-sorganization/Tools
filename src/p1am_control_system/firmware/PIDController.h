#ifndef P1AM_CONTROL_SYSTEM_FIRMWARE_PID_CONTROLLER_H_
#define P1AM_CONTROL_SYSTEM_FIRMWARE_PID_CONTROLLER_H_

#include "SignalBroker.h"

class PIDController {
 public:
  PIDController();

  // Reset PID state (integral, last error, etc.)
  void Reset();

  // Compute PID output and write it to CV Tag in the SignalBroker.
  // Precondition: dt > 0.0f
  // No-op while held (see Hold()).
  void Compute(SignalBroker& broker, float dt);

  // Freeze the loop and shed accumulated state. Called while the safety
  // interlock is tripped so a latched plant cannot wind up a controller that
  // would then slam the output the moment the trip clears (issue #4002).
  void Hold();

  // Resume control from a clean integral.
  void Release();

  bool IsHeld() const;

  // Getters and Setters
  int GetPvTagId() const;
  void SetPvTagId(int tag_id);

  int GetCvTagId() const;
  void SetCvTagId(int tag_id);

  float GetSetpoint() const;
  void SetSetpoint(float setpoint);

  float GetKp() const;
  void SetKp(float kp);

  float GetKi() const;
  void SetKi(float ki);

  float GetKd() const;
  void SetKd(float kd);

 private:
  // Clear integral/derivative history without touching tuning or routing.
  void ResetDynamicState();

  int pv_tag_id_;
  int cv_tag_id_;
  float setpoint_;
  float kp_;
  float ki_;
  float kd_;

  float integral_;
  float last_error_;
  bool first_run_;
  bool held_;
};

#endif  // P1AM_CONTROL_SYSTEM_FIRMWARE_PID_CONTROLLER_H_
