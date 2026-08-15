#include "PIDController.h"
#include <cmath>

namespace {

bool IsValidRoutingTagId(int tag_id) {
  return tag_id == SignalBroker::kUnmappedTag ||
         (tag_id >= 0 && tag_id < SignalBroker::kNumTags);
}

float FiniteOrZero(float value) {
  return std::isfinite(value) ? value : 0.0f;
}

}  // namespace

PIDController::PIDController() {
  Reset();
}

void PIDController::Reset() {
  pv_tag_id_ = SignalBroker::kUnmappedTag;
  cv_tag_id_ = SignalBroker::kUnmappedTag;
  setpoint_ = 0.0f;
  kp_ = 0.0f;
  ki_ = 0.0f;
  kd_ = 0.0f;

  integral_ = 0.0f;
  last_error_ = 0.0f;
  first_run_ = true;
  held_ = false;
}

void PIDController::Hold() {
  held_ = true;
  ResetDynamicState();
}

void PIDController::Release() {
  held_ = false;
  ResetDynamicState();
}

bool PIDController::IsHeld() const {
  return held_;
}

void PIDController::ResetDynamicState() {
  integral_ = 0.0f;
  last_error_ = 0.0f;
  first_run_ = true;
}

void PIDController::Compute(SignalBroker& broker, float dt) {
  if (held_) {
    return;
  }
  if (!std::isfinite(dt) || dt <= 0.0f) {
    return;
  }

  if (pv_tag_id_ == SignalBroker::kUnmappedTag ||
      cv_tag_id_ == SignalBroker::kUnmappedTag) {
    return;
  }

  // Read process variable
  float pv = broker.GetTag(pv_tag_id_);
  float error = setpoint_ - pv;

  // Proportional Term
  float p_term = kp_ * error;

  // Integral Term with Anti-Windup Clamping
  integral_ += error * dt;
  if (ki_ > 0.0001f || ki_ < -0.0001f) {
    float max_i_contrib = 100.0f;
    float min_i_contrib = -100.0f;
    float i_contrib = ki_ * integral_;

    if (i_contrib > max_i_contrib) {
      integral_ = max_i_contrib / ki_;
    } else if (i_contrib < min_i_contrib) {
      integral_ = min_i_contrib / ki_;
    }
  } else {
    integral_ = 0.0f;
  }
  float i_term = ki_ * integral_;

  // Derivative Term
  if (first_run_) {
    last_error_ = error;
    first_run_ = false;
  }
  float derivative = (error - last_error_) / dt;
  last_error_ = error;
  float d_term = kd_ * derivative;

  // Sum terms
  float output = p_term + i_term + d_term;

  // Clamp output to standardized tag range [0.0, 100.0]
  if (output < 0.0f) {
    output = 0.0f;
  } else if (output > 100.0f) {
    output = 100.0f;
  }

  // Write control variable back to broker
  broker.SetTag(cv_tag_id_, output);
}

int PIDController::GetPvTagId() const {
  return pv_tag_id_;
}

void PIDController::SetPvTagId(int tag_id) {
  if (!IsValidRoutingTagId(tag_id)) {
    tag_id = SignalBroker::kUnmappedTag;
  }
  pv_tag_id_ = tag_id;
}

int PIDController::GetCvTagId() const {
  return cv_tag_id_;
}

void PIDController::SetCvTagId(int tag_id) {
  if (!IsValidRoutingTagId(tag_id)) {
    tag_id = SignalBroker::kUnmappedTag;
  }
  cv_tag_id_ = tag_id;
}

float PIDController::GetSetpoint() const {
  return setpoint_;
}

void PIDController::SetSetpoint(float setpoint) {
  const float next = FiniteOrZero(setpoint);
  // Reset the accumulated state only when the setpoint is ZEROED, which is the
  // condition issue #4002 specifies. An E-stop's only effect that reaches the
  // plant is zeroing these setpoints, and a wound-up integral was holding the
  // analog output at 100% for tens of seconds after the operator commanded a
  // stop.
  //
  // Deliberately NOT `next != setpoint_`. SyncModbusToDCS calls this on every
  // scan whenever the host register differs, so resetting on any change means a
  // host-driven ramp -- or 1-LSB float jitter through the register round-trip --
  // clears the integrator every scan and the loop silently runs P+D only, never
  // closing steady-state offset. If reset-on-change is ever wanted for bumpless
  // transfer on large steps, it needs a deadband, not an equality test.
  if (next == 0.0f && setpoint_ != 0.0f) {
    ResetDynamicState();
  }
  setpoint_ = next;
}

float PIDController::GetKp() const {
  return kp_;
}

void PIDController::SetKp(float kp) {
  kp_ = FiniteOrZero(kp);
}

float PIDController::GetKi() const {
  return ki_;
}

void PIDController::SetKi(float ki) {
  ki_ = FiniteOrZero(ki);
}

float PIDController::GetKd() const {
  return kd_;
}

void PIDController::SetKd(float kd) {
  kd_ = FiniteOrZero(kd);
}
