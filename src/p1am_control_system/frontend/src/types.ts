/**
 * Shared domain types for the P1AM control-system frontend.
 *
 * Extracted from the App.tsx god component (#3543) so hooks, contexts, and the
 * per-tab components can import them without depending on App.tsx (which created
 * an import cycle: RoutingMatrix imported `RoutingConfig` from `../App`).
 */

export interface PIDConfig {
  pv_tag_id: number;
  cv_tag_id: number;
  setpoint: number;
  kp: number;
  ki: number;
  kd: number;
}

export interface InterlockConfig {
  lolo_limit: number;
  low_limit: number;
  high_limit: number;
  hihi_limit: number;
}

export interface RoutingConfig {
  input_routing: number[];
  output_routing: number[];
  pids: PIDConfig[];
  interlocks: InterlockConfig[];
}

export type NotificationType = "success" | "error" | "info";

export interface NotificationState {
  message: string;
  type: NotificationType;
}

export type TriggerNotification = (
  message: string,
  type: NotificationType,
) => void;

/* ---- Temperature controller (heater) ------------------------------------- *
 * Moved here from TemperatureControl.tsx when that file was split: the same
 * import cycle this module was created to break had reappeared, with
 * useTelemetryStream importing `TemperatureStatus` from a component. The
 * component re-exports these names so existing importers are unaffected.       */

/** Thermocouple type selectable for the heater control. */
export type TcType = "K" | "R";

export interface ThermocoupleChannel {
  tag: string;
  full_scale_c: number;
  label: string;
}

export interface TemperatureConfig {
  type_k: ThermocoupleChannel;
  type_r: ThermocoupleChannel;
  active_tc_type: TcType;
  /** Derived (read-only) from the active channel — see backend computed fields. */
  temp_tag: string;
  temp_full_scale_c: number;
  active_tc_label: string;
  setpoint_min_c: number;
  setpoint_max_c: number;
  deadband_c: number;
  min_on_time_s: number;
  min_off_time_s: number;
  hh_limit_c: number;
  heater_label: string;
}

export interface TemperatureStatus {
  state: "idle" | "armed" | "running" | "tripped";
  permissive: boolean;
  setpoint_c: number;
  measured_temp_c: number;
  relay_on: boolean;
  trips: string[];
  hh_limit_c: number;
  deadband_c: number;
  min_on_time_s: number;
  min_off_time_s: number;
  active_tc_type: TcType;
  active_tc_label: string;
  /** Operator's setpoint from the last session, recalled by the backend on
   * restart (null when none was ever persisted). Used to pre-fill the entry. */
  last_setpoint_c?: number | null;
  /** Latest type-K reading (deg C), regardless of which TC is controlling, so
   * the HMI can show/plot both channels. null/undefined before the first scan. */
  type_k_temp_c?: number | null;
  /** Latest type-R reading (deg C), regardless of which TC is controlling, so
   * the HMI can show/plot both channels. null/undefined before the first scan. */
  type_r_temp_c?: number | null;
  /** True while the deglitch filter is holding the control thermocouple's
   * last-good value through a live dropout — a hint the control sensor is
   * intermittently faulting (a sustained fault escalates to a TC_FAULT trip). */
  control_sensor_holding?: boolean;
  /** P1-04THM open-circuit fail direction. True = high-side (an open reads full
   * scale -> heater shuts off, fail-safe); false = low-side (an open reads cold). */
  burnout_high_side?: boolean;
}

// Re-export the server-contract types so existing importers can keep importing
// domain types from one place.
export type {
  AlicatMFCState,
  EventLogEntry,
  ActiveAlarm,
  TuningResult,
  MpcSimResult,
} from "./api/schemas";
