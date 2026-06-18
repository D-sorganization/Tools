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

// Re-export the server-contract types so existing importers can keep importing
// domain types from one place.
export type {
  AlicatMFCState,
  EventLogEntry,
  ActiveAlarm,
  TuningResult,
  MpcSimResult,
} from "./api/schemas";
