/**
 * Golf Swing Analysis Library
 * Professional-grade biomechanical analysis for golf swing improvement
 *
 * @module @upstreamdrift/golf-analyzer
 */

// Types
export * from "./types";

// Angle calculations
export {
  calculateAngle,
  calculateAngleFromHorizontal,
  calculateHorizontalRotation,
  getMidpoint,
  calculateDistance,
  calculateBodyAngles,
  getIdealAngleRanges,
  isAngleInRange,
  calculateAngleDeviation,
  smoothAngles,
  calculateAngularVelocity,
  detectStanceDirection,
} from "./angleCalculator";

// Phase detection
export {
  detectSwingPhases,
  getPhaseAtFrame,
  calculateTempoQuality,
  type PhaseTransition,
  type PhaseDetectionResult,
} from "./phaseDetector";

// Main analyzer
export { analyzeSwing, quickAnalyze } from "./swingAnalyzer";

// Persistence
export {
  initDatabase,
  saveSession,
  getSession,
  getAnalysisWithFrames,
  getAllSessions,
  deleteSession,
  compareSwings,
  exportSessionToJSON,
  importSessionFromJSON,
  generateSummary,
  getUserSetting,
  setUserSetting,
  clearAllData,
  getStorageStats,
} from "./persistence";

// Report generation
export {
  generateHTMLReport,
  generateCSVReport,
  downloadReport,
  downloadHTMLReport,
  downloadCSVReport,
  downloadJSONExport,
} from "./reportGenerator";

// Version info
export const VERSION = "1.0.0";
export const LIBRARY_NAME = "@upstreamdrift/golf-analyzer";
